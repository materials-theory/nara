import numpy as np
from typing import List, Dict, Optional, Union
from ase import Atoms, Atom
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE, QuasiNewton
from ase.data import atomic_numbers
from ase.data import vdw_radii, covalent_radii

from nara.get_d_pos import get_fixed_atoms_index
from nara.distance import to_new_cell, get_distances_torch
from ase.neighborlist import primitive_neighbor_list

import torch

_eps = 1e-8

def get_random_atoms_with_conditioning(
        base_atoms:Atoms,
        add_info:Dict[str, int],
        min_bondinfo:Union[str, float, Dict[str, float]],
        region:Optional[List[Union[None,float]]] = None,
        initial_region:Optional[List[Union[None,float]]] = None,
        set_region_as_prohibited:bool = False,
        mic:bool = True,
        max_step:int = 1000,
        max_iter:Optional[int] = 10,
        verbose:bool = False,
        trajectory:Optional[str] = None,
        optimizer = QuasiNewton,
        tol = 1e-4,
) -> Atoms:
    '''
    User level random structure generator. (enables min_bondinfo with string value)

    :param base_atoms : Host atoms, with cell
    :param add_info : dict(symbol:number of atoms). Ex add_info={"Cu":3, "O":5}
    :param min_bondinfo : float (uniform min_dist) or string (vdw/covalent) or dictionary (pair:r)
    :param region : box defined by [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]
    :param initial_region : If you want to specify the initial region, use this
     |___ ex. randomly distribute in small region, and allow larger regions with conditioning
    :param set_region_as_prohibited : if True, the box is set as prohibited region
    :param mic : considering minimum image convention if True
    :param max_step : maximum number of steps for local optimization
    :param max_iter : if the generated structure doesn't meet the condition even after the optimization, new random structure is generated, until max_iter.
                if None, return structures even though if it doesn't meet the condition
    :param verbose : show optimizer's ongoing status if True else silent
    :param optimizer : ase.optimizer.optimizer should be used (ex. BFGS, FIRE, QuasiNewton, ...)
    :param tol : if abs(energy) < tol, it succeeded.
    '''
    return_as_it_is = True if max_iter is None else False
    max_iter = 1 if max_iter is None else max_iter
    if mic:
        assert np.any(base_atoms.pbc), f"you enabled mic=True but base_atoms doesn't have pbc:{base_atoms.pbc}"
    else:
        assert not np.all(base_atoms.pbc), f"you disabled mic, but base_atoms have pbc:{base_atoms.pbc}"

    if region is not None:
        if len(region) != 6:
            raise ValueError("region should be list length of 6")

        for i in range(3):
            _min, _max = region[i*2], region[i*2+1]
            if (_min is None) or (_max is None):
                assert (_min is None) and (_max is None), "Region for certain axis should be both set or both not set"

    else:
        region = [None]*6

    if initial_region is None:
        initial_region = region # itniail & allow region is identical

    if isinstance(min_bondinfo, str):
        _min_bondinfo = min_bondinfo.lower()
        syms = np.unique(list(base_atoms.get_chemical_symbols()) + list(add_info))
        min_bondinfo = {}
        for i, sym_i in enumerate(syms):
            for sym_j in syms[i:]:
                if _min_bondinfo.startswith("vdw"):
                    r = vdw_radii[atomic_numbers[sym_i]] + vdw_radii[atomic_numbers[sym_j]]
                elif _min_bondinfo.startswith("cov"):
                    r = covalent_radii[atomic_numbers[sym_i]] + covalent_radii[atomic_numbers[sym_j]]
                else:
                    raise ValueError("min_bondinfo should be vdw or cov if string")
                min_bondinfo[f"{sym_i}-{sym_j}"] = r

    elif isinstance(min_bondinfo, (float, int)):
        pass # good
    elif min_bondinfo is None:
        min_bondinfo = 0
    else:
        assert isinstance(min_bondinfo, dict), "min_bondinfo should be either string/number/dictionary"

    for trial in range(max_iter):
        base_atoms = to_new_cell(base_atoms)
        model_trial = random_pos_in_region(
            base_atoms,
            add_info = add_info,
            region = initial_region,
            set_region_as_prohibited = set_region_as_prohibited,
            return_aseobj = True
        )

        conditioner = Conditioner(
            min_bondinfo= min_bondinfo,
            region = region,
            set_region_as_prohibited = set_region_as_prohibited,
            k_boundary = 1.0,
            k_repulsion = 2.0,
            max_force_norm = None,
            mic = mic,
        )

        model_trial.calc = conditioner

        logfile = "-" if verbose else None
        opt = optimizer(model_trial, trajectory = trajectory, logfile = logfile)
        opt.run(steps = max_step, fmax = tol/10)
        opt.logfile.close()
        # if np.max(np.linalg.norm(model_trial.get_forces(), axis=1)) <= tol:
        if model_trial.get_potential_energy() <= tol:
            done = True
            break
        else:
            done = False

    if done or return_as_it_is:
        return model_trial.copy()
    else:
        raise RuntimeError("Failed generating random structure. Reduce min_bondinfo or increase the region")


def check_in_region(position, region = None):
    '''
    Check if the position is in the region or not

    :param position: (x, y, z)
    :param region: (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) or List of regions
    :return: bool (True indicates that the 'position' is in the region)
    '''
    assert len(position) == 3
    x, y, z = position

    if region is None:
        return False # not overlapped!
    else:
        if isinstance(region[0], list):
            overlapped = False
            for region in region:
                if check_in_region(position, region):
                    overlapped = True # overlap in this region
                    break
            return overlapped
        else:
            assert len(region) == 6
            if all([x is None for x in region]):
                return False

    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = region
    chpr = [(x_lo, x_hi, x), (y_lo, y_hi, y), (z_lo, z_hi, z)]

    # Single prohibited region
    for lo, hi, value in chpr:
        if lo is None and hi is None:
            pass
        elif lo is None and hi is not None:
            if value > hi:
                return False
        elif lo is not None and hi is None:
            if value < lo:
                return False
        else:  # both selected
            if value < lo or value > hi:
                return False
    return True

def random_pos_in_region(base_atoms, add_info, region = None, set_region_as_prohibited = False, return_aseobj = False):
    '''
    randomly distribute atom in given region (if region is None, distribute atoms to all region)
    if a lattice vector is not located in x axis, we may need to align it, by using to_new_cell function
    '''
    if region is None:
        region = [None]*6

    model = base_atoms.copy()
    fixed_index = get_fixed_atoms_index(model)

    for atom in model:
        if atom.index in fixed_index:
            atom.tag = 1
        else:
            atom.tag = 2

    a, b, c = model.cell.cellpar()[:3]
    x_reg2, y_reg2, z_reg2 = None, None, None

    if not set_region_as_prohibited:
        minmax_default = [0, a, 0, b, 0, c]
        x_min, x_max, y_min, y_max, z_min, z_max = [r if r is not None else d for r, d in zip(region, minmax_default)]
        x_reg1, y_reg1, z_reg1 = (x_min, x_max), (y_min, y_max), (z_min, z_max)
    else:
        _xmin, _xmax, _ymin, _ymax, _zmin, _zmax = region
        if (_xmin is None) and (_xmax is None):
            x_reg1 = 0, a
        else:
            assert (_xmin is not None) and (_xmax is not None)
            x_reg1 = 0, _xmin
            x_reg2 = _xmax, a
        if (_ymin is None) and (_ymax is None):
            y_reg1 = 0, b
        else:
            assert (_ymin is not None) and (_ymax is not None)
            y_reg1 = 0, _ymin
            y_reg2 = _ymax, b
        if (_zmin is None) and (_zmax is None):
            z_reg1 = 0, c
        else:
            assert (_zmin is not None) and (_zmax is not None)
            z_reg1 = 0, _zmin
            z_reg2 = _zmax, c

    def random_coords(size, reg1, reg2=None):
        if reg2 is None:
            return np.random.uniform(reg1[0], reg1[1], size=size)
        else:
            choices = np.random.choice([0,1], size=size)
            mins, maxs = np.array([reg1, reg2], dtype=float).T
            return np.random.uniform(mins[choices], maxs[choices], size=size)

    def get_x_coords(size):
        return random_coords(size, x_reg1, x_reg2)

    def get_y_coords(size):
        return random_coords(size, y_reg1, y_reg2)

    def get_z_coords(size):
        return random_coords(size, z_reg1, z_reg2)

    for sym, num in add_info.items():
        Xs = get_x_coords(num)
        Ys = get_y_coords(num)
        Zs = get_z_coords(num)
        coords = np.vstack([Xs, Ys, Zs]).T
        for _ in coords:
            model.append(Atom(symbol=sym, position=_, tag=3))
    return model if return_aseobj else model.get_positions()


### TODO: boost-up the Conditioner calculator via torch (not done yet)
# class Conditioner_torch(Calculator):
#     implemented_properties = ['energy', 'forces']
#     def __init__(
#             self,
#             min_bondinfo:Union[Dict,float,str],
#             region:Optional[List[Union[None,float]]] = None,
#             set_region_as_prohibited:bool = False,
#             k_boundary:float = 1.0,
#             k_repulsion:float = 1.0,
#             # max_force_norm:Optional[float] = None,
#             mic:bool = True,
#             device:Optional[torch.device] = None,
#             dtype:str = "float64",
#             work_on_relax_atoms_too: bool = False,
#             **kwargs
#     ):
#         """
#         :param min_bondinfo : float or dict
#         :param region : list of length 6, [x_min, x_max, y_min, y_max, z_min, z_max]
#         :param set_region_as_prohibited : bool
#         :param k_boundary : float, spring constant
#         :param k_repulsion : float, spring constant
#         :param max_force_norm : None or float
#         """
#         super().__init__(**kwargs)
#         if isinstance(min_bondinfo, str):
#             min_bondinfo = min_bondinfo.lower()
#         self.min_bondinfo = min_bondinfo
#         self.region = [None]*6 if region is None else region
#         self.set_region_as_prohibited = set_region_as_prohibited
#         self.k_boundary = k_boundary
#         self.k_repulsion = k_repulsion
#         # self.max_force_norm = max_force_norm
#         self.mic = mic
#
#         if region is not None:
#             for i in range(3):
#                 _min, _max = region[i * 2], region[i * 2 + 1]
#                 if (_min is None) or (_max is None):
#                     assert (_min is None) and (_max is None),\
#                         "Region for certain axis should be both set or both not set"
#
#         if device is None:
#             self.device = torch.device("cpu")
#         elif isinstance(device, str):
#             self.device = torch.device(device)
#         else:
#             assert isinstance(device, torch.device)
#             self.device = device
#
#         if isinstance(dtype, str):
#             self.dt = torch.float64 if dtype.lower()=="float64" else torch.float32
#         else:
#             assert isinstance(dtype, torch.dtype)
#             self.dt = dtype
#
#         self.work_on_relax_atoms_too = work_on_relax_atoms_too
#         if work_on_relax_atoms_too and region is not None:
#             print("Warning: self.region should contain the atoms with tag==1 & 2")
#
#     @torch.no_grad
#     def calculate(self, atoms, properties, system_changes=all_changes):
#         Calculator.calculate(self, atoms, properties, system_changes)
#         _pos = atoms.get_positions()
#         n_atoms = len(atoms)
#         pos = torch.tensor(_pos, dtype=self.dt, device=self.device)
#         tags = torch.tensor([atom.tag for atom in atoms], dtype=torch.long, device=self.device)
#         energy = torch.tensor(0.0, dtype=self.dt, device=self.device)
#         forces = torch.zeros_like(pos, dtype=self.dt, device=self.device)
#
#         # (1) Region
#         for axis in range(3):
#             coord = pos[:, axis]
#             lower = self.region[axis * 2]
#             upper = self.region[axis * 2 + 1]
#
#             if (lower is None) and (upper is None):
#                 continue
#
#             if not self.set_region_as_prohibited:
#                 # Atoms should be located inside the region
#                 mask = coord < lower
#                 if not self.work_on_relax_atoms_too:
#                     mask = mask & (tags == 3)
#                 disp = lower - coord[mask]
#                 forces[mask, axis] += self.k_boundary * disp
#                 energy += 0.5 * self.k_boundary * torch.sum(disp ** 2)
#
#                 mask = coord > upper
#                 if not self.work_on_relax_atoms_too:
#                     mask = mask & (tags == 3)
#                 disp = coord[mask] - upper
#                 forces[mask, axis] -= self.k_boundary * disp
#                 energy += 0.5 * self.k_boundary * torch.sum(disp ** 2)
#             else:
#                 # Atoms should not be located inside the region
#                 mask_inside = (coord > lower) & (coord < upper)
#                 if not self.work_on_relax_atoms_too:
#                     mask_inside = mask_inside & (tags == 3)
#
#                 if mask_inside.any():
#                     d_lower = coord[mask_inside] - lower
#                     d_upper = upper - coord[mask_inside]
#                     mask_to_lower = d_lower < d_upper
#                     mask_to_upper = ~mask_to_lower
#                     idx = mask_inside.nonzero(as_tuple=False).squeeze()
#                     if idx.ndim == 0:
#                         idx = idx.unsqueeze(0)
#                     if mask_to_lower.any():
#                         idx_lower = idx[mask_to_lower]
#                         disp = d_lower[mask_to_lower]
#                         forces[idx_lower, axis] -= self.k_boundary * disp
#                         energy += 0.5 * self.k_boundary * torch.sum(d_lower[mask_to_lower] ** 2)
#                     if mask_to_upper.any():
#                         idx_upper = idx[mask_to_upper]
#                         disp = d_upper[mask_to_upper]
#                         forces[idx_upper, axis] += self.k_boundary * disp
#                         energy += 0.5 * self.k_boundary * torch.sum(d_upper[mask_to_upper] ** 2)
#
#         # (2) Repulsion force
#         _cell = atoms.cell.array
#         cell = torch.tensor(_cell, device=self.device, dtype=self.dt)
#
#         if isinstance(self.min_bondinfo, str):
#             syms = np.unique(atoms.get_chemical_symbols())
#             # 여기서 바로 self.min_bondinfo = {} 수정해 버리면, 이후 다른 원소를 갖는 구조가 들어왔을 때 해당 원소가 반영되지 않음
#             # 근데 문제는 optimizer에서 매 step마다 새로 min_bondinfo를 계산해야 하니... 아주 조금의 효율성 loss는 있을 듯
#             min_bondinfo = {}
#             for i, sym_i in enumerate(syms):
#                 for sym_j in syms[i:]:
#                     if self.min_bondinfo.startswith("vdw"):
#                         _r = vdw_radii[atomic_numbers[sym_i]] + vdw_radii[atomic_numbers[sym_j]]
#                     elif self.min_bondinfo.startswith("cov"):
#                         _r = covalent_radii[atomic_numbers[sym_i]] + covalent_radii[atomic_numbers[sym_j]]
#                     else:
#                         raise ValueError("min_bondinfo should be vdw or cov if string")
#                     min_bondinfo[f"{sym_i}-{sym_j}"] = _r
#         else:
#             min_bondinfo = self.min_bondinfo
#
#         if isinstance(min_bondinfo, dict):
#             _r_cut = np.max(list(min_bondinfo.values()))
#         else:
#             assert isinstance(min_bondinfo, (float, int))
#             _r_cut = min_bondinfo
#
#         Is, Js, Vecs, Ds, _ = get_distances_torch(
#             pos = pos,
#             cell = cell,
#             return_vec = True,
#             return_all_neighborlist = True,
#             r_cut_for_neighbor = _r_cut)
#
#         syms = atoms.get_chemical_symbols()  # 리스트
#         if isinstance(self.min_bondinfo, dict):
#             thresholds_list = []
#             for i, j in zip(Is, Js):
#                 i, j = i.item(), j.item()
#                 key_ij = f"{syms[i]}-{syms[j]}"
#                 key_ji = f"{syms[j]}-{syms[i]}"
#                 if key_ij in self.min_bondinfo:
#                     thresholds_list.append(self.min_bondinfo[key_ij])
#                 elif key_ji in self.min_bondinfo:
#                     thresholds_list.append(self.min_bondinfo[key_ji])
#                 else:
#                     thresholds_list.append(torch.inf)
#             thresholds = torch.tensor(thresholds_list, device=self.device, dtype=self.dt)
#         elif isinstance(self.min_bondinfo, (float, int)):
#             thresholds = torch.full_like(Ds, float(self.min_bondinfo))
#         elif isinstance(self.min_bondinfo, str):
#             raise ValueError("min_bondinfo of type str should be processed to a dictionary already.")
#         else:
#             raise ValueError("Unexpected type for min_bondinfo.")
#
#         bond_mask = Ds < thresholds
#         delta = thresholds - Ds
#
#         energy += 0.5 * self.k_repulsion * torch.sum((delta[bond_mask]) ** 2)
#         f_mag = self.k_repulsion * delta  # (n_bonds,)
#         f_dir = Vecs / Ds.unsqueeze(1)
#         f_vec = f_mag.unsqueeze(1) * f_dir  # (n_bonds, 3)
#
#         f_vec_masked = torch.zeros_like(f_vec)
#         f_vec_masked[bond_mask] = f_vec[bond_mask]
#
#         apply_i = (tags[Is] == 3)
#         apply_j = (tags[Js] == 3)
#         if apply_i.any():
#             forces.index_add_(0, Is[apply_i], -f_vec_masked[apply_i])
#         if apply_j.any():
#             forces.index_add_(0, Js[apply_j], f_vec_masked[apply_j])
#         self.results = {'energy': energy.item(), 'forces': forces.cpu().numpy()}

class Conditioner(Calculator):
    implemented_properties = ['energy', 'forces']
    def __init__(
            self,
            min_bondinfo:Union[Dict,float,str],
            region:Optional[List[Union[None,float]]] = None,
            set_region_as_prohibited:bool = False,
            k_boundary:float = 1.0,
            k_repulsion:float = 1.0,
            max_force_norm:Optional[float] = None,
            mic:bool = True,
            work_on_relax_atoms_too:bool = False,
            **kwargs
    ):
        """
        :param min_bondinfo : float or dict
        :param region : list of length 6, [x_min, x_max, y_min, y_max, z_min, z_max]
        :param set_region_as_prohibited : bool
        :param k_boundary : float, spring constant
        :param k_repulsion : float, spring constant
        :param max_force_norm : None or float
        """
        super().__init__(**kwargs)
        if isinstance(min_bondinfo, str):
            min_bondinfo = min_bondinfo.lower()
        self.min_bondinfo = min_bondinfo
        self.region = [None]*6 if region is None else region
        self.set_region_as_prohibited = set_region_as_prohibited
        self.k_boundary = k_boundary
        self.k_repulsion = k_repulsion
        self.max_force_norm = max_force_norm
        self.mic = mic

        if self.mic:
            for i in range(3):
                _c1 = (region[i*2] is None) and (region[i*2+1] is None)
                _c2 = (region[i*2] is not None) and (region[i*2+1] is not None)
                assert _c1 or _c2, "If mic=True, region for that axis should be both set or both not set"

        self.work_on_relax_atoms_too = work_on_relax_atoms_too
        if work_on_relax_atoms_too and region is not None:
            print("Warning: self.region should contain the atoms with tag==1 & 2")

    def calculate(self, atoms, properties, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        positions = atoms.get_positions()
        n_atoms = len(positions)

        forces = np.zeros((n_atoms, 3))
        energy = 0.0

        for i, atom in enumerate(atoms):
            if (atom.tag != 3) and (not self.work_on_relax_atoms_too):
                continue
            for axis in range(3):
                coord = positions[i, axis]
                lower = self.region[axis * 2]
                upper = self.region[axis * 2 + 1]

                if not self.set_region_as_prohibited:
                    if lower is not None and coord < lower:
                        disp = lower - coord
                        forces[i, axis] += self.k_boundary * disp
                        energy += 0.5 * self.k_boundary * disp ** 2
                    if upper is not None and coord > upper:
                        disp = coord - upper
                        forces[i, axis] -= self.k_boundary * disp
                        energy += 0.5 * self.k_boundary * disp ** 2
                else:
                    if (lower is not None) and (upper is not None) and (lower < coord < upper):
                        dist_lower = coord - lower # positive
                        dist_upper = upper - coord # positive
                        disp = -dist_lower if dist_lower < dist_upper else dist_upper
                        forces[i, axis] += self.k_boundary * disp
                        energy += 0.5 * self.k_boundary * np.abs(disp) ** 2

        if isinstance(self.min_bondinfo, str):
            syms = np.unique(atoms.get_chemical_symbols())
            min_bondinfo = {}
            for i, sym_i in enumerate(syms):
                for sym_j in syms[i:]:
                    if self.min_bondinfo.startswith("vdw"):
                        _r = vdw_radii[atomic_numbers[sym_i]] + vdw_radii[atomic_numbers[sym_j]]
                    elif self.min_bondinfo.startswith("cov"):
                        _r = covalent_radii[atomic_numbers[sym_i]] + covalent_radii[atomic_numbers[sym_j]]
                    else:
                        raise ValueError("min_bondinfo should be vdw or cov if string")
                    min_bondinfo[f"{sym_i}-{sym_j}"] = _r
        else:
            min_bondinfo = self.min_bondinfo

        if isinstance(min_bondinfo, dict):
            _bondinfo = {tuple(k.split("-")): v for k, v in min_bondinfo.items()}
        else:
            _bondinfo = min_bondinfo

        if self.mic:
            assert np.any(atoms.pbc), "pbc is all False, while you set mic=True"
            pbc = atoms.pbc
        else:
            pbc = [False, False, False]

        Is, Js, Vecs, Ds = primitive_neighbor_list(
            "ijDd",
            pbc = pbc,
            cell = atoms.cell,
            positions = atoms.get_positions(),
            cutoff = _bondinfo,
            numbers = atoms.get_atomic_numbers(),
            self_interaction = True,
            # But how about in this case? It cannot be optimized
            # At least energies will be updated
            use_scaled_positions = False,
        )
        mask_self_exclude = ~((Is==Js)&(Ds<_eps))
        Is = Is[mask_self_exclude]
        Js = Js[mask_self_exclude]
        Vecs = Vecs[mask_self_exclude]
        Ds = Ds[mask_self_exclude]

        for i, j, r_vec, r in zip(Is, Js, Vecs, Ds):
            if isinstance(min_bondinfo, dict):
                _keys = atoms[i].symbol, atoms[j].symbol
                key_ij = "-".join(_keys)
                key_ji = "-".join(_keys[::-1])
                threshold = min_bondinfo[key_ij] if key_ij in min_bondinfo else min_bondinfo[key_ji]
            else:
                threshold = min_bondinfo
            delta = threshold - r
            energy += 0.5 * self.k_repulsion * delta ** 2
            f_mag = self.k_repulsion * delta
            f_vec = (f_mag / r) * r_vec
            if atoms[i].tag == 3:
                forces[i] -= f_vec
            if atoms[j].tag == 3:
                forces[j] += f_vec

        if self.max_force_norm is not None:
            for i in range(n_atoms):
                if atoms[i].tag != 3:
                    continue
                norm = np.linalg.norm(forces[i])
                if (norm > _eps) and (self.max_force_norm is not None):
                    forces[i] = self.max_force_norm * np.tanh(norm / self.max_force_norm) * (forces[i] / norm)

        self.results = {'energy': energy, 'forces': forces}