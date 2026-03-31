from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union, Tuple
from collections import Counter
from ase import Atoms
import numpy as np

from nara.conditioner import get_random_atoms_with_conditioning, Conditioner, random_pos_in_region
from nara.conditioner import check_in_region as cir
from ase.optimize import QuasiNewton, FIRE
from nara.distance import to_new_cell

#### ------------------------------------------------------------------------------------------------------------------
#### ------------------------------------------------------------------------------------------------------------------
class BaseConstraint(ABC):
    '''
    Defining the rule of Constraint classes
    '''

    @abstractmethod
    def __init__(self, *args, **kwargs):
        '''
        Anything,, that you want to save in
        '''
        self.args = args # Just save what they want in tuple :D
        self.kwargs = kwargs # Just save what they want in dictionary :D
        pass

    @abstractmethod
    def check_valid(self) -> bool:
        '''
        Define the validity :D
        :return: bool
        '''
        pass

    @abstractmethod
    def get_random(self, normalize=False):
        '''
        Define the range/method to generate random 'x' of new agent
        :return:
        '''
        pass

class NumConstraint(BaseConstraint):
    def __init__(self, boundary: List[List[int]] = None, ndim: Optional[int] = None):
        '''

        :param boundary: boundary condition of scalar values
        ex) If (x, y) has the boundary condition of -5<=x<=5 & -4<=y<=3,
        put boundary as [[-5, 5], [-4, 3]]

        :param ndim: N dimension
        '''
        # TODO: For the boundary conditions, maybe we need to add other options.
        # TODO: Ex) including boundaries, infinite boundary condition in one side
        self.boundary = boundary
        if ndim is None:
            self.ndim = len(boundary)
        else:
            assert len(boundary) == ndim, "Length of boundary and ndim should be identical"
            self.ndim = int(ndim)

    def check_valid(self, x) -> bool:
        if self.boundary is not None:
            for t_b, t_x in zip(self.boundary, x):
                low, high = t_b
                if low > t_x or high < t_x:
                    return False
        return True

    def get_random(self, normalize:bool = False) -> np.ndarray:
        '''

        :param normalize: if normalize, return the random value between [-1/2, 1/2].
                         else, random value between the boundary condition will be returned
        :return: random value(s)
        '''
        res_eps = []
        epsilon = np.random.random(self.ndim) - 0.5

        if normalize:
            return epsilon

        if self.boundary is not None:
            for bc, e in zip(self.boundary, epsilon):
                new_e = e*(np.abs(bc[1]-bc[0])) + np.mean(bc)
                res_eps.append(new_e)
        return np.array(res_eps, dtype=float)


class ASEConstraint(BaseConstraint):
    def __init__(self,
                 motif: Atoms,
                 addatoms_info: Dict[str, int],
                 region: Optional[List[Union[float, None]]] = None,
                 initial_region: Optional[List[Union[float, None]]] = None,
                 set_region_as_prohibited: bool = False,
                 min_bondinfo: Optional[Union[float, Dict[Tuple[str, str], float]]] = None,
                 mic = True,
                 ):
        '''

        :param motif: input structures. Substrate for interface structures, or empty unit cell is also available
        :param addatoms_info: dictionary, key = elements (str) & value = the number of atoms (int)
        :param region: List[x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]. If None, any places in unit cell is allowed to atoms
        :param initial_region: For defining initial distribution region (upper one is for constrained region)
        :param set_region_as_prohibited: if True, atoms will not be distributed to the region
        :param min_bondinfo: single scalar value or dictionary (which has pairwise information)
            ** If number (Default, 0.5Å)
             : uniform bond length minimum limit to all bonds (in Angstrom unit)
            ** If dictionary
             : key = elements-wise selection of bonds (tuple of string) / value = minimum bond length (float)
            ** If None
             : Not checking the bond length at all
        :param mic: if True, considering minimum image convention while measuring the distances
        '''
        assert isinstance(motif, Atoms), "motif should be ase.Atoms instance"
        self.motif = motif
        self.addatoms_info = addatoms_info
        self.mic = mic

        if region is None:
            self.region = [None]*6
        else:
            assert len(region) == 6, "region should be defined in list with 6 scalars"
            self.region = region

        if initial_region is None:
            self.initial_region = self.region
        else:
            assert len(region) == 6, "region should be defined in list with 6 scalars"
            self.initial_region = initial_region

        self.set_region_as_prohibited = set_region_as_prohibited

        if min_bondinfo is None:
            self.min_bondinfo = None
        elif isinstance(min_bondinfo, (int, float, dict)):
            self.min_bondinfo = min_bondinfo
        elif isinstance(min_bondinfo, str):
            self.min_bondinfo = min_bondinfo.lower()
        else:
            raise NotImplementedError("Other than str/number/dictionary type for min_bondinfo is not supported.")

        self.conditioner = Conditioner(min_bondinfo = self.min_bondinfo,
                                       region = self.region,
                                       set_region_as_prohibited = self.set_region_as_prohibited,
                                       k_boundary = 1.0,
                                       k_repulsion = 1.0,
                                       max_force_norm = 10.0,
                                       mic = self.mic)

    def check_region(self, x:Atoms, after_wrap=True) -> bool:
        x_copy = x.copy()
        if after_wrap:
            x_copy.wrap()
        for atom in x_copy:
            if atom.tag != 3:
                continue
            if (self.region is not None) and (cir(atom.position, self.region) == self.set_region_as_prohibited):
                return False
        return True

    def check_valid(self, x:Atoms, etol=1) -> bool:
        '''
        Current implementation:
            1) elements counting (Compulsory)
            2) minimum_bonds counting (Optional)
        '''
        # 1) elements counting
        template_natoms = Counter(self.motif.get_chemical_symbols()) if len(self.motif)!=0 else dict()
        current_natoms = Counter(x.get_chemical_symbols()) if len(x)!=0 else dict()

        for _key in current_natoms:
            current_natoms[_key] -= template_natoms.get(_key, 0)

        for _key in self.addatoms_info:
            if current_natoms.get(_key, 0) != self.addatoms_info.get(_key, 0):
                return False

        # 2) minimum_bonds counting + region check
        if self.min_bondinfo is not None:
            x_copy = x.copy()
            x_copy.calc = self.conditioner
            if x_copy.get_potential_energy() > etol:
                return False
        return True # After all tests, return True

        # # 2) minimum_bonds counting
        # if self.min_bondinfo is not None:
        #     if "all" in self.min_bondinfo:
        #         Dmat = x.get_all_distances() + np.diag([np.inf]*len(x))
        #         if Dmat < self.min_bondinfo["all"]:
        #             return False
        #     else:
        #         for pair in self.min_bondinfo:
        #             x1 = x[np.array(x.get_chemical_symbols(), dtype=str) == pair[0]]
        #             x2 = x[np.array(x.get_chemical_symbols(), dtype=str) == pair[1]]
        #             Dmat = get_distances(x1.get_positions(), x2.get_positions(), pbc=x.pbc, cell=x.cell)[1]
        #             if pair[0] == pair[1]:
        #                 assert len(x1) == len(x2), "You cannot see this. If you encounter this error, report to developer"
        #                 Dmat += np.diag([np.inf]*len(x1))
        #             if np.min(Dmat) < self.min_bondinfo[pair]:
        #                 return False
        # return True # After all tests, return True

    def get_random(self, normalize:bool = False, return_aseobj = False, max_iter=None):
        if normalize:
            # random walk!
            # if multiplied by dr, it will be identical to Basin hopping's perturbation
            n_total = 0
            for key, value in self.addatoms_info.items():
                n_total += int(value)
            if len(self.motif)==0:
                return np.random.uniform(-1, 1, (n_total, 3))
            else:
                return np.vstack((np.zeros((len(self.motif), 3)), np.random.uniform(-1, 1, (n_total, 3))))

        m1 = get_random_atoms_with_conditioning(
            base_atoms=self.motif,
            add_info = self.addatoms_info,
            min_bondinfo = self.min_bondinfo,
            region = self.initial_region,
            set_region_as_prohibited = self.set_region_as_prohibited,
            mic = self.mic,
            max_step = 1000,
            max_iter = max_iter,
            verbose = False,
            trajectory = None,
            optimizer = QuasiNewton,
            tol = 1e-4,
        )
        m1.calc = self.motif.calc
        # m1.calc = copy.deepcopy(self.motif.calc)
        ### |___ Deepcopying is for copying the calculator, but for the GAP Potential obj using quippy,
        ### |___ several deepcopying raises "double free or corruption (!prev) error"
        ### |    So I don't recommend to use deepcopying atoms or calculator

        return m1 if return_aseobj else m1.get_positions()

        ### When using Mod_Hookean (deprecated)
        # NCS = []
        # if m1.constraints:
        #     NCS += m1.constraints # m1.constraints

        # _ind = len(self.motif), len(m1)
        # for i in range(*_ind, 1):
        #     at = m1[i]
        #     for _ri, _rplane in zip(range(6), [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]):
        #         rval = self.region[_ri]
        #         if rval is not None:
        #             abcd = list(_rplane) + [-rval*np.sum(_rplane)]
        #             NCS.append(Hookean(i, a2 = abcd, k = 20))

        # for i in range(len(m1)):
        #     if self.min_bondinfo is None:
        #         break
        #     assert isinstance(self.min_bondinfo, dict), "If min_bondinfo is not None, it should be dictionary"
        #     if self.min_bondinfo.get("all") is not None:
        #         all_minbl = self.min_bondinfo["all"]
        #         for j in range(i+1, len(m1), 1):
        #             NCS.append(Mod_Hookean(a1=i, a2=j, rt=all_minbl, k=30))
        #     else:
        #         for j in range(i+1, len(m1), 1):
        #             ijsym = m1[i].symbol, m1[j].symbol
        #             if ijsym in self.min_bondinfo:
        #                 minbl = self.min_bondinfo[ijsym]
        #             elif ijsym[::-1] in self.min_bondinfo:
        #                 minbl = self.min_bondinfo[ijsym[::-1]]
        #             else:
        #                 continue
        #             NCS.append(Mod_Hookean(a1=i, a2=j, rt=minbl, k=30))

        # m1.set_constraint(NCS)
        # return m1 if return_aseobj else m1.get_positions()
