import numpy as np
from typing import List, Optional, Any, Dict

from ase import Atoms
from ase.io.extxyz import read_xyz
# from ase.geometry import get_distances

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch_scatter import scatter

_eps = 1e-8

def to_new_cell(model: Atoms, decimals=13):
    from math import cos, sin, tan, sqrt
    new_cell = np.zeros((3, 3))
    a, b, c, alpha, beta, gamma = model.cell.cellpar()
    alpha, beta, gamma = np.array([alpha, beta, gamma]) * np.pi / 180
    new_cell[0, 0] = a
    new_cell[1] = b * cos(gamma), b * sin(gamma), 0
    new_cell[2, 0] = c * cos(beta)
    new_cell[2, 1] = c * (cos(alpha) - cos(beta) * cos(gamma)) / (sin(gamma))
    new_cell[2, 2] = sqrt(c ** 2 - new_cell[2, 0] ** 2 - new_cell[2, 1] ** 2)
    if decimals != None:
        new_cell = new_cell.round(decimals=decimals)
    tmp = model.copy()
    tmp.set_cell(new_cell)
    tmp.set_scaled_positions(model.get_scaled_positions())
    return tmp

def minimize_xz_tilt(
        model: Atoms,
        verbose=False,
        max_na=10,
        max_nb=10,
        wrap=True):
    '''
    First, using to_new_cell, change the cell to

    x   0  0
    xy  y  0
    xz yz  z

    Then here, c vector can be written in the vector, n1a + n2b + c.
    when n1 and n2 is integer.

    Oh I also found that ASE has minimize_tilt function. Looks similar..
    '''
    from math import cos, sin, tan, sqrt

    temp_model = to_new_cell(model)
    X, Y, Z, XY, XZ, YZ = temp_model.cell[(0, 1, 2, 1, 2, 2), (0, 1, 2, 0, 0, 1)]
    original_dist = np.linalg.norm((XZ, YZ))

    na_grid = np.array(range(-max_na, max_na + 1))
    nb_grid = np.array(range(-max_nb, max_nb + 1))
    na_GRIDS, nb_GRIDS = np.meshgrid(na_grid, nb_grid)

    Coords_x = XZ + na_GRIDS * X + nb_GRIDS * XY
    Coords_y = YZ + nb_GRIDS * Y
    Dists = np.linalg.norm([Coords_x, Coords_y], axis=0)
    Minimum = np.where(Dists == np.min(Dists))
    sel_na, sel_nb = na_grid[Minimum[1][0]], nb_grid[Minimum[0][0]]

    new_cell = temp_model.cell.copy()
    new_cell[2] += sel_na * new_cell[0] + sel_nb * new_cell[1]
    new_model = temp_model.copy()
    new_model.set_cell(new_cell, scale_atoms=False)  # We cannot use scale_atoms here!!! --> Atoms will break

    if wrap:
        new_model.wrap()  # We can wrap atoms here

    return new_model


@torch.no_grad()
def c2cpar(cell, eps=1e-15) -> List:
    assert isinstance(cell, torch.Tensor)
    pi = torch.pi
    cellpar = [torch.linalg.norm(c) for c in cell]
    for i, j in [1, 2], [0, 2], [0, 1]:
        prod = cellpar[i] * cellpar[j]
        if prod > eps:
            cosine = cell[i] @ cell[j] / prod
            cosine = torch.clamp(cosine, -1.0, 1.0)
            cellpar.append(torch.rad2deg(torch.acos(cosine)))
        else:
            cellpar.append(torch.tensor(90.0))
    return cellpar

def get_distances_torch(pos: torch.Tensor,
                        cell: torch.Tensor = None,
                        return_vec=False,
                        max_shift=1,  # use 3x3 supercell
                        ghost_skin_margin=3,
                        return_all_neighborlist=False,
                        r_cut_for_neighbor=5.0,
                        return_ALL=False
                        ):
    '''
    Instead of using ASE's get_distances function with mic.
    (when return_all_neighborlist, this can be used instead of primitive_neighbor_list)
    r_cut_for_neighbor = 5.0

    When return_all, it will return both NxN distance matrix & neighborlists

    Here, unlike ase.geometry.get_distances function, pos2 is not supported.
    '''
    assert isinstance(pos, torch.Tensor), "pos should be torch.Tensor"
    assert isinstance(cell, torch.Tensor), "cell should be torch.Tensor instance. Use torch.tensor(atoms.cell.array)"

    if return_ALL:
        return_all_neighborlist = False  # temporary -> It will be changed later
        return_vec = True

    a, b, c, alpha, beta, gamma = c2cpar(cell)
    crit = r_cut_for_neighbor if return_all_neighborlist else ghost_skin_margin
    if (a <= crit) or (b <= crit) or (c <= crit):
        while (a * max_shift <= crit) or (b * max_shift <= crit) or (c * max_shift <= crit):
            max_shift += 1

    # 1. IF ab plane of cell is not located in xy plane, change it
    if torch.tensor((cell[0, 1], cell[0, 2], cell[1, 2])).abs().max() < _eps:
        new_cell = cell  # pass!
    else:
        pi = torch.pi
        alpha = alpha * pi / 180
        beta = beta * pi / 180
        gamma = gamma * pi / 180

        new_cell = torch.zeros([3, 3], device=pos.device, dtype=pos.dtype)
        new_cell[0, 0] = a
        new_cell[1, 0] = b * torch.cos(gamma)
        new_cell[1, 1] = b * torch.sin(gamma)
        new_cell[2, 0] = c * torch.cos(beta)
        new_cell[2, 1] = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / (torch.sin(gamma))
        new_cell[2, 2] = torch.sqrt(c ** 2 - new_cell[2, 0] ** 2 - new_cell[2, 1] ** 2)
        new_cell = torch.round(new_cell, decimals=8)

    # 2. Check whether xz_minimization is needed or not
    _a, _b, _c = new_cell
    try:
        _x0y0 = torch.linalg.solve(torch.stack([_a[:2], _b[:2]], dim=1), _c[:2])
        requires_xz_min = None
    except RuntimeError:
        # Should we continue in this case?
        requires_xz_min = True

    if requires_xz_min is None:
        x0, y0 = _x0y0[0].item(), _x0y0[1].item()
        w1 = ghost_skin_margin / torch.linalg.norm(_a)
        w2 = ghost_skin_margin / torch.linalg.norm(_b)
        if (x0 >= w1 - 1) and (x0 <= 1 - w1) and (y0 >= w2 - 1) and (y0 <= 1 - w2):
            requires_xz_min = False
        else:
            requires_xz_min = True

    if requires_xz_min:
        # use ASE only for here...
        model = Atoms(
            positions=pos.cpu().detach().numpy(),
            symbols=["H"] * len(pos),  # temporary
            cell=cell.cpu().detach().numpy(),
            pbc=True
        )
        model = minimize_xz_tilt(model, verbose=False, wrap=True)
        pos = torch.tensor(model.get_positions(), dtype=pos.dtype, device=pos.device)
        cell = torch.tensor(model.cell.array, dtype=pos.dtype, device=pos.device)

    # 3. get minimum pairwise distance in 3x3x3 supercell
    cell_inv = torch.inverse(cell)
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    diff_frac = torch.matmul(diff, cell_inv)
    # shifts = torch.stack(torch.meshgrid([torch.tensor([-1, 0, 1])]*3, indexing='ij')).view(3, -1).T.to(pos.device)
    _shift = torch.arange(-max_shift, max_shift + 1, device=pos.device, dtype=torch.long)
    shifts = torch.stack(torch.meshgrid([_shift] * 3, indexing='ij')).view(3, -1).T
    diff_frac_expanded = diff_frac.unsqueeze(2) + shifts.unsqueeze(0).unsqueeze(0)  # (N, N, max_shift^3, 3)
    diff_cart = torch.matmul(diff_frac_expanded, cell)  # (N, N, 27, 3)

    all_norms = torch.linalg.norm(diff_cart, dim=-1)

    if not return_all_neighborlist:
        # return NxN matrix, which has min d values
        _ = all_norms.min(dim=2)
        dists, min_idx = _.values, _.indices

        if return_vec:
            N = diff_cart.size(0)
            i = torch.arange(N, device=diff_cart.device)
            j = torch.arange(N, device=diff_cart.device)
            ii, jj = torch.meshgrid(i, j, indexing='ij')
            vec_mat = diff_cart[ii, jj, min_idx, :]
            if not return_ALL:
                return vec_mat, dists
            # else -> get out
        else:
            return dists

    # else
    mask = (all_norms < r_cut_for_neighbor + _eps) & (all_norms > _eps)
    Is, Js, Ks = torch.where(mask)
    Vecs = diff_cart[Is, Js, Ks, :]
    Ds = all_norms[Is, Js, Ks]
    shifts = -shifts[Ks]

    if not return_ALL:
        return Is, Js, Vecs, Ds, shifts
    else:
        return Is, Js, Vecs, Ds, shifts, vec_mat, dists
