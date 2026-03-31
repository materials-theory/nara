from llumys.gnn import *
from ase.constraints import FixAtoms
from ase.geometry import find_mic, get_distances
from ase.io.extxyz import read_xyz, write_xyz

from scipy.spatial.transform import Rotation
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment

import copy
from itertools import combinations, permutations

_eps = 1e-8

@torch.no_grad()
def get_desc_eq(atoms, gnn_model, return_dict=False, device=None):
    if not return_dict:
        return gnn_model.predict(atoms, return_desc=True, device=device)
    else:
        _dict = gnn_model.atoms2dict(atoms, device=device, include_D_matrix = True)
        atomsdict = collate_fn([_dict], single_input=True)
        return gnn_model.forward(atomsdict, return_desc=True), atomsdict

def eq2inv(eq_desc, irreps_eq:str = None, idx_slices:List = None):
    if idx_slices is not None:
        assert irreps_eq is not None, "Either irreps_eq or idx_slices should be typed in"      
    else:
        irreps_eq = Irreps(irreps_eq)
        irreps_inv = []
        
        idx_slices = []
        start_idx = 0
        total_ginv_dim = 0
        for mul, irrep in irreps_eq:
            ginv_dim = mul * irrep.dim
            if irrep.l==0 and irrep.p==1:
                idx_slices.append(slice(start_idx, start_idx+ginv_dim))
                irreps_inv.append((mul, irrep))
                total_ginv_dim += ginv_dim
            start_idx += ginv_dim        
    assert isinstance(idx_slices, list)

    parts = [eq_desc[:,sl] for sl in idx_slices]
    return torch.cat(parts, dim=-1)

def get_fixed_atoms_index(atoms, invert=False):
    fixed_index = []
    for _ in atoms.constraints:
        if isinstance(_, FixAtoms):
            fixed_index += list(_.get_indices())
    if not invert:
        return np.unique(fixed_index)
    else:
        unfixed_index = []
        for i in range(len(atoms)):
            if i not in fixed_index:
                unfixed_index.append(i)
        return unfixed_index

def superpose_permute(a: Atoms, b: Atoms, pbc:bool = True, return_result = True):
    '''
    find atom index matching to get minimum RMSD
    '''
    assert len(a) == len(b), "Length of a and b should be same"
    assert isinstance(a, Atoms) and isinstance(b, Atoms), "a and b should be both ase.Atoms object"
    temp_a, temp_b = copy.deepcopy(a), copy.deepcopy(b)
    _ta, _tb = temp_a.get_atomic_numbers(), temp_b.get_atomic_numbers()
    sort_1_a, sort_1_b = np.argsort(_ta), np.argsort(_tb)
    temp_a, temp_b = temp_a[sort_1_a], temp_b[sort_1_b]
    _ta, _tb = _ta[sort_1_a], _tb[sort_1_b]
    assert np.all(_ta == _tb), "Species should be same between a and b"

    _t_unique = sorted(np.unique(_ta))
    indices_a, indices_b = [], []
    start_ind = 0
    for _ele in _t_unique:
        _a_mark = _ta == _ele
        _b_mark = _tb == _ele
        assert np.sum(_a_mark) == np.sum(_b_mark)
        Ds_vec, Ds = get_distances(temp_a[_a_mark].get_positions(),
                                   temp_b[_b_mark].get_positions(),
                                   pbc=pbc, cell=temp_a.cell.copy())
        according_Ds = Ds
        a_permute_i, b_permute_i = linear_sum_assignment(according_Ds)
        atomic_indices_a = list(a_permute_i + start_ind)
        atomic_indices_b = list(b_permute_i + start_ind)
        indices_a += atomic_indices_a
        indices_b += atomic_indices_b
        start_ind += np.sum(_a_mark)
    final_sort_a = sort_1_a[indices_a]
    final_sort_b = sort_1_b[indices_b]

    _ = np.argsort(final_sort_a)
    final_sort_a, final_sort_b = final_sort_a[_], final_sort_b[_]
    return (temp_a[indices_a], temp_b[indices_b]) if return_result else (final_sort_a, final_sort_b)

def get_possible_R_spglib(atoms, symprec=1e-3):
    import spglib
    _cell_to_spglib = (atoms.cell.array, atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    try:
        
        dataset = spglib.get_symmetry(_cell_to_spglib, symprec=symprec)
    except:
        try:
            print("Old version of spglib is being used; trying with the Atoms object directly.")
            dataset = spglib.get_symmetry(atoms, symprec=symprec)
        except:
            raise RuntimeError("spglib cannot be used: failed to obtain symmetry information.")
    rotations = dataset["rotations"]
    # translations = dataset["translations"]
    return rotations

def get_possible_R_pymatgen(atoms, symprec=1e-3):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.io.ase import AseAtomsAdaptor
    structure = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    sym_ops_pmg = sga.get_symmetry_operations(cartesian=False)
    Rs = []
    for op in sym_ops_pmg:
        rot = op.rotation_matrix
        Rs.append(rot)
    return Rs

def RMSD(src:Atoms, dst:Atoms, pbc:bool = True, permute=True, return_vec=False):
    if permute:
        _Ai, _Bi = superpose_permute(src, dst, return_result=False)
        dst = dst[_Bi]
    if pbc:
        v, _ = find_mic(dst.get_positions()-src.get_positions(), cell=src.cell)
    else:
        v = dst.get_positions() - src.get_positions()
    rmsd = np.sqrt(np.mean(np.sum(v**2, axis=1)))
    return (v, rmsd) if return_vec else rmsd

from scipy.optimize import minimize
def minimize_translations(
    src:Atoms,
    dst:Atoms,
    t_init = [0,0,0],
    permute=False,
    pbc=True,
    verbose=False,
    method="CG",
    tol=1e-8,
    trajectory=None,
    include_dst_atoms_in_trajectory = False,
):
    
    def get_rmsd(T):
        model = src.copy()
        model.translate(T)
        d = RMSD(model, dst, pbc=pbc, permute=permute, return_vec=False) # 이거 pbc 끄는 게 좋아보이는데?
        return d
    
    global it
    it = 1
    def callbackF(x):
        global it
        it += 1
        if verbose:
            print(f"{it:6d}\t{x[0]:10.6f}\t{x[1]:10.6f}\t{x[2]:10.6f}")
    if verbose:
        print(f"{'iter':>6s}\t{'x':>10s}\t{'y':>10s}\t{'z':>10s}")
        print(f"{it:6d}\t{t_init[0]:10.6f}\t{t_init[1]:10.6f}\t{t_init[2]:10.6f}")
    opts = dict(disp=verbose, return_all = True, maxiter=500)
    min_res = minimize(method=method, fun=get_rmsd, x0=t_init, options=opts, callback=callbackF, tol=tol)

    if trajectory is not None:
        assert isinstance(trajectory, str), "Trajectory name should be in string"
        assert trajectory.endswith(".xyz"), "Trajectory's file extension should be xyz"
        traj = []
        for _T in min_res.allvecs:
            model = src.copy()
            model.translate(_T)
            traj.append(model)
        if include_dst_atoms_in_trajectory:
            traj.append(dst.copy())
        with open(trajectory, "w") as fo:
            write_xyz(fo, traj)
    return min_res.x

def get_d_pos(
        atoms_A,
        atoms_B,
        descA_inv,
        descB_inv,
        match_r_cut = 3,
        length_crit = 0.5,
        pbc=True,
        match_xy_only = False,
        proper_rotation_only = False,
        symprec = 1e-3,
        fallback_if_rmsd_is_bigger = True,
        return_modified_atoms_B = False
):
    '''
    :param match_r_cut
    |__ r_cut in here can be different from r_cut used in gnn_model. The first peak in RDF might be reasonable choice
    
    :param pbc
    |__ True
      |__ match_xy_only = True
        (i) get R matrix based on three best-matching pairs - based on inv.descriptor
        (ii) 
      |__ match_xy_only = False: 
        : 
    |__ False:
      Overall: new_atoms = (atoms_B@R + T)@R2 + T2
      : R=main rotation -> T=translation -> Atom index assignment
      -> R2=Kabsch for further minimization of RMSD -> T2=center aligning
      
      (i) R: Rotation matrix based on three best-matching pairs - based on inv. descriptor
      (ii) T: perform translations to align first best_matching pair
      (iii) perform hungarrian algorithm to assign optimal atom indexes - based on coordinates
      (iv) R2: R2 matrix based on Kabsch algorithm after centering both structures
      (v) T2: Translation (-new_B_center + A_center)
      (vi) get d_pos without consideration of mic
    
    :param match_xy_only
    |__ True: pbc = [True, True, False]
    |__ False: pbc = [True, True, True] if pbc=True else pbc=False
    
    :param proper_rotation_only
    |__ True: Considering proper rotations only (Kabsch algorithm) (allow det = +1 only)
    |__ False: Considering rotations + reflection/inversion as well (allow det = +1 or -1)
    '''
    if pbc and not np.any(atoms_A.pbc):
        raise IOError("Input structure seems non-pbc structure, whereas you set pbc=True in get_d_pos function.\
        set pbc=False if you want to get d_pos in non-pbc structures")
    
    assert isinstance(descA_inv, torch.Tensor), "type(descA_inv) should be torch.Tensor"
    
    fixed_atoms_A = get_fixed_atoms_index(atoms_A)
    fixed_atoms_B = get_fixed_atoms_index(atoms_B)
    assert len(fixed_atoms_A) == len(fixed_atoms_B), f"Fixed atoms for A and B should be identical.\
    Currently A has {len(fixed_atoms_A)} fixed atoms, whereas B has {len(fixed_atoms_B)} fixed atoms"
    
    N = len(atoms_A)
    assert len(atoms_A) == len(atoms_B),f"Length of atoms_A and atoms_B should be identical.\
    Currently A has {len(atoms_A)} atoms, whereas B has {len(atoms_B)} atoms"
    
    if fallback_if_rmsd_is_bigger:
        d_pos_for_fallback, RMSD_crit_for_fallback = RMSD(atoms_A, atoms_B, pbc=pbc, permute=True, return_vec=True)
    else:
        RMSD_crit_for_fallback = np.inf
    
    # if precomputed_neighbors_A is not None:
    #     Is_A, Js_A, Vecs_A, Ds_A = precomputed_neighbors_A
    # else:
    pos_A = torch.tensor(atoms_A.get_positions(), dtype=descA_inv.dtype, device=descA_inv.device)
    cell_A = torch.tensor(atoms_A.cell.array, dtype=descA_inv.dtype, device=descA_inv.device)

    while True:
        if pbc:
            Is_A, Js_A, Vecs_A, Ds_A, shifts_A = get_distances_torch(pos_A, cell=cell_A, return_all_neighborlist = True, r_cut_for_neighbor = match_r_cut)
        else:
            diffs_A = pos_A.unsqueeze(1) - pos_A.unsqueeze(0)
            all_norms_A = torch.linalg.norm(diffs_A, dim=-1)
            mask_A = (all_norms_A < match_r_cut + _eps) & (all_norms_A > _eps)
            Is_A, Js_A = torch.where(mask_A)
            Vecs_A = diffs_A[Is_A, Js_A]
            Ds_A = all_norms_A[Is_A, Js_A]

        # if precomputed_neighbors_B is not None:
        #     Is_B, Js_B, Vecs_B, Ds_B = precomputed_neighbors_B
        # else:
        pos_B = torch.tensor(atoms_B.get_positions(), dtype=descB_inv.dtype, device=descB_inv.device)
        cell_B = torch.tensor(atoms_B.cell.array, dtype=descB_inv.dtype, device=descB_inv.device)
        if pbc:
            Is_B, Js_B, Vecs_B, Ds_B, shifts_B = get_distances_torch(pos_B,
                                                                     cell=cell_B,
                                                                     return_all_neighborlist = True,
                                                                     r_cut_for_neighbor = match_r_cut)
        else:
            diffs_B = pos_B.unsqueeze(1) - pos_B.unsqueeze(0)
            all_norms_B = torch.linalg.norm(diffs_B, dim=-1)
            mask_B = (all_norms_B < match_r_cut + _eps) & (all_norms_B > _eps)
            Is_B, Js_B = torch.where(mask_B)
            Vecs_B = diffs_B[Is_B, Js_B]
            Ds_B = all_norms_B[Is_B, Js_B]

        # first pair: best matching index
        D = descB_inv.unsqueeze(0) - descA_inv.unsqueeze(1)
        D = torch.linalg.norm(D, dim=-1)
        D[fixed_atoms_A] = torch.inf
        D[:,fixed_atoms_B] = torch.inf
        i, j = torch.unravel_index(torch.argmin(D), D.shape) # best matched!
        i = i.item()
        j = j.item()
        first_pair = (i, j)

        i_neighsA = torch.where(Is_A==first_pair[0])[0]
        i_neighsB = torch.where(Is_B==first_pair[1])[0]

        if len(i_neighsA)>=3 and len(i_neighsB)>=3:
            break
        else:
            print(f"Current match_r_cut seems to tight. Increase +0.5 Å: from {match_r_cut:.4f} to {match_r_cut+0.5:.4f}")
            match_r_cut += 0.5
    
    ### Based on Inv descriptor, considering all pairs
    min_diff = np.inf
    
    neighsA = Js_A[i_neighsA]
    neighsB = Js_B[i_neighsB]
    
    unique_neighsA = torch.unique(neighsA)
    unique_neighsB = torch.unique(neighsB)
    
    pA = list(permutations(range(len(unique_neighsA)), 2))
    pB = list(permutations(range(len(unique_neighsB)), 2))

    vec_A = None
    vec_B = None
    for _a, _b in pA:
        a, b = unique_neighsA[_a].item(), unique_neighsA[_b].item()
        for _k, _l in pB:
            k, l = unique_neighsB[_k].item(), unique_neighsB[_l].item()
            if atoms_A[a].symbol != atoms_B[k].symbol:
                continue
            if atoms_A[b].symbol != atoms_B[l].symbol:
                continue
            _diff = (torch.linalg.norm(descB_inv[k]-descA_inv[a]) + torch.linalg.norm(descB_inv[l]-descA_inv[b])).item()
            if _diff < min_diff:
                # check whether the triangle has small length deviation less than the criteria
                A_edge_a = torch.where((Is_A==i)&(Js_A==a))[0]
                A_edge_b = torch.where((Is_A==i)&(Js_A==b))[0]
                
                B_edge_k = torch.where((Is_B==j)&(Js_B==k))[0]
                B_edge_l = torch.where((Is_B==j)&(Js_B==l))[0]
            
                A_edge_ab = torch.where((Is_A==a)&(Js_A==b))[0]
                B_edge_kl = torch.where((Is_B==k)&(Js_B==l))[0]

                this_vec_A = None
                this_vec_B = None
                for aei in A_edge_a:
                    for bei in A_edge_b:
                        for kei in B_edge_k:
                            for lei in B_edge_l:
                                cand_vec_A = torch.stack([Vecs_A[aei], Vecs_A[bei]])
                                cand_vec_B = torch.stack([Vecs_B[kei], Vecs_B[lei]])
                                cand_vec_A_d = None
                                for ab_i in A_edge_ab:
                                    if torch.allclose(Vecs_A[ab_i], cand_vec_A[1]-cand_vec_A[0]):
                                        cand_vec_A_d = Vecs_A[ab_i]
                                if cand_vec_A_d is None:
                                    continue

                                cand_vec_B_d = None
                                for kl_i in B_edge_kl:
                                    if torch.allclose(Vecs_B[kl_i], cand_vec_B[1]-cand_vec_B[0]):
                                        cand_vec_B_d = Vecs_B[kl_i]
                                if cand_vec_B_d is None:
                                    continue

                                if torch.abs(torch.linalg.norm(cand_vec_A, dim=1) - torch.linalg.norm(cand_vec_B, dim=1)).max() >= length_crit + _eps:
                                    continue
                                if torch.abs(torch.linalg.norm(cand_vec_A_d)-torch.linalg.norm(cand_vec_B_d)) >= length_crit + _eps:
                                    continue

                                this_vec_A, this_vec_B = cand_vec_A, cand_vec_B
                                        
                if this_vec_A is None or this_vec_B is None:
                    continue

                # if the length difference is less than the criteria, change the min_diff & indexes of pairs
                min_diff = _diff
                second_pair = (a, k)
                third_pair = (b, l)

                vec_A = this_vec_A
                vec_B = this_vec_B

                if match_xy_only:
                    vec_A[:,2] = 0
                    vec_B[:,2] = 0

    if (vec_A is None) or (vec_B is None):
        print("None of the vectors matching well.. Only translation will be performed")
        _Ai, _Bi = superpose_permute(atoms_A, atoms_B, return_result=False, pbc=pbc)
        new_atoms = atoms_B[_Bi].copy()
        assert np.all(_Ai==np.argsort(_Ai)), "Outdated superpose_permute function is used"
        _B_to_go = minimize_translations(src = new_atoms, dst = atoms_A, pbc=pbc)
        new_atoms.translate(_B_to_go)

        d_pos, rmsd_after_transform = RMSD(atoms_A, new_atoms, pbc=pbc, permute=False, return_vec=True)
        if rmsd_after_transform > RMSD_crit_for_fallback:
            print("RMSD after transform exceeds RMSD before transform. Use atoms_B itself")
            return (d_pos_for_fallback, atoms_B.copy()) if return_modified_atoms_B else d_pos
        return (d_pos, new_atoms) if return_modified_atoms_B else d_pos

    # Get rotation matrix, R
    if pbc:
        if len(fixed_atoms_B) > 3:
            all_possible_rotations = get_possible_R_spglib(atoms_B[fixed_atoms_B], symprec=symprec)
        else:
            all_possible_rotations = get_possible_R_spglib(atoms_B, symprec=symprec)
    
        all_possible_rots_matching_conditions = []
        
        if match_xy_only:
            for apr in all_possible_rotations:
                if np.allclose(apr[2], np.array([0,0,1])) and np.allclose(apr[:,2], np.array([0,0,1])):
                    all_possible_rots_matching_conditions.append(apr)
        else:
            all_possible_rots_matching_conditions = all_possible_rotations

        if proper_rotation_only:
            R, d_norm = Rotation.align_vectors(vec_A, vec_B) # Use Kabsch algorithm. det(R) = 1
            R = R.as_matrix()
        else:
            R, scale = orthogonal_procrustes(vec_B, vec_A) # det(R) = 1 or -1
            R = R.T

        L = atoms_A.cell.array
        L_inv = np.linalg.inv(L)
        R_int_approx = L_inv.T@R@L.T

        R_int = np.round(R_int_approx)

        if len(all_possible_rots_matching_conditions)==1:
            R_int = np.round(R_int_approx)
            if not np.allclose(R_int@R_int.T, np.eye(3), atol=1e-8):
                # If matrix M is orthogonal, M@M.T = I, which means M^-1 = M.T
                # print("Currently R_int seems... non-orthogonal matrix. Not using Rotation transform.")
                R_snap = np.eye(3)
            else:
                R_snap = L.T@R_int@L_inv.T
        else:
            _ = np.argmin([np.linalg.norm(R_int_approx-M, ord="fro") for M in all_possible_rots_matching_conditions]) # Frobenius norm
            R_int = all_possible_rots_matching_conditions[_]
            R_snap = L.T@R_int@L_inv.T
    
        new_atoms = atoms_B.copy()
        new_atoms.set_positions(atoms_B.get_positions()@R_snap.T, apply_constraint=False)
        to_translate = atoms_A[first_pair[0]].position - new_atoms[first_pair[1]].position # no need of mic
        new_atoms.translate(to_translate)
        new_atoms.wrap()
        _Ai, _Bi = superpose_permute(atoms_A, new_atoms, pbc=pbc, return_result=False)
        assert np.all(_Ai==np.argsort(_Ai)), "Outdated superpose_permute function is used"
        new_atoms = new_atoms[_Bi]
        
        if len(fixed_atoms_A) > 0:
            _A, _B = atoms_A[fixed_atoms_A], new_atoms[fixed_atoms_B]
        else:
            _A, _B = atoms_A, new_atoms
        _Ai, _Bi = superpose_permute(_A, _B, return_result=False, pbc=pbc)
        _B = _B[_Bi]
        _B_to_go = minimize_translations(src = _B, dst = _A, pbc=pbc)
        
        if len(fixed_atoms_A) > 0:
            _Ai, _Bi = superpose_permute(atoms_A, new_atoms, return_result=False, pbc=pbc)
            new_atoms = new_atoms[_Bi]
            
        new_atoms.translate(_B_to_go)
        new_atoms.wrap()

        d_pos, rmsd_after_transform = RMSD(atoms_A, new_atoms, pbc=pbc, permute=False, return_vec=True)
        if rmsd_after_transform > RMSD_crit_for_fallback:
            print("RMSD after transform exceeds RMSD before transform. Use atoms_B itself")
            return (d_pos_for_fallback, atoms_B.copy()) if return_modified_atoms_B else d_pos
        return (d_pos, new_atoms) if return_modified_atoms_B else d_pos
        
    else:
        if proper_rotation_only:
            R, d_norm = Rotation.align_vectors(vec_A, vec_B) # Use Kabsch algorithm. det(R) = 1
            R = R.as_matrix()
        else:
            R, scale = orthogonal_procrustes(vec_B, vec_A) # det(R) = 1 or -1
            R = R.T
    
        # (atoms_B@R + T)@R2 + T2 ## R -- main rotation > T -- translation > atom index assignment > R2 -- Kabsch for further minimization of RMSD > T2 -- centering
        new_atoms = atoms_B.copy()
        new_atoms.set_positions(atoms_B.get_positions()@R.T, apply_constraint=False)
        to_translate = atoms_A[first_pair[0]].position - new_atoms[first_pair[1]].position # no need of mic
        new_atoms.translate(to_translate)
    
        if len(fixed_atoms_A) > 1:
            _A, _B = atoms_A[fixed_atoms_A], new_atoms[fixed_atoms_B]
        else:
            _A, _B = atoms_A, new_atoms
    
        _A_center = np.mean(_A.get_positions(), axis=0)
        _B_center = np.mean(_B.get_positions(), axis=0)
        _A.translate(-_A_center)
        _B.translate(-_B_center)
        _Ai, _Bi = superpose_permute(_A, _B, return_result=False, pbc=pbc)
        assert np.all(_Ai==np.argsort(_Ai)), "Outdated superpose_permute function is used."
        _B = _B[_Bi]
        R2, _ = Rotation.align_vectors(_A.get_positions(), _B.get_positions()) # _B를 _A로
        R2 = R2.as_matrix()
    
    
        if len(fixed_atoms_A) > 2:
            _B.set_positions(_B.get_positions()@R2.T, apply_constraint=False)
            if not np.allclose(_A.get_positions(), _B.get_positions(), atol=1e-3):
                print("Rotation based on best-matching pairs unsuccessful. Not using Rotation matrix.")
                _A, _B = atoms_A[fixed_atoms_A], atoms_B[fixed_atoms_B]
                if np.allclose(_A.get_positions(), _B.get_positions(), atol=1e-3):
                    new_atoms = atoms_B.copy()
                else:
                    _A_center = np.mean(_A.get_positions(), axis=0)
                    _B_center = np.mean(_B.get_positions(), axis=0)
                    _A.translate(-_A_center)
                    _B.translate(-_B_center)
                    if np.allclose(_A.get_positions(), _B.get_positions(), atol=1e-3):
                        new_atoms = atoms_B.copy()
                        new_atoms.translate(- _B_center + _A_center)
                    else:
                        _Ai, _Bi = superpose_permute(_A, _B, return_result = False, pbc=pbc)
                        _B = _B[_Bi]
                        if np.allclose(_A.get_positions(), _B.get_positions(), atol=1e-3):
                            new_atoms = atoms_B.copy()
                            new_atoms.translate(- _B_center + _A_center)
                        else:
                            raise RuntimeError("If there are fixed atoms with pbc=False, how can this be happened? I cannot match the indexes rigorously")
    
                _Ai, _Bi = superpose_permute(atoms_A, new_atoms, return_result=False, pbc=pbc)
                new_atoms = new_atoms[_Bi]
                _go_ahead_with_R = False
                # There are more than two fixed atoms, but it seems that R@R2 can not assign them correctly
                # Do not perform R transform
            else:
                _go_ahead_with_R = True
        else:
            # Yep there are no more than two fixed atoms. Lets perform R@R2 with translations
            _go_ahead_with_R = True
    
        if _go_ahead_with_R:
            # _B_to_go = minimize_translations(src = _B, dst = _A) # maybe we don't need to do this
            new_atoms.set_positions((_B.get_positions()-_B_center)@R2.T + _A_center, apply_constraint=False) # add +_B_to_go after _A_center if further translation. T3 is needed. (maybe not)
    
        d_pos, rmsd_after_transform = RMSD(atoms_A, new_atoms, pbc=pbc, permute=False, return_vec=True)
        if rmsd_after_transform > RMSD_crit_for_fallback:
            print("RMSD after transform exceeds RMSD before transform. Use atoms_B itself")
            return (d_pos_for_fallback, atoms_B.copy()) if return_modified_atoms_B else d_pos
        return (d_pos, new_atoms) if return_modified_atoms_B else d_pos

# def get_d_pos_wrapper(atoms_A, atoms_B, gnn_model, descA_eq = None, descB_eq = None, device=None, max_allowance = 1, pbc = True, match_xy_only=False):
#     if device is None:
#         device = torch.device("cpu")
#
#     if 3 not in atoms_A.get_tags():
#         raise RuntimeError("What is the Movable atoms in atoms_A? Movable atoms should be tagged as 3")
#     if 3 not in atoms_B.get_tags():
#         raise RuntimeError("What is the Movable atoms in atoms_B? Movable atoms should be tagged as 3")
#
#     _fixed_A = get_fixed_atoms_index(atoms_A)
#     _fixed_B = get_fixed_atoms_index(atoms_B)
#     a_mov_mask, a_rel_mask, a_fix_mask = [], [], []
#     b_mov_mask, b_rel_mask, b_fix_mask = [], [], []
#
#     for i, (_a, _b) in enumerate(zip(atoms_A, atoms_B)):
#
#         if _a.tag == 1:
#             assert i in _fixed_A
#             a_fix_mask.append(i)
#         elif _a.tag == 2:
#             a_rel_mask.append(i)
#         elif _a.tag == 3:
#             a_mov_mask.append(i)
#         else:
#             raise IOError(f"{_a.tag} cannot be parsed")
#
#         if _b.tag == 1:
#             assert i in _fixed_B
#             b_fix_mask.append(i)
#         elif _b.tag == 2:
#             b_rel_mask.append(i)
#         elif _b.tag == 3:
#             b_mov_mask.append(i)
#
#     for _a, _b, _lab in zip([a_fix_mask, a_rel_mask, a_mov_mask],
#                             [b_fix_mask, b_rel_mask, b_mov_mask],
#                             ["fixed", "relaxable", "movable"]):
#         if len(_a)!=len(_b):
#             raise RuntimeError(f"Number of {_lab:s} atoms in A({len(_a)} atoms) and B({len(_b)} atoms) differs")
#
#     # fixed atoms (tag==1)
#     a_fix = atoms_A[a_fix_mask]
#     b_fix = atoms_B[b_fix_mask]
#     _d, _ = RMSD(src=a_fix, dst=b_fix, pbc=pbc, permute=True, return_vec = True)
#     assert np.max(np.linalg.norm(_d, axis=1)) < _eps, f"fixed atoms should be fixed tight"
#
#     # relaxable atoms (tag==2)
#     a_rel = atoms_A[a_rel_mask]
#     b_rel = atoms_B[b_rel_mask]
#     D_POS_REL, _ = RMSD(src=a_rel, dst=b_rel, pbc=pbc, permute=True, return_vec = True)
#     max_norm_rel = np.max(np.linalg.norm(D_POS_REL, axis=1))
#     if max_norm_rel > max_allowance + _eps:
#         print(f"Relaxable atoms moved a lot. Max norm: {max_norm_rel:.4f} Å")
#         include_rel_atoms = True
#     else:
#         include_rel_atoms = False
#
#     A_index_to_get_d_pos_with_transform = (a_rel_mask + a_mov_mask) if include_rel_atoms else a_mov_mask
#     B_index_to_get_d_pos_with_transform = (b_rel_mask + b_mov_mask) if include_rel_atoms else b_mov_mask
#
#     model_A = atoms_A[A_index_to_get_d_pos_with_transform]
#     model_B = atoms_B[B_index_to_get_d_pos_with_transform]
#
#     if descA_eq is None:
#     	A_eq = get_desc_eq(atoms_A, gnn_model, return_dict = False, device = device)[A_index_to_get_d_pos_with_transform]
#     else:
#     	A_eq = descA_eq[A_index_to_get_d_pos_with_transform]
#
#     if descB_eq is None:
#     	B_eq = get_desc_eq(atoms_B, gnn_model, return_dict = False, device = device)[B_index_to_get_d_pos_with_transform]
#     else:
#     	B_eq = descB_eq[B_index_to_get_d_pos_with_transform]
#
#     irreps_hidden = gnn_model.blocks[-1].gate.irreps_out
#     descA_inv = eq2inv(A_eq, irreps_hidden)
#     descB_inv = eq2inv(B_eq, irreps_hidden)
#
#     D_POS, new_atoms = get_d_pos(atoms_A = model_A, atoms_B = model_B,
#         descA_inv = descA_inv, descB_inv = descB_inv,
#         match_r_cut = 3, pbc=True, match_xy_only = match_xy_only,
#         proper_rotation_only = False,
#         symprec = 1e-3,
#         fallback_if_rmsd_is_bigger = False,
#         return_modified_atoms_B = True)
#
#     D_POS_ALL = np.zeros(atoms_A.get_positions().shape)
#     D_POS_ALL[A_index_to_get_d_pos_with_transform] = D_POS
#     if not include_rel_atoms:
#         D_POS_ALL[a_rel_mask] = D_POS_REL
#     return D_POS_ALL


def get_d_pos_wrapper(atoms_A, atoms_B, gnn_model, descA_eq=None, descB_eq=None, device=None, max_allowance=1, pbc=True,
                      match_xy_only=False):
    if device is None:
        device = torch.device("cpu")

    if descA_eq is None:
        A_eq = get_desc_eq(atoms_A, gnn_model, return_dict=False, device=device)
    else:
        A_eq = descA_eq

    if descB_eq is None:
        B_eq = get_desc_eq(atoms_B, gnn_model, return_dict=False, device=device)
    else:
        B_eq = descB_eq

    irreps_hidden = gnn_model.blocks[-1].gate.irreps_out
    descA_inv = eq2inv(A_eq, irreps_hidden)
    descB_inv = eq2inv(B_eq, irreps_hidden)
    return get_d_pos_wrapper_inv(atoms_A = atoms_A,
                                 atoms_B = atoms_B,
                                 descA_inv = descA_inv,
                                 descB_inv = descB_inv,
                                 max_allowance = max_allowance,
                                 pbc = pbc,
                                 match_xy_only = match_xy_only)

def get_d_pos_wrapper_inv(atoms_A, atoms_B, descA_inv, descB_inv, max_allowance=1,
                          pbc = True, match_xy_only = False, match_r_cut = 3):

    if 3 not in atoms_A.get_tags():
        raise RuntimeError("What is the Movable atoms in atoms_A? Movable atoms should be tagged as 3")
    if 3 not in atoms_B.get_tags():
        raise RuntimeError("What is the Movable atoms in atoms_B? Movable atoms should be tagged as 3")

    _fixed_A = get_fixed_atoms_index(atoms_A)
    _fixed_B = get_fixed_atoms_index(atoms_B)
    a_mov_mask, a_rel_mask, a_fix_mask = [], [], []
    b_mov_mask, b_rel_mask, b_fix_mask = [], [], []

    for i, (_a, _b) in enumerate(zip(atoms_A, atoms_B)):

        if _a.tag == 1:
            assert i in _fixed_A
            a_fix_mask.append(i)
        elif _a.tag == 2:
            a_rel_mask.append(i)
        elif _a.tag == 3:
            a_mov_mask.append(i)
        else:
            raise IOError(f"{_a.tag} cannot be parsed")

        if _b.tag == 1:
            assert i in _fixed_B
            b_fix_mask.append(i)
        elif _b.tag == 2:
            b_rel_mask.append(i)
        elif _b.tag == 3:
            b_mov_mask.append(i)

    for _a, _b, _lab in zip([a_fix_mask, a_rel_mask, a_mov_mask],
                            [b_fix_mask, b_rel_mask, b_mov_mask],
                            ["fixed", "relaxable", "movable"]):
        if len(_a) != len(_b):
            raise RuntimeError(f"Number of {_lab:s} atoms in A({len(_a)} atoms) and B({len(_b)} atoms) differs")

    # fixed atoms (tag==1)
    if len(a_fix_mask)>0:
        a_fix = atoms_A[a_fix_mask]
        b_fix = atoms_B[b_fix_mask]
        _d, _ = RMSD(src=a_fix, dst=b_fix, pbc=pbc, permute=True, return_vec=True)
        assert np.max(np.linalg.norm(_d, axis=1)) < _eps, f"fixed atoms should be fixed tight"

    # relaxable atoms (tag==2)
    if len(a_rel_mask)>0:
        a_rel = atoms_A[a_rel_mask]
        b_rel = atoms_B[b_rel_mask]
        D_POS_REL, _ = RMSD(src=a_rel, dst=b_rel, pbc=pbc, permute=True, return_vec=True)
        max_norm_rel = np.max(np.linalg.norm(D_POS_REL, axis=1))
        if max_norm_rel > max_allowance + _eps:
            print(f"Relaxable atoms moved a lot. Max norm: {max_norm_rel:.4f} Å")
            include_rel_atoms = True
        else:
            include_rel_atoms = False
    else:
        include_rel_atoms = False # since there is no rel_atoms

    A_index_to_get_d_pos_with_transform = (a_rel_mask + a_mov_mask) if include_rel_atoms else a_mov_mask
    B_index_to_get_d_pos_with_transform = (b_rel_mask + b_mov_mask) if include_rel_atoms else b_mov_mask

    model_A = atoms_A[A_index_to_get_d_pos_with_transform]
    model_B = atoms_B[B_index_to_get_d_pos_with_transform]

    descA_inv_parsed = descA_inv[A_index_to_get_d_pos_with_transform]
    descB_inv_parsed = descB_inv[B_index_to_get_d_pos_with_transform]

    if not isinstance(descA_inv_parsed, torch.Tensor):
        descA_inv_parsed = torch.tensor(descA_inv_parsed, dtype=torch.float64)
        descB_inv_parsed = torch.tensor(descB_inv_parsed, dtype=torch.float64)

    D_POS, new_atoms = get_d_pos(atoms_A=model_A, atoms_B=model_B,
                                 descA_inv=descA_inv_parsed, descB_inv=descB_inv_parsed,
                                 match_r_cut = match_r_cut, pbc=pbc, match_xy_only=match_xy_only,
                                 proper_rotation_only=False,
                                 symprec=1e-3,
                                 fallback_if_rmsd_is_bigger=False,
                                 return_modified_atoms_B=True)

    D_POS_ALL = np.zeros(atoms_A.get_positions().shape)
    D_POS_ALL[A_index_to_get_d_pos_with_transform] = D_POS
    if (len(a_rel_mask)>0) and (not include_rel_atoms):
        D_POS_ALL[a_rel_mask] = D_POS_REL
    return D_POS_ALL
