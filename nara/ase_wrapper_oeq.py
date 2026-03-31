from ase.calculators.calculator import Calculator, all_changes
from llumys.gnn_oeq import EquivariantGNN
from llumys.gnn_LL_oeq import EquivariantGNN_UC
from nara.get_d_pos import eq2inv
import torch

class GNNWrapper(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, gnn_model, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cpu") if device is None else device
        self.dtype = torch.float64 if dtype is None else dtype

        if isinstance(gnn_model, str):
            gnn_model = EquivariantGNN.load(gnn_model, dtype=self.dtype, device=self.device)
        assert isinstance(gnn_model, EquivariantGNN)
        self.gnn_model = gnn_model

    def calculate(self, atoms, properties=["energy"], system_changes=all_changes):
        self.gnn_model.eval()
        compute_forces = "forces" in properties
        res_dict = self.gnn_model.predict(atoms, return_desc=False, compute_forces=compute_forces, device=self.device)
        desc_eq = res_dict["node_features"].detach()
        atoms.arrays["node_features"] = desc_eq.cpu().numpy()
        atoms.arrays["node_features_inv"] = eq2inv(desc_eq,
                                                   irreps_eq = self.gnn_model.blocks[-1].gate.irreps_out).cpu().numpy()
        _E = res_dict["energy_pred"].item()
        self.results["energy"] = _E
        atoms.info["energy"] = _E
        if compute_forces:
            _F = res_dict["forces_pred"].detach().cpu().numpy()
            self.results["forces"] = _F
            atoms.arrays["forces"] = _F


class GNNUCWrapper(Calculator):
    implemented_properties = ['energy', 'forces']
    def __init__(self, gnn_model, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cpu") if device is None else device
        self.dtype = torch.float64 if dtype is None else dtype

        if isinstance(gnn_model, str):
            gnn_model = EquivariantGNN_UC.load(gnn_model, dtype=self.dtype, device=self.device)
        assert isinstance(gnn_model, EquivariantGNN_UC)
        self.gnn_model = gnn_model

    def calculate(self, atoms, properties=["energy"], system_changes=all_changes):
        self.gnn_model.eval()
        compute_forces = "forces" in properties
        res_dict = self.gnn_model.predict(atoms, return_desc=False, compute_forces=compute_forces, device=self.device)
        desc_eq = res_dict["node_features"].detach()
        atoms.arrays["node_features"] = desc_eq.cpu().numpy()
        atoms.arrays["node_features_inv"] = eq2inv(desc_eq,
                                                   irreps_eq = self.gnn_model.blocks[-1].gate.irreps_out).cpu().numpy()
        _E = res_dict["energy_pred"].item()
        self.results["energy"] = _E
        atoms.info["energy"] = _E
        # atoms.info["lhat"] = res_dict["Eloss_hat"].item()
        # atoms.arrays["lhat_per_atom"] = res_dict["Eloss_hat_per_node"].detach().cpu().numpy().flatten()
        atoms.arrays["lhat_per_atom"] = res_dict["Floss_hat_per_node"].detach().cpu().numpy().flatten()
        atoms.info["lhat"] = atoms.arrays["lhat_per_atom"].sum()
        
        if compute_forces:
            _F = res_dict["forces_pred"].detach().cpu().numpy()
            self.results["forces"] = _F
            atoms.arrays["forces"] = _F



class GNNWrapper_old(Calculator):
    # when using old version of EquivariantGNN_UC
    implemented_properties = ['energy', 'forces']

    def __init__(self, gnn_model, device=None, dtype=None, *args, **kwargs):
        Calculator.__init__(self, *args, **kwargs)
        self.device = torch.device("cpu") if device is None else device
        self.dtype = torch.float64 if dtype is None else dtype

        if isinstance(gnn_model, str):
            gnn_model = EquivariantGNN.load(gnn_model, dtype=self.dtype, device=self.device)
        assert isinstance(gnn_model, EquivariantGNN)
        self.gnn_model = gnn_model

    def calculate(self, atoms, properties=["energy"], system_changes=all_changes):
        self.gnn_model.eval()
        compute_forces = "forces" in properties
        res_dict = self.gnn_model.predict(atoms, return_desc=False, compute_forces=compute_forces, device=self.device)[0]
        self.results["energy"] = res_dict["energy_pred"].item()
        if compute_forces:
            self.results["forces"] = res_dict["forces_pred"].detach().cpu().numpy()