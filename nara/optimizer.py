# from ase.optimize.basin import BasinHopping
from ase.optimize import FIRE, QuasiNewton
from ase.io.trajectory import Trajectory
from ase.units import kB
from nara.get_d_pos import *

try:
    from vcsqnm_for_ase import aseOptimizer
    import contextlib
    class SQNM_opt(aseOptimizer):
        # for reference
        # def __init__(self, initial_structure, vc_relax = False, force_tol=1e-2, maximalSteps = 500, initial_step_size = -0.01
        #         , nhist_max =10, lattice_weigth = 2.0, alpha_min = 1e-3, eps_subsp = 1e-3):
        def __init__(self, atoms, logfile=None, trajectory=None, *args, **kwargs):
            super().__init__(initial_structure=atoms, *args, **kwargs)

            if trajectory is None:
                self.traj = None
            else:
                self.traj = Trajectory(trajectory, "w")
            self.logfile = logfile

        def logging(self, i):
            word = f"Relaxation step: {i:d} | " \
                   f"energy: {self.initial_structure.get_potential_energy():f} | " \
                   f"max f_norm: {np.max(np.abs(self.initial_structure.get_forces())):f} | "\
                   f"Derivative Norm: {self._getDerivativeNorm():f}"
            if self.logfile is None:
                pass
            elif self.logfile == "-":
                print(word)
            else:
                with open(self.logfile, 'a') as fo:
                    print(word, file=fo)

            if self.traj is not None:
                self.traj.write(
                    self.initial_structure,
                    energy=self.initial_structure.get_potential_energy())

        def run(self, fmax=None, steps=None):
            if fmax is None:
                fmax = self.force_tol
            if steps is None:
                steps = self.maximalSteps
            # convergence error.. we may don't need this in stderr
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stderr(devnull):
                    i = 0
                    self.logging(i)
                    while (i < steps and self._getDerivativeNorm() > fmax):
                        self.step(self.initial_structure)
                        i += 1
                        self.logging(i)

except (ModuleNotFoundError, ImportError) as e:
    class SQNM_opt:
        def __init__(self, *args, **kwargs):
            raise e

class BH:
    def __init__(self, atoms = None,
                 temperature = 300,
                 dr = 0.5,
                 optimizer = QuasiNewton,
                 fmax = 5e-2,
                 basin_traj = "accepted_basins.traj",
                 gbest_traj = "gbest.traj",
                 all_traj = None,
                 max_local_opt = 150):
        self.current_basin = atoms  # 여기 copy 쓰면 calculator 사라지네
        self.Eo = atoms.get_potential_energy()
        self.kT = temperature * kB
        self.dr = dr
        self.optimizer = optimizer
        self.fmax = fmax
        self._basin_traj = basin_traj
        self.basin_traj = Trajectory(basin_traj, 'w')
        self._all_traj = all_traj
        self.all_traj = Trajectory(all_traj, 'w') if isinstance(all_traj, str) else None
        self._gbest_traj = gbest_traj
        self.gbest_traj = Trajectory(gbest_traj, 'w') if isinstance(gbest_traj, str) else None
        self.gbest = None
        self.gbest_E = np.inf
        self.records = []  # recording (step, energy) # logs about basin change
        self.current_step = 0
        self.max_local_opt = max_local_opt

    def next_step(self):
        self.current_step += 1

        perturbed = self.current_basin.copy()
        displacements = np.random.uniform(-self.dr, self.dr, size=perturbed.get_positions().shape)

        # if 3 in self.current_basin.get_tags():
        #     displacements[self.current_basin.get_tags() == 1] = 0
        #     displacements[self.current_basin.get_tags() == 2] = 0

        new_positions = perturbed.get_positions() + displacements

        perturbed.set_positions(new_positions, apply_constraint=True)
        perturbed.calc = self.current_basin.calc
        dyn = self.optimizer(perturbed, logfile="-")
        dyn.run(fmax=self.fmax, steps=self.max_local_opt)

        Eo = self.Eo
        En = perturbed.get_potential_energy()

        # Metropolis acceptance criterion
        if (np.exp((Eo - En) / self.kT) > np.random.uniform()): # (En < Eo)
            accepted = True
            self.current_basin = perturbed
            self.Eo = En
            self.basin_traj.write(perturbed, energy=En)
            if En < self.gbest_E:
                self.gbest_traj.write(perturbed, energy=En)
                self.gbest = perturbed
                self.gbest_E = perturbed.get_potential_energy()
        else:
            accepted = False
            if self.all_traj is not None:
                self.all_traj.write(perturbed, energy=perturbed.get_potential_energy())

        to_say = [self.current_step, perturbed.get_potential_energy()]
        print(f"Current step: {to_say[0]:d}, E: {to_say[1]}")
        if accepted:
            self.records.append(to_say)
        return accepted, self.current_basin
