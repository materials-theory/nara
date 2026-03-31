# import multiprocessing as mp
import logging, os, copy, numbers, pickle, random, time
from datetime import datetime
from random import choice
import warnings

from string import ascii_letters
from abc import ABC, abstractmethod
from collections import Counter

from scipy.optimize import minimize, linear_sum_assignment
from scipy.signal import dfreqresp
from scipy.stats import levy # For levy flight in FA algorithm
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from ase import __version__ as ase_version
from ase.io.trajectory import Trajectory
from ase.io.vasp import write_vasp, read_vasp
from ase.io.extxyz import read_xyz, write_xyz
from ase.optimize import QuasiNewton, FIRE
from ase.calculators.singlepoint import SinglePointCalculator
from packaging.version import parse as get_V # for checking ASE version

from nara.distance import *
from nara.firefly.constraint import *
from nara.get_d_pos import get_d_pos_wrapper_inv
from nara.optimizer import BH, SQNM_opt
from nara.ase_wrapper import GNNUCWrapper

from llumys.gnn import *
from llumys.gnn_LL import EquivariantGNN_UC
from llumys.train import main_UC

__version__ = "1.0.0"

# Since ase 3.26.0, irun method in ASE optimizable requires gradient for log method
_ASE_OPT_NEEDS_GRAD = True if get_V(ase_version) >= get_V("3.26.0") else False

def safe_write_xyz(xyz_filename, atoms_list):
    for_safe_write = []
    for atoms in atoms_list:
        model = atoms.copy()
        model.info["energy"] = atoms.get_potential_energy()
        # model.info["stress"] = atoms.get_stress() # currently not implemented
        model.arrays["forces"] = atoms.get_forces()
        model.calc = None
        for_safe_write.append(model)

    with open(xyz_filename, 'w') as fo:
        write_xyz(fo, for_safe_write)

#### ------------------------------------------------------------------------------------------------------------------
#### ------------------------------------------------------------------------------------------------------------------
#### Guideline of defining agent!
class BaseAgent(ABC):
    def __init__(self,
                 x=None,
                 label: Optional[str] = None,
                 label_generation: Optional[str] = None,
                 constraint=None,
                 basedir: str = None,  # Base directory (Calculations will be held in [basedir]/[label] directory)
                 ):
        self._x = x
        self._label = label if label is not None else "".join([choice(ascii_letters) for i in range(10)])
        self.label_generation = label_generation
        self._fitness = None
        self.basedir = basedir

        if constraint is None:
            self.constraint = None
        else:
            for cc in self.compatible_constraint:
                if (not isinstance(constraint, cc)) and (constraint is not cc):
                    raise RuntimeError("Input constraint is not compatible with this agent...")
            self.constraint = constraint

    @property
    def label(self):
        if self.label_generation is not None:
            return f"{self.label_generation}_Agent_{self._label}"
        else:
            return f"Agent_{self._label}"

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, new_x):
        self._x = new_x
        self._fitness = None  # fitness should be re-calculated

    @x.deleter
    def x(self):
        self._x = None
        self._fitness = None

    @property
    def desc(self):
        '''
        Descriptor for this agent
        (when self.x cannot be a unique descriptor)
        '''
        return self.x

    @desc.setter
    def desc(self, new_desc):
        # not needed in most cases
        pass

    def __add__(self, other):
        if isinstance(other, type(self)):
            return self.x + other.x
        elif isinstance(other, (float, int, np.ndarray)):
            return self.x + other
        else:
            raise RuntimeError(f"Type of other is not supported: {type(other)}")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        assert isinstance(other, type(self))
        return self.x - other.x

    def check_bc(self, x=None) -> bool:
        '''
        : Check whether the 'x' is inside the boundary conditions or not. If False, x should be re-generated !
        '''
        if x is None:
            x = self._x

        if self.constraint is None:
            raise RuntimeError("Constraint not set in this agent")
        else:
            return self.constraint.check_valid(x)

    def get_random_x(self, normalize: bool = False, **kwargs) -> np.ndarray:
        '''
        :param nomalize: if False, it will generate new x, in given boundary condition.
                         if True, it will generate the random walk
        :return: random 'x'
        '''
        if self.constraint is None:
            # Since we cannot generate random values without any rules, raise RuntimeError
            raise RuntimeError("For generating the random x, you need to attach/define the constraint")
        return self.constraint.get_random(normalize=normalize, **kwargs)

    def get_random_agent(self, **kwargs):
        return self.__class__(x=self.get_random_x(normalize=False), constraint=self.constraint, **kwargs)
        # new_x = self.get_random_x(normalize=False)
        # current_constraint = self.constraint
        # res = self.__class__(x = self._x, constraint = current_constraint, **kwargs)
        # res.x = new_x # To utilize x.setter
        # return res

    def check_overlap(self, other, tol = 1e-6) -> bool:
        # If not performing local opt, tol should be tight
        assert isinstance(other, type(self))
        return True if np.linalg.norm(self-other) <= tol else False

    def __repr__(self):
        if self._fitness is None:
            return f"{self.label}: {self.x} / Fitness: not updated"
        else:
            return f"{self.label}: {self.x} / Fitness: {self.fitness}"

    def __str__(self):
        return self.__repr__()

    @property
    def fitness(self):
        if self._fitness is None:
            self._fitness = self.evaluate()
        return self._fitness

    @fitness.setter
    def fitness(self, precalculated_fitness=None):
        self._fitness = precalculated_fitness

    @fitness.deleter
    def fitness(self):
        self._fitness = None

    def move(self,
             dx,
             alpha:float = 1,
             maxmove:int = 10,
             use_levy:bool = False,
             levy_scale:float = 0.5):
        '''
        return total walk & random walk

        set alpha=0 for no random walk
        '''
        orig_x = self.x.copy()
        n_trial = 0
        while True:
            n_trial += 1
            if n_trial >= maxmove:
                # raise RuntimeError(f"{self.label}: move failed in {maxmove} trials") # This is too harsh
                if n_trial >= 2*maxmove:
                    warnings.warn(f"{self.label}: move finally failed in {2*maxmove:d} trials, starts from random agents")
                    self.x = self.get_random_x(normalize=False)
                else:
                    warnings.warn(f"{self.label}: This dx generates too unstable structures. Use only random perturbations")
                    dx = np.zeros(dx.shape)

            if use_levy:
                _M = self.get_random_x(normalize=True)
                # 1. 각 element (x, y, z)에 Levy flight 적용 --> 근데 문제점이.. 원자 수가 100개인 재료의 경우
                # 여기서 Levy flight 적용하면 300개 중 한 개는 거의 다 1000 보다 큰 값들이 나옴. alpha=0.1 이라 하면 100Å이 넘음
                # random_walk = alpha * np.sign(_M) * levy.rvs(size=_M.shape, loc=1, scale=levy_scale)

                # 2. 그래서, 그 대신 Levy flight를 scalar 값으로 두고, random walk에 적용하는 것으로 변경
                random_walk = alpha * _M * levy.rvs(size=1, loc=1, scale=levy_scale)
            else:
                random_walk = alpha * self.get_random_x(normalize=True)

            # print(orig_x, dx, random_walk)
            new_x = orig_x + dx + random_walk
            if self.check_bc(new_x):
                self.x = new_x
                return dx + random_walk, random_walk
            else:
                continue

    @abstractmethod
    def evaluate(self) -> float:
        '''
        Must define this in child class
        : Evaluate the fitness
        '''
        pass

    @staticmethod
    def get_distance(agent_A, agent_B) -> float:
        '''
        Needs overriding in some cases.
        '''
        return np.linalg.norm(agent_A.x - agent_B.x)

    @property
    @abstractmethod
    def compatible_constraint(self):
        '''
        Must define this in child class
        : return which constraint is compatible with this agent
        '''
        # ex) return [NumConstraint]
        pass


class ASEAgent(BaseAgent):
    '''
    Here, argument x should be ase.Atoms instance

    self.x <- positions
    atoms <- atoms, (_x <- atoms in backend)
    '''

    def __init__(self,
                 x: Atoms = None,
                 label: Optional[str] = None,
                 label_generation: Optional[str] = None,
                 constraint=None,
                 match_xy_only=False,
                 basedir: str = None,  # Base directory (Calculations will be held in [basedir]/[label] directory)
                 ):
        self._x = x
        self._label = label if label is not None else "".join([choice(ascii_letters) for i in range(10)])
        self.label_generation = label_generation
        self._fitness = None
        self.match_xy_only = match_xy_only
        self.basedir = basedir

        if constraint is None:
            self.constraint = None
        else:
            for cc in self.compatible_constraint:
                if (not isinstance(constraint, cc)) and (constraint is not cc):
                    raise RuntimeError("Input constraint is not compatible with this agent...")
            self.constraint = constraint

    @property
    def atoms(self):
        return self._x

    @atoms.setter
    def atoms(self, new: Atoms):
        self.x = new  # using x.setter

    @atoms.deleter
    def atoms(self):
        self._x = None
        self._fitness = None

    @property
    def x(self):
        return self._x.get_positions()

    @x.setter
    def x(self, new_x):
        if isinstance(new_x, Atoms):
            model = new_x
            model.calc = self._x.calc
            model.set_constraint(self._x.constraints)

            if not np.all(model.pbc):
                non_pbc_axis = np.where(np.invert(model.pbc))[0]
                model.center(axis=non_pbc_axis)  # put into center!
            self._x = model
        else:
            assert self.x.shape == new_x.shape, f"Shape not matching!!!" \
                                                f"We needs {self.x.shape}, but you put {new_x.shape}"
            if np.allclose(self.x, new_x):
                # Not updating atoms
                return None

            self._x.set_positions(new_x, apply_constraint=True)
            if not np.all(self._x.pbc):
                non_pbc_axis = np.where(np.invert(self._x.pbc))[0]
                self._x.center(axis=non_pbc_axis)
        self._x.wrap()  # wrapped, only for pbc axis
        self._fitness = None  # fitness should be re-calculated

    @x.deleter
    def x(self):
        self._x = None
        self._fitness = None

    @property
    def calc(self):
        return self._x.calc

    @calc.setter
    def calc(self, calculator):
        self._x.calc = calculator

    @calc.deleter
    def calc(self):
        self._x.calc = None

    @property
    def desc(self):
        return self._x.arrays["node_features_inv"]

    @desc.setter
    def desc(self, new_desc):
        self._x.arrays["node_features_inv"] = new_desc
        # raise RuntimeError("setting desc is not permitted")

    @desc.deleter
    def desc(self):
        del self._x.arrays["node_features"]
        del self._x.arrays["node_features_inv"]
        self._fitness = None # need calculation!

    def __add__(self, other: Union[Atoms, np.ndarray]) -> np.ndarray:
        cur_pos = self._x.get_positions()
        if isinstance(other, Atoms):
            to_add = other.get_positions()
        elif isinstance(other, ASEAgent):
            to_add = other.atoms.get_positions()
        else:
            to_add = other
        cur_pos += to_add
        return cur_pos

    def __radd__(self, other: Union[Atoms, np.ndarray]) -> np.ndarray:
        return self.__add__(other)

    def __sub__(self, other) -> np.ndarray:
        assert isinstance(other, type(self)), f"{other} is not an instance of {type(self)}"
        assert (self._fitness is not None) and (other._fitness is not None), \
            "Both self and other need evaluations, in order to get the descriptor"

        return get_d_pos_wrapper_inv(
            atoms_A=other.atoms,
            atoms_B=self.atoms,
            descA_inv=other.desc,
            descB_inv=self.desc,
            max_allowance=1,
            pbc = np.any(self.atoms.pbc),
            match_xy_only=self.match_xy_only)

    @property
    def compatible_constraint(self):
        return [ASEConstraint]

    def evaluate(self):
        if self._fitness is None:
            self._fitness = self._x.get_potential_energy()/len(self._x)
        assert "node_features_inv" in self._x.arrays, \
            "The calculator you are using now seems not compatible."
        return self._fitness

    def get_random_agent(self, **kwargs):
        '''overriding!!!'''
        return self.__class__(
            x = self.constraint.get_random(normalize=False, return_aseobj=True),
            constraint = self.constraint,
            **kwargs)

    def check_overlap(self, other, tol=0.5) -> bool:
        assert isinstance(other, type(self))
        return True if (np.max(np.linalg.norm(other-self, axis=1))<tol) else False

    @staticmethod
    def get_distance(agent_A, agent_B) -> float:
        '''
        Overriding! using desc instead of position itself
        '''
        # 1) using mean. for global desc
        # return np.linalg.norm(np.mean(agent_A.desc, axis=0) - np.mean(agent_B.desc, axis=0))

        # 2) Align & compare
        _D = cdist(agent_A.desc, agent_B.desc, metric="euclidean")
        a_perm_i, b_perm_i = linear_sum_assignment(_D)
        return np.sqrt(np.mean(np.sum((agent_A.desc[a_perm_i]-agent_B.desc[b_perm_i])**2, axis=1)))

class ASEAgent_localopt(ASEAgent):
    '''
    When using ASE optimizers (ex. QuasiNewton, FIRE, BFGS, GPMin...)

    Here, you need to attach the optimizer, to self.optimizer [Compulsory]
    Also, if you want to change the force convergence criteria, set self.fmax attribute to any other positive float
    '''

    def __init__(self,
                 optimizer = QuasiNewton,
                 fmax = 1e-2,
                 max_steps = 300,
                 save_local_opt = True,
                 save_local_opt_directory_name = "local_opts",
                 log_local_opt:bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer  # Needs to attach the optimizer!
        self.fmax = fmax
        self.max_steps = max_steps
        self.save_local_opt = save_local_opt
        self.save_local_opt_directory_name = save_local_opt_directory_name
        self.log_local_opt = log_local_opt # in stderr

    def get_random_agent(self, **kwargs):
        '''overriding!!!'''
        res = super().get_random_agent(**kwargs)
        res.optimizer = self.optimizer
        res.fmax = self.fmax
        res.max_steps = self.max_steps
        res.save_local_opt = self.save_local_opt
        return res

    def evaluate(self):
        current_dir = os.getcwd()
        try:
            if self.save_local_opt:
                if self.basedir is not None:
                    os.chdir(self.basedir)
                if not os.path.exists(self.save_local_opt_directory_name):
                    os.mkdir(self.save_local_opt_directory_name)
                os.chdir(self.save_local_opt_directory_name)

                while True:
                    if os.path.exists(self.label):
                        overlapped = True
                        self.label += "_"
                    else:
                        overlapped = False
                    if not overlapped:
                        break

                traj = f"{self.label}.traj"
            else:
                traj = None

            ## original way
            # dyn = self.optimizer(self._x, trajectory = traj_name)
            # dyn.run(fmax = self.fmax, steps = self.max_steps)
            # E = super().evaluate()
            # os.chdir(current_dir)
            # return E

            ## when using MLIPs (taking considerations of failure)
            if self.log_local_opt:
                _logfile = "-"
            else:
                _logfile = None
            dyn = self.optimizer(self._x, logfile = _logfile, trajectory=traj)  # logfile="-" if you want stderr

            conv = False
            for i in range(self.max_steps):
                dyn.fmax = self.fmax
                
                if dyn.nsteps == 0:
                    try:
                        if _ASE_OPT_NEEDS_GRAD:
                            g0 = dyn.optimizable.get_gradient()
                            dyn.log(g0)
                            dyn.call_observers()
                            conv = dyn.converged(g0)
                        else:
                            dyn.optimizable.get_forces()
                            dyn.log()
                            dyn.call_observers()
                            conv = dyn.converged()
                    except:
                        conv = False
                        break # break due to the error
                if conv:
                    break # break since it already meets convergence criteria

                # if not conv -> continue
                try:
                    dyn.step()
                    dyn.nsteps += 1
                    if _ASE_OPT_NEEDS_GRAD:
                        g = dyn.optimizable.get_gradient()
                        dyn.log(g)
                        dyn.call_observers()
                        conv = dyn.converged(g)
                    else:
                        dyn.log()
                        dyn.call_observers()
                        conv = dyn.converged()
                except:
                    conv = False
                    break # break due to the error
                if conv:
                    break

            if dyn.trajectory is not None:
                dyn.trajectory.close()

            if (not conv) and (dyn.nsteps >= self.max_steps):
                print(f"Agent:{self.label:s} conv failed since # step reaches max_steps ({self.max_steps:d}).")
                # if not converged -> will be handled later

            return super().evaluate() if conv else None
            # return None if not converged. -> will be handled later (move method)

        finally:
            os.chdir(current_dir)

    def move(self,
             dx,
             alpha:float      = 0.5,
             maxmove:int      = 10,
             use_levy:bool    = False,
             levy_scale:float = 0.5):
        '''
        overriding -- taking considerations of local opt failure

        move self.atoms & return total walk & random walk

        Unlike other agent, perform local opt here
        '''
        orig_x = self.x.copy()
        n_trial = 0
        while True:
            n_trial += 1
            if n_trial >= maxmove:
                # raise RuntimeError(f"{self.label}: move failed in {maxmove} trials")
                if n_trial >= 2*maxmove:
                    warnings.warn(f"{self.label}: move finally failed in {2*maxmove:d} trials, starts from random agents")
                    self.x = self.get_random_x(normalize=False)
                else:
                    warnings.warn(f"{self.label}: This dx generates too unstable structures. Use only random perturbations")
                    dx = np.zeros(dx.shape)

            if use_levy:
                _M = self.get_random_x(normalize=True)
                random_walk = alpha * _M * levy.rvs(size=1, loc=1, scale=levy_scale)
            else:
                random_walk = alpha * self.get_random_x(normalize=True)

            new_x = orig_x + dx + random_walk

            self.x = new_x
            prev_fitness = self._fitness

            model = self.atoms.copy()
            model.calc = self.constraint.conditioner
            dyn = self.optimizer(model, logfile=None)
            dyn.run(fmax=1e-2)
            self.x = model

            if self.evaluate() is None:
                # local opt failed
                self.x = orig_x
                self._fitness = prev_fitness
                continue
            else:
                # self.x is already changed
                # self._fitness is evaluated
                pass

            if self.check_bc():
                return dx + random_walk, random_walk
            else:
                self.x = orig_x
                self._fitness = prev_fitness
                continue

class ASEAgent_internal_localopt(ASEAgent):
    '''
    When using internal optimization process such as VASP geometry optimization

    '''
    def __init__(self,
                 save_local_opt = True,
                 save_local_opt_directory_name = "local_opts",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_local_opt = save_local_opt
        self.save_local_opt_directory_name = save_local_opt_directory_name

    def evaluate(self):
        current_dir = os.getcwd()
        if self.basedir is not None:
            os.chdir(self.basedir)

        if self.save_local_opt:
            if not os.path.exists(self.save_local_opt_directory_name):
                os.mkdir(self.save_local_opt_directory_name)
            os.chdir(self.save_local_opt_directory_name)
            os.mkdir(self.label)
            os.chdir(self.label)
        self._fitness = self._x.get_potential_energy() # geometry optimization is performed here
        # and self._x will be updated automatically if you use ASE calculators (ex. ase.calculators.vasp.Vasp)
        os.chdir(current_dir)
        return self._fitness

class TfuncAgent1(BaseAgent):
    @property
    def compatible_constraint(self):
        return [NumConstraint]

    # Test function #1, Ackley function
    def evaluate(self):
        '''
        Ackley function

        * BC : [-5, 5]
        * GM : f(0, 0)=0
        '''
        x, y = self.x
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(
            0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20


class TfuncAgent2(TfuncAgent1):
    # Test function #2, Himmelblau function. Same boundary condition! :D
    def evaluate(self):
        '''
        himmelblau function

        * BC : [-5, 5]
        * GM : f(3, 2)=0, f(-2.805118, 3.131312)=0, f(-3.779310, -3.283186)=0, f(3.584428, -1.848126)=0
        '''
        x, y = self.x
        return (x**2+y-11)**2 + (x+y**2-7)**2


class TfuncAgent3(TfuncAgent1):
    # Test function #3, 2D Rastrigin function. (A=10)
    def evaluate(self):
        '''
        Rastrigin function

        * BC : [-5.12, 5.12]
        * GM : f(0,0) = 0
        '''
        A = 10
        return A * len(self.x) + np.sum([x_i ** 2 - A * np.cos(2 * np.pi * x_i) for x_i in self.x], axis=0)

class TfuncAgent4(TfuncAgent1):
    # Test function #4, 2D Eggholder function.
    def evaluate(self):
        '''
        Eggholder function

        * BC : [-512, 512]
        * GM : f(512,404.2319)=-959.6407
        '''
        x, y = self.x
        return -(y+47)*np.sin(np.sqrt(np.abs(x/2+y+47)))-x*np.sin(np.sqrt(np.abs(x-y-47)))

class TfuncAgent1_localopt(BaseAgent):
    @property
    def compatible_constraint(self):
        return [NumConstraint]

    # Test function #1, Ackley function, but this time, let's use local optimization
    def evaluate(self):
        def f(x):
            x, y = x
            return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(
                0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

        opts = dict(disp=False, return_all=True, maxiter=500)
        min_res = minimize(method="CG", fun=f, x0=self.x, options=opts)
        self.x = min_res.allvecs[-1]
        return f(self.x)


class TfuncAgent2_localopt(BaseAgent):
    @property
    def compatible_constraint(self):
        return [NumConstraint]

    # Test function #2, Himmelblau function, but this time, let's use local optimization
    def evaluate(self):
        def f(x):
            x, y = x
            return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

        opts = dict(disp=False, return_all=True, maxiter=500)
        min_res = minimize(method="CG", fun=f, x0=self.x, options=opts)
        self.x = min_res.allvecs[-1]
        return f(self.x)



#### ------------------------------------------------------------------------------------------------------------------
#### ------------------------------------------------------------------------------------------------------------------
#### Defining each generation, as 'Population' class
class Population:
    # For each generation
    def __init__(self,
                 agents:list,
                 label:Optional[str] = None,
                 maximize_fitness:bool = False,
                 basedir: str = None):
        self.agents = agents
        self.n_agents = len(agents)
        self._label = label if label is not None else "unknown"
        self.max_fit = maximize_fitness
        self.basedir = basedir

        for a in self.agents:
            a.basedir = basedir

    @property
    def label(self):
        return f"Gen_{self._label:s}"

    def __repr__(self):
        if self._evaluated:
            return f"{self.label}: {self.n_agents} agents / Best Fitness: {self.best} ({self.best_agent._label})"
        else:
            return f"{self.label}: {self.n_agents} agents / Fitness not updated yet"

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self.n_agents

    def __iter__(self):
        for i in range(len(self)):
            yield self.agents[i]

    def __getitem__(self, i):
        if isinstance(i, numbers.Integral):
            if i < -self.n_agents or i >= self.n_agents:
                raise IndexError("Index out of range!")
            return copy.deepcopy(self.agents[i])
        elif not isinstance(i, slice):
            i = np.array(i)
            if len(i) == 0:
                i = np.array([], dtype=int)
            if i.dtype == bool:
                if len(i) != self.n_agents:
                    errmsg = f"Length of mask {len(i)} must equal number of agents {self.n_agents}"
                    raise IOError(errmsg)
                i = np.arange(self.n_agents)[i]
        else:
            return self.__class__(agents = self.agents[i], label = self._label, maximize_fitness = self.max_fit)

        selected_agents = []
        for si in i:
            selected_agents.append(self[si])
        parsed_gen = self.__class__(agents = selected_agents, label = self._label, maximize_fitness = self.max_fit)
        return parsed_gen

    @property
    def best(self):
        if self._evaluated:
            this_popul = [agent.fitness for agent in self.agents]
            return np.max(this_popul) if self.max_fit else np.min(this_popul)
        else:
            return None

    @property
    def _evaluated(self):
        for agent in self.agents:
            if agent._fitness is None:
                return False
        return True

    @property
    def best_agent(self):
        if self._evaluated:
            this_popul = [agent.fitness for agent in self.agents]
            this_index = np.argmax(this_popul) if self.max_fit else np.argmin(this_popul)
            return self.agents[this_index]
        else:
            return None

    @property
    def Ds(self):
        # Distance matrix
        agent_type = self.agents[0]
        N = self.n_agents
        _D = np.zeros((N, N), dtype=float)
        ij = np.triu_indices(N, k=1)

        _D_eff = []
        for i, j in zip(*ij):
            _D_eff.append(agent_type.get_distance(self.agents[i], self.agents[j]))
        _D[ij] = _D_eff
        _D += _D.T
        _D /= np.median(_D_eff) # normalize by median value
        return _D

        # for i in range(N):
        #     for j in range(N):
        #         if j > i:
        #             _D[i, j] = agent_type.get_distance(self.agents[i], self.agents[j])
        #         elif i == j:
        #             continue
        #         else:
        #             _D[i, j] = _D[j, i]
        # return _D

    def update_fitnesses(self):
        _Fs = []
        for agent in self.agents:
            f = agent.fitness
            assert f is not None, f"{self.label} evaluation not done well"
            _Fs.append(f)
        return np.array(_Fs)

    @property
    def Is(self): # Intensity per agent: (N,) matrix
        return np.array([agent.fitness for agent in self.agents]) if self._evaluated else None

#### ------------------------------------------------------------------------------------------------------------------
#### ------------------------------------------------------------------------------------------------------------------
#### Defining the Firefly algorithm!
#### Make first generation (in Population instance) which contains several agents (Child class of BaseAgent class)

class FA:
    def __init__(self,
                 mother_agent,
                 alpha:float = 0.5,  # randomization parameter
                 beta_0:float = 1,  # absolute attractiveness # beta_0 = 1 is recommended
                 gamma:float = 0.02,  # absorption coefficient
                 gamma_in_percent:Optional[float] = None,  # gamma = -ln(1-gamma_in_percent)
                 use_levy:bool = False,
                 levy_scale:float = 0.5,
                 population_size:int = 30,
                 verbosity:int = 2,
                 logfile:Optional[str] = None,
                 restart:Optional[str] = None,  # filename of pickle file
                 history:Optional[str] = "history.pickle", # filename of pickle file which contains all generations
                 label:str = "Firefly",
                 maximize_fitness:bool = False,
                 ):
        '''
        FireFly Algorithms :D
        when use_levy=True, using Levy flight for random walk

        :param mother_agent: Which type of agent do you want to use!?
        :param alpha: randomization parameter / Default=0.1
        :param beta_0: Attractiveness at r=0. If None, use Intensity value / Default=1
        :param gamma: Absorption coefficient / Default=0.01
        :param gamma_in_percent: if x_j-x_i has norm value of '1', (for ASEAgnet, when norm(d_pos) is 1Å)
                              x_i will move forward to x_j when x_i has fitness bigger than (1-gamma_in_percent)*x_j
                              That is: In x_i's point of view, x_j's fitness is reduced in (100 * gamma_in_percent) %
                              gamma = 1 | gamma_in_percent = 0.632 (63.2% reduced)
                              gamma = 0.1 | gamma_in_percent = 0.10 (10% reduced)
                              gamma = 0.01 | gamma_in_percent = 0.01 (1% reduced) # Default
                              gamma = 0.001 | gamma_in_percent = 0.001 (0.1% reduced)
                              |__ This has higher priority than gamma

        :param use_levy: Use Levy flight for random walk: alpha*(rand-1/2)*Levy
        :param levy_scale: see rvs method https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy.html
        :param population_size: Number of agents (i.e. N_fireflies).
                                Too many agents also can occur convergence problems.
        :param verbosity: Choose the logging level, among '0(error) > 1(warning) > 2(info) > 3(debug)'
        :param logfile: Filename of log. if None, it will not write any files
        :param restart: If None, FA starts from the scratch.
        :param history: If None, FA will not save all generations
        :param maximize_fitness: If True, FA Maximize the fitness / If False, FA Minimize the fitness
        '''
        self.mother_agent = mother_agent
        self._label = str(label)  # TODO: label for FA is not needed.. maybe?
        self.alpha = alpha
        self.beta_0 = beta_0
        self.gamma = gamma
        self.use_levy = use_levy
        self.levy_scale = levy_scale


        if gamma_in_percent is not None:
            assert 0 < gamma_in_percent < 1, "gamma_in_percent must be between 0 and 1"
            self.gamma = -np.log(1 - gamma_in_percent)
            # self.gamma = 1 / Gamma_in_distance ** 2

        self.population_size = int(population_size)
        self.verbosity = int(verbosity)
        self.maximize_fitness = maximize_fitness
        assert self.verbosity in [0, 1, 2, 3],\
            "Verbosity level should be one among '0(Error/Important) < 1(warning) < 2(info) < 3(debug)'"

        self._root_path = os.getcwd()
        self._cbest = None

        if logfile is None:
            self.logfile = logfile
        elif os.sep in logfile:
            self.logfile = logfile
        else:
            assert isinstance(logfile, str), "logfile name should be string"
            self.logfile = os.path.join(self._root_path, logfile)

        # Refresh the files & write always
        if self.logfile is not None:
            with open(self.logfile, 'w') as fo:
                print("", file=fo, end="")

        self.history = history

        self.add_log(f"NARA performed on: {datetime.now()}\n\n", level=0) # Always write
        self.add_log(">>> FA params & logging settings done. Let's evaluate & evolve!\n\n", level=2)

        params_info = f'''
< Firefly algorithms parameters >
    * label  = {self.label}

    * alpha  = {self.alpha}
    * beta_0 = {self.beta_0}
    * gamma  = {self.gamma}

    * population_size        = {self.population_size}

    * verbosity = {self.verbosity} # 0(error) < 1(warning) < 2(info) < 3(debug) \n\n
        '''
        self.add_log(params_info, level = 2)

        self.restart = "ongoing.pickle"
        if restart is not None:
            self.restart = restart
            if isinstance(restart, str):# filename
                if os.path.exists(restart):
                    self.add_log(f">>> Read initial generation & lowest minima from {restart} files!\n\n", level=2)
                    self.load_population(restart) # Define self.current_generation & self.gbest_history
                else:
                    self.add_log(">>> Restart file not found. Starts from the scratch.\n\n", level=1)
                    self.current_gen = None
            else:
                self.add_log(">>> Restart file should be given in str type. Starts from the scratch.\n\n", level=1)
                self.current_gen = None
        else:
            self.current_gen = None

        if self.current_gen is None:
            # initiate
            self.n_gen = 1
            self.gbest_history = []

            _agents = []
            for i in range(self.population_size):
                new_agent = self.mother_agent.get_random_agent(label=f"{i+1}", label_generation=f"Gen_{self.n_gen}")
                _agents.append(new_agent)
            self.current_gen = Population(agents = _agents,
                                          label = f"{self.n_gen}",
                                          maximize_fitness = self.maximize_fitness,
                                          basedir = os.getcwd())

    @property
    def label(self):
        return self._label

    @property
    def agents(self):
        return self.current_gen.agents

    @property
    def gbest_history_values(self):
        return [agent.fitness for agent in self.gbest_history]

    def is_reviewed_already(self):
        if not self.current_gen._evaluated:
            return False # 1) 계산조차 안 됐는데?
        if self._cbest is None:
            return False
        elif self._cbest.label != self.current_gen.best_agent.label:
            return False # 2) 계산은 됨. 근데 리뷰는 안됨
        return True # 3) 계산과 리뷰 둘 다 됨

    def __repr__(self):
        name = "Firefly searched" if self.label is None else self.label
        return f"{name:s} | n_gen={self.n_gen} | Best fitness={self.current_gen.best}"

    @property
    def _evaluated(self):
        return self.current_gen._evaluated

    def evaluate(self):
        '''
        This is separated from __init__ method, since the "evaluation" can be expensive in some cases
        Evaluate & Update histories
        :return: True if found new g_best, else False
        '''
        if self.current_gen._evaluated:
            self.add_log("Evaluation is already done. Just use the calculated values", level=1)
        else:
            self.add_log(f">>> Evaluate the generation: {self.current_gen.label}...\n\n", level=2)
            self.current_gen.update_fitnesses()

        this_best = self.current_gen.best
        if this_best is None:
            self.current_gen.update_fitnesses() # how can this be possible?

        is_reviewed_already = self.is_reviewed_already()
        if is_reviewed_already:
            assert len(self.gbest_history)!=0, "If already reviewed, gbest_history can't be empty"
            return self.current_gen.best_agent.label == self.gbest_history[-1].label

        if len(self.gbest_history)==0:
            gbest_renewed = True
            self.add_log(f">>> *** 1st g_best set! {self.current_gen.best_agent.label} "\
                         f"| Fitness: {self.current_gen.best}", level=2)
            self.add_log(f"{self.current_gen.best_agent.x}", level=3)
        else:
            s = (-1)**(self.maximize_fitness) # 1 if False, -1 if True
            gbest_renewed = s*(self.gbest_history_values[-1]) > s*(this_best)
            if gbest_renewed:
                self.add_log(f">>> *** New g_best found! {self.current_gen.best_agent.label} "\
                             f"| Fitness: {self.current_gen.best}", level=2)
                self.add_log(f"{self.current_gen.best_agent.x}", level=3)

        if gbest_renewed:
            this_best_agent = self.current_gen.best_agent # TODO-check: do we need copy.deepcopy?
            self.gbest_history.append(this_best_agent)

        # self.logger.info(f"\n\n>>> Gen {self.n_gen} evaluated: {self.population_size} agents. Best: {this_best}\
 # ({self.current_gen.best_agent.label})\n\n")
        cur_gen_best_info = f">>> Gen {self.n_gen} evaluated: {self.population_size} agents. Best: {this_best} " \
                            f"({self.current_gen.best_agent.label})\n\n"
        self.add_log(cur_gen_best_info, level = 2)

        each_agent_info = f"\n< {self.current_gen.label}: Agents summary >"
        for a in self.current_gen.agents:
            each_agent_info += f"\n @ {a.label} | f(X): {a.fitness}"

        each_agent_details = f"\n< {self.current_gen.label}: Agents details >"
        for a in self.current_gen.agents:
            each_agent_details += f"\n @@ {a.label} | X: {a.x}"

        self.add_log(each_agent_info, level=2)
        self.add_log(each_agent_details, level = 3)
        self.add_log("\n\n", level=2)
        self._cbest = self.current_gen.best_agent
        return gbest_renewed

    def evolve(self, frozen_index = None) -> bool:
        if not self.current_gen._evaluated:
            self.add_log("You need to evaluate the generation first. use evaluate method!", level=0)
            return None

        self.add_log(f">>> Generation {self.n_gen} evolving....", level=2)
        st = time.time()
        if self.history is not None:
            self.append_history(self.history)

        if frozen_index is None:
            frozen_index = []

        Is = self.current_gen.Is.copy() # Intensities
        if not self.maximize_fitness:
            Is *= -1 # opposite direction! shape: (N, )


        # beta_0 doesn't have to be multiplied here. It doesn't affect the trends of attractions
        Is = np.array(Is, dtype=float)
        Ds = self.current_gen.Ds

        debug_mes = ""
        self.n_gen += 1
        all_raw_dx = []

        # 1. calculate where to go
        for i in range(self.population_size):
            agent_i = self.current_gen[i]
            agent_i.label_generation = f"Gen_{self.n_gen}" # For ASEAgent, evaluation is performed during the 'move'

            if i in frozen_index:
                self.add_log(f"\n {agent_i.label} doesn't move. [freeze]", level=2)
                dx = np.zeros(self.current_gen[i].x.shape)

            else:
                dx = np.zeros(self.current_gen[i].x.shape)
                for j in range(self.population_size):
                    if i == j:
                        continue
                    agent_j = self.current_gen[j]
                    if Is[i] < Is[j]: # since we unified the optimization direction above
                        _dx = agent_j - agent_i
                        # dx += _dx * (self.beta_0 * np.exp(-self.gamma * np.linalg.norm(_dx)**2))
                        # dx += _dx * (self.beta_0 * np.exp(-self.gamma * (_dx**2).sum())) # same but more efficient
                        dx += _dx * (self.beta_0 * np.exp(-self.gamma * Ds[i,j]**2))
            all_raw_dx.append(dx)

        New_agents = []
        all_total_dx, all_rand_dx = [], []
        # 2. Actual movement
        for i in range(self.population_size):
            dx = all_raw_dx[i]
            agent_i = self.current_gen[i]
            agent_i.label_generation = f"Gen_{self.n_gen}"
            if i in frozen_index:
                total_dx = rand_dx = np.zeros(self.current_gen[i].x.shape)
            else:
                total_dx, rand_dx = agent_i.move(dx,
                                                 alpha = self.alpha,
                                                 use_levy = self.use_levy,
                                                 levy_scale = self.levy_scale
                                                 )

            self.add_log(f"dx: {total_dx} (beta*(xj-xi): {dx} + random_walk: {rand_dx})", level=3)

            New_agents.append(agent_i)
            all_total_dx.append(total_dx)
            all_rand_dx.append(rand_dx)  # related to alpha
        self.add_log("\n\n", level=2)
        self.current_gen = Population(agents = New_agents,
                                      label = f"{self.n_gen}",
                                      maximize_fitness = self.maximize_fitness,
                                      basedir = self.current_gen.basedir)
        self.add_log(f">>> Generation {self.n_gen-1} evolution done in {time.time()-st:.2f} seconds", level=2)
        return all_total_dx, all_raw_dx, all_rand_dx


    def autorun_gbest_only(self,
                max_n_gen = None,
                patience:int = 100,
                ) -> bool:
        '''
        :param max_n_gen:

        :param patience: convergence criteria of Firefly algorithm searches (for global min)
        :return: bool, True if it converged
        '''

        if max_n_gen is None:
            max_n_gen = np.inf
        gen_no_improve = 0
        while True:
            self.add_log(f"Gen [{gen_no_improve}/{patience}] | from total [{self.n_gen}/{max_n_gen}]", level=2)
            if self.n_gen > max_n_gen:
                self.prepare_to_exit("Reached max_n_gen without full convergence.", level=1)
                return False # escape the autorun. Finished properly, but not converged within max_n_gen
            _gbest_renewed = self.evaluate()
            gen_no_improve = 0 if _gbest_renewed else (gen_no_improve + 1)
            if gen_no_improve >= patience:
                if len(self.gbest_history)==1:
                    self.prepare_to_exit("FA couldn't find any other gbest", level=0)
                    return False # finished properly = False -> This means... FA couldn't find any other g_bests
                else:
                    self.prepare_to_exit(f"End FA in {self.n_gen} generations.", level=2)
                    return True # escape the loop when reached the patience
            self.save_population(self.restart)
            if self.n_gen != max_n_gen:
                self.evolve() # self.n_gen += 1 in here

    #### Not done yet
    def autorun_locals(self,
                       max_n_gen=None,
                       n_clusters: int = 4,
                       patience: int = 50,
                       patience_for_locmins: Optional[int] = None):

        if patience_for_locmins is None:
            patience_for_locmins = patience
        if max_n_gen is None:
            max_n_gen = np.inf

        cluster_agents = {}  # {label:agent}
        cluster_counters = {}  # {label:counter}
        deleted_c_best_agents = {}  # {label:[counter, agent]}

        gen_no_improve = 0

        while True:
            if self.n_gen > max_n_gen:
                self.prepare_to_exit("Reached max_n_gen without full convergence.", level=1)
                return False

            _gbest_renewed = self.evaluate()
            prev_and_new = self.agents + [cluster_agents[label] for label in sorted(cluster_agents)]
            prev_and_new_desc = np.array([agent.desc for agent in prev_and_new], dtype=float)
            Fs = np.array([agent.fitness for agent in prev_and_new], dtype=float)

            kmeans = KMeans(n_clusters=n_clusters, n_init="auto", max_iter=300, tol=1e-4, random_state=None)
            labels = kmeans.fit_predict(prev_and_new_desc)

            # kmeans = KMeans(...).fit(prev_and_new_desc)
            # labels, centers = kmeans.labels_, kmeans.cluster_centers_

            prev_cluster_index = {}
            if cluster_agents:
                start_idx = len(self.agents)
                sorted_keys = sorted(cluster_agents)
                for i in range(len(sorted_keys)):
                    lab = sorted_keys[i]
                    prev_cluster_index[lab] = labels[start_idx + i]

            new_cluster_agents = {}  # {label: agent}
            new_cluster_agents_ci = {}  # {label: cluster index}
            for c_i in range(n_clusters):
                indices = np.array([i for i, l in enumerate(labels) if l == c_i], dtype=int)
                assert len(indices) != 0, "Plz report developer"
                if self.maximize_fitness:
                    best_index = indices[np.argmax(Fs[indices])]
                else:
                    best_index = indices[np.argmin(Fs[indices])]
                new_agent = prev_and_new[best_index]
                new_cluster_agents[new_agent.label] = new_agent
                new_cluster_agents_ci[new_agent.label] = c_i

            updated_agents = {}
            updated_counters = {}
            for _, new_agent in new_cluster_agents.items():
                overlap_found = False
                for prev_label, prev_agent in cluster_agents.items():
                    if new_agent.check_overlap(prev_agent):
                        overlap_found = True
                        if new_agent.label == prev_agent.label:
                            updated_agents[prev_label] = prev_agent
                            updated_counters[prev_label] = cluster_counters[prev_label] + 1
                        else:
                            if prev_label not in deleted_c_best_agents:
                                deleted_c_best_agents[prev_label] = [cluster_counters[prev_label], prev_agent]
                            updated_agents[new_agent.label] = new_agent
                            updated_counters[new_agent.label] = 1
                        break
                if not overlap_found:
                    updated_agents[new_agent.label] = new_agent
                    updated_counters[new_agent.label] = 1

            for prev_label, prev_agent in cluster_agents.items():
                if not any(new_cluster_agents[cand].check_overlap(prev_agent) for cand in new_cluster_agents):
                    if prev_label not in deleted_c_best_agents:
                        deleted_c_best_agents[prev_label] = [cluster_counters[prev_label], prev_agent]

            cluster_agents = updated_agents
            cluster_counters = updated_counters

            if cluster_counters and min(cluster_counters.values()) > patience_for_locmins:
                self.prepare_to_exit(f"Local convergence achieved after generation {self.n_gen}.", level=2)
                return True

            gen_no_improve = 0 if _gbest_renewed else gen_no_improve + 1
            if gen_no_improve >= patience:
                if len(self.gbest_history) <= 1:
                    self.add_log("FA couldn't find any new global best.", level=0)
                    # return False
                else:
                    self.add_log(f"Global convergence achieved after {self.n_gen} generations.", level=2)
                    # return True

            with open("Cluster.info", 'a') as fo:
                for lab, count in cluster_counters.items():
                    fo.write(
                        f"{lab} | {count}/{patience_for_locmins} times | Fitness: {cluster_agents[lab].fitness:.6f}\n")
            self.save_population(self.restart)

            frozen_index = []
            for lab, count in cluster_counters.items():
                if count >= patience_for_locmins and lab in new_cluster_agents_ci:
                    frozen_index += list(np.where(labels == new_cluster_agents_ci[lab])[0])
            self.evolve(frozen_index=frozen_index)


    def add_log(self, message:str, level = None):
        _write = False if self.logfile is None else True
        _print = True if level <= self.verbosity else False

        if _print:
            if _write:
                with open(self.logfile, 'a') as fo:
                    print(message, file=fo)
            else:
                print(message)

    def save_population(self, restart_filename:str):
        agent_states = []
        for agent in self.current_gen.agents:
            state = {
                "x": agent.x,
                "desc": agent.desc,
                "label": agent._label,
                "label_generation": agent.label_generation,
                "fitness" : agent.fitness,
            }
            agent_states.append(state)
        gbest_states = []
        for agent in self.gbest_history:
            state = {
                "x": agent.x,
                "desc": agent.desc,
                "label": agent._label,
                "label_generation": agent.label_generation,
                "fitness": agent.fitness,
            }
            gbest_states.append(state)
        data = {
            "population": agent_states,
            "gbest_history": gbest_states,
            "n_gen": self.n_gen,
        }
        with open(restart_filename, 'wb') as fo:
            pickle.dump(data, fo)

    def load_population(self, restart_filename:str):
        with open(restart_filename, 'rb') as fi:
            data = pickle.load(fi)
        agent_states = data.get("population", [])
        gbest_states = data.get("gbest_history", [])
        self.n_gen = data.get("n_gen", 1)

        agents = []
        for state in agent_states:
            agent = self.mother_agent.get_random_agent(label=state["label"], label_generation=state["label_generation"])
            agent.x = state["x"]
            agent.desc = state["desc"]
            agent._fitness = state["fitness"]
            agents.append(agent)

        basedir = self.current_gen.basedir if hasattr(self, "current_gen") and self.current_gen else os.getcwd()
        self.current_gen = Population(agents=agents, label=f"gen_{self.n_gen}", maximize_fitness=self.maximize_fitness,
                                      basedir=basedir)

        gbest_agents = []
        for state in gbest_states:
            agent = self.mother_agent.get_random_agent(label=state["label"], label_generation=state["label_generation"])
            agent.x = state["x"]
            agent.desc = state["desc"]
            agent._fitness = state["fitness"]
            gbest_agents.append(agent)
        self.gbest_history = gbest_agents

    def append_history(self, history_filename:str):
        atoms_list = []
        for agent in self.current_gen:
            atoms = agent.atoms.copy()
            atoms.info["label"] = agent.label
            calc = SinglePointCalculator(
                energy  = agent.atoms.get_potential_energy(),
                forces  = agent.atoms.get_forces(),
                stress  = None,
                magmoms = None,
                atoms   = atoms)
            atoms.calc = calc
            atoms_list.append(atoms)

        with open(history_filename, 'ab+') as fi:
            pickle.dump(atoms_list, fi)

    def load_history(self, history_filename:str):
        with open(history_filename, 'rb') as fi:
            while True:
                try:
                    yield pickle.load(fi)
                except EOFError:
                    break

    def prepare_to_exit(self, message:str = None, level=None):
        if self.history is not None:
            all_histories = self.load_history(self.history)
            n = 0
            for last_history in all_histories:
                n += 1
                if len(last_history) != self.population_size:
                    raise IOError(f"{n:d}th Generation has weird population size: {len(last_history):d}")
            if n != self.n_gen:
                assert n+1 == self.n_gen, f"len(history)+1 should be identical to current gen"
                self.append_history(self.history)
        if message is not None:
            self.add_log(message, level = level)                    

### Deprecated Agents
# from scipy.optimize import minimize, Bounds
# class RMSDAgent_trans(BaseAgent):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._reference_atoms = None
#         self._modifying_atoms = None
#
#     @property
#     def compatible_constraint(self):
#         return [NumConstraint]
#
#     @property
#     def reference_atoms(self):
#         if self._reference_atoms is None:
#             raise IOError("You should put the reference_atoms attribute! Evaluation will use it!")
#         elif not isinstance(self._reference_atoms, Atoms):
#             raise IOError("You should put the reference atoms in ase.Atoms obj")
#         return self._reference_atoms
#
#     @reference_atoms.setter
#     def reference_atoms(self, x:Atoms):
#         self._reference_atoms = x.copy()
#
#     @reference_atoms.deleter
#     def reference_atoms(self):
#         self._reference_atoms = None
#
#     @property
#     def modifying_atoms(self):
#         if self._modifying_atoms is None:
#             raise IOError("You should put the modifying_atoms attribute! Evaluation will use it!")
#         elif not isinstance(self._modifying_atoms, Atoms):
#             raise IOError("You should put the modifying atoms in ase.Atoms obj")
#         return self._modifying_atoms
#
#     @modifying_atoms.setter
#     def modifying_atoms(self, x):
#         self._modifying_atoms = x.copy()
#
#     @modifying_atoms.deleter
#     def modifying_atoms(self):
#         self._modifying_atoms = None
#
#     def evaluate(self):
#         model_a = self.reference_atoms.copy()
#         model_b = self.modifying_atoms.copy()
#
#         def pbc_rmsd(X):
#             _mb = model_b.copy()
#             _mb.translate(X)
#             _ma, _mb = superpose_permute(model_a, _mb)
#             return rmsd_from_array(_ma.get_positions(), _mb.get_positions())
#
#         opts = dict(disp=False, return_all=True, maxiter=500)
#         min_res = minimize(method="CG", fun=pbc_rmsd, x0=self.x, options=opts)
#         model_b.euler_rotate(*min_res.x, center=(0,0,0))
#         self.modifying_atoms = model_b
#         self.x = min_res.x
#         self._fitness = rmsd_from_array(model_a.get_positions(), model_b.get_positions())
#         return self._fitness
#
#
# class RMSDAgent_rot(RMSDAgent_trans):
#     @property
#     def x(self):
#         return self._X
#
#     @x.setter
#     def x(self, new_x):
#         if self.constraint is None:
#             self._X = new_x
#         else:
#             new_comp = []
#             for bc, comp in zip(self.constraint.boundary, new_x):
#                 count_panel = [0, 0, 0]
#                 while True:
#                     if comp < np.min(bc):
#                         comp += 360
#                         count_panel[0] = 1
#                     elif comp > np.max(bc):
#                         if count_panel[0] == 1:
#                             break
#                         comp -= 360
#                         count_panel[1] = 1
#                     else:
#                         count_panel[2] = 1
#                         break
#                 new_comp.append(comp)
#             self._X = np.array(new_comp, dtype=float)
#         self._fitness = None # fitness should be re-calculated
#
#     @x.deleter
#     def x(self):
#         self._X = None
#         self._fitness = None
#
#     def evaluate(self):
#         model_a = self.reference_atoms.copy()
#         model_b = self.modifying_atoms.copy()
#         model_a.translate(-np.mean(model_a.get_positions(), axis=0))
#         model_b.translate(-np.mean(model_b.get_positions(), axis=0))
#
#         def nonpbc_rmsd(X):
#             _mb = model_b.copy()
#             _mb.euler_rotate(*X, center=(0, 0, 0))
#             _ma, _mb = superpose_permute(model_a, _mb)
#             return rmsd_from_array(_ma.get_positions(), _mb.get_positions())
#
#         opts = dict(disp=False, return_all=True, maxiter=500)
#
#         if self.constraint is not None:
#             lb, ub = np.array(self.constraint.boundary, dtype=float).T
#             bounds = Bounds(lb=lb, ub=ub, keep_feasible=True)
#             min_res = minimize(method="Powell", fun=nonpbc_rmsd, x0=self.x, options=opts, bounds=bounds)
#         else:
#             min_res = minimize(method="CG", fun=nonpbc_rmsd, x0=self.x, options=opts)
#         model_b.euler_rotate(*min_res.x, center=(0,0,0))
#         self.modifying_atoms = model_b
#         self.x = min_res.x
#         self._fitness = rmsd_from_array(model_a.get_positions(), model_b.get_positions())
#         return self._fitness


class LCB_tracker:
    def __init__(self, gnn_calc, lr_coef:dict, kappa = 1, optimizer=None):
        self.gnn_calc = gnn_calc
        self.lr_coef = lr_coef
        self.kappa = kappa
        self.optimizer = optimizer

        self.LCBs_GNN = []
        self.LCBs_atoms_GNN = []
        self.LCBs_Es = []
        self.LCBs_desc = []

    def get_uc(self, atoms):
        _lhat_per_atom = atoms.arrays["lhat_per_atom"]
        uc_per_atom = []
        for atom, _lpa in zip(atoms, _lhat_per_atom):
            a, b = self.lr_coef[atom.symbol]
            uc_per_atom.append(np.max([a*_lpa+b, 0]))
        return np.array(uc_per_atom, dtype=float)

    def add(self, atoms):
        try:
            _E = atoms.get_potential_energy()
            uc_per_atom = self.get_uc(atoms)
        except:
            atoms.calc = self.gnn_calc
            _E = atoms.get_potential_energy()
            uc_per_atom = self.get_uc(atoms)

        _lcb = _E - self.kappa * np.sum(uc_per_atom)
        self.LCBs_GNN.append(_lcb)
        self.LCBs_atoms_GNN.append(atoms)
        self.LCBs_Es.append(_E)
        self.LCBs_desc.append(atoms.arrays["node_features_inv"])

    def check(self, atoms, return_lcb=False):
        try:
            _E = atoms.get_potential_energy()
            uc_per_atom = self.get_uc(atoms)
        except:
            atoms.calc = self.gnn_calc
            _E = atoms.get_potential_energy()
            uc_per_atom = self.get_uc(atoms)
        _lcb = _E - self.kappa*np.sum(uc_per_atom)

        if len(self.LCBs_GNN)==0 or (_lcb < np.min(self.LCBs_GNN)):
            return (True, _lcb) if return_lcb else True
        else:
            return (False, _lcb) if return_lcb else False

    def update_atoms(self, atoms_list, fmax=1e-2):
        self.LCBs_atoms_GNN = atoms_list
        self.update(gnn_calc=self.gnn_calc, lr_coef = self.lr_coef, fmax = fmax)

    def update(self, gnn_calc, lr_coef, atoms_list = None, fmax = 1e-2):
        self.gnn_calc = gnn_calc
        self.lr_coef = lr_coef

        if atoms_list is None:
            atoms_list = self.LCBs_atoms_GNN

        self.LCBs_GNN = []
        self.LCBs_atoms_GNN = []
        self.LCBs_Es = []
        self.LCBs_desc = []

        for i, atoms in enumerate(atoms_list):
            model = atoms.copy()
            model.calc = gnn_calc
            if self.optimizer is not None:
                dyn = self.optimizer(model, logfile = None)
                dyn.run(fmax = fmax)
            _E = model.get_potential_energy()
            _lcb = _E - self.kappa * np.sum(self.get_uc(model))
            self.LCBs_GNN.append(_lcb)
            self.LCBs_Es.append(_E)
            self.LCBs_atoms_GNN.append(model)
            self.LCBs_desc.append(model.arrays["node_features_inv"])

    @property
    def emin(self):
        return np.min(self.LCBs_Es)
    @property
    def lcbs(self):
        return np.array(self.LCBs_GNN, dtype=float)

    @property
    def Es(self):
        return np.array(self.LCBs_Es, dtype=float)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.LCBs_atoms_GNN[index]
        assert len(index)!=0, "If index is not an integer value, it should has length"
        new_tracker = LCB_tracker(self.gnn_calc, self.lr_coef, self.kappa)
        new_tracker.LCBs_GNN = [self.LCBs_GNN[i] for i in index]
        new_tracker.LCBs_atoms_GNN = [self.LCBs_atoms_GNN[i] for i in index]
        new_tracker.LCBs_Es = [self.LCBs_Es[i] for i in index]
        new_tracker.LCBs_desc = [self.LCBs_desc[i] for i in index]
        return new_tracker

    def __delitem__(self, index):
        del self.LCBs_GNN[index]
        del self.LCBs_atoms_GNN[index]
        del self.LCBs_Es[index]
        del self.LCBs_desc[index]

    def __iter__(self):
        return iter(self.LCBs_atoms_GNN)

@torch.no_grad
def fit_lr_valid(valid_atoms, gnn_model, device=None):
    if device is None:
        device = torch.device("cpu")
    unique_elements = set()
    for atoms in valid_atoms:
        unique_elements.update(atoms.get_chemical_symbols())
    unique_elements = sorted(unique_elements)

    x = []
    y = []
    for atoms in valid_atoms:
        species = atoms.get_chemical_symbols()
        E_real = atoms.get_potential_energy()
        _res_dict = gnn_model.predict(atoms, device=device, compute_forces=False)
        E_pred = _res_dict["energy_pred"].item()
        lhat_per_atom = _res_dict["Floss_hat_per_node"].detach().cpu().numpy().flatten()
        assert len(lhat_per_atom) == len(atoms)

        features = []
        for sym in unique_elements:
            lhat_sum = sum(lhat for s, lhat in zip(species, lhat_per_atom) if s==sym)
            count = species.count(sym) # 리스트에 string 값 있는지 체크
            features.extend([lhat_sum, count])
        x.append(features)
        error = np.abs(E_real - E_pred)
        y.append(error)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    params, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    coeffs = {}
    for i, sym in enumerate(unique_elements):
        a_sym = params[2*i]
        b_sym = params[2*i+1]
        coeffs[sym] = (a_sym, b_sym)
    return coeffs

# def fit_lr_valid(valid_atoms, gnn_model, device=None):
#     if device is None:
#         device = torch.device("cpu")
#
#     lhat_lists = []
#     ehat_lists = []
#     real_es = [atoms.get_potential_energy() for atoms in valid_atoms]
#     for atoms in valid_atoms:
#         _res_dict = gnn_model.predict(atoms, device=device, compute_forces = False)
#         ehat_lists.append(_res_dict["energy_pred"].item())
#         lhat_lists.append(_res_dict["Eloss_hat"].item())
#     errs = np.abs(np.array(real_es) - np.array(ehat_lists))
#     a, b = np.polyfit(lhat_lists, errs, 1)
#     return a, b

class FA_AL(FA):
    def __init__(self,
                 mother_agent,
                 target_calc,
                 alpha: float = 0.5,
                 beta_0: float = 0.9,
                 gamma: float = 0.1,
                 population_size: int = 30,
                 verbosity: int = 2,
                 logfile: Optional[str] = "NARA.log",
                 restart: Optional[str] = None,  # filename of pickle file 
                 gnn_filename:str  = "NARA_GAJA.pt",
                 train_xyz_filename:str = "NARA_train.xyz",
                 valid_xyz_filename:str = "NARA_valid.xyz",
                 train_valid_ratio = 0.8,
                 n_init = 30,
                 n_force_init = 5,
                 gnn_config = {},
                 batch_size = 5,
                 valid_batch_size = 1,
                 lcb_kappa = 1,
                 device = None,
                 label = "Firefly",
                 ):

        self.mother_agent = mother_agent
        self.target_calc = mother_agent.calc if target_calc is None else target_calc
        self._label = str(label)

        #### If we have to read atoms from files and there are no tags, we may need to store the sequence of tags
        #### 0: fixed atoms , 1: relaxed atoms, 2: movable atoms

        # m1 = get_random_atoms_with_conditioning(
        #     base_atoms=mother_agent.constraint.motif,
        #     add_info = mother_agent.constraint.addatoms_info,
        #     min_bondinfo = 0,
        #     region = None,
        #     set_region_as_prohibited = False,
        #     mic = mother_agent.constraint.mic,
        #     max_step = 1,
        #     max_iter = 1,
        #     verbose = False,
        #     trajectory = None,
        #     optimizer = QuasiNewton,
        #     tol = 1e-4,
        # )
        # self.tags_record = m1.get_tags()

        self.alpha = alpha
        self.beta_0 = beta_0
        self.gamma = gamma

        self.population_size = int(population_size)
        self.n_init = int(n_init)
        self.n_force_init = n_force_init
        self.verbosity = int(verbosity)
        self.maximize_fitness = False
        assert self.verbosity in [0, 1, 2, 3], \
            "Verbosity level should be one among '0(Error/Important) < 1(warning) < 2(info) < 3(debug)'"

        self._root_path = os.getcwd()
        self._cbest = None

        self.use_levy = False
        self.levy_scale = 0.0

        if logfile is None:
            self.logfile = logfile
        elif os.sep in logfile:
            self.logfile = logfile
        else:
            assert isinstance(logfile, str), "logfile name should be string"
            self.logfile = os.path.join(self._root_path, logfile)

        # Refresh the files & write always
        if self.logfile is not None:
            with open(self.logfile, 'w') as fo:
                print("", file=fo, end="")

        self.add_log(f"NARA performed on: {datetime.now()}\n\n", level=0)  # Always write
        self.add_log(">>> FA params & logging settings done. Let's evaluate & evolve!\n\n", level=2)

        params_info = f'''
    < Firefly algorithms parameters >
        * label  = {self.label}

        * alpha  = {self.alpha}
        * beta_0 = {self.beta_0}
        * gamma  = {self.gamma}

        * population_size        = {self.population_size}

        * verbosity = {self.verbosity} # 0(error) < 1(warning) < 2(info) < 3(debug) \n\n
            '''
        self.add_log(params_info, level=2)

        self.train_xyz_filename = train_xyz_filename
        self.valid_xyz_filename = valid_xyz_filename
        self.train_valid_ratio = train_valid_ratio
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.lcb_kappa = lcb_kappa

        self.gnn_filename = gnn_filename

        r_cut = gnn_config.get("r_cut", 5)
        self.r_cut = r_cut
        dtype = gnn_config.get("dtype", "float64")
        if device is None:
            device = torch.device("cpu")

        self.dtype = dtype
        self.device = device

        # 1-1) Initial 구조들이 이미 계산되어 있으면 -> 새로 랜덤구조 50개 재계산 필요 없음
        if (train_xyz_filename is not None) and os.path.exists(os.path.join(self._root_path, train_xyz_filename)):
            with open(train_xyz_filename, 'r') as fi:
                self.train_atoms = list(read_xyz(fi, index=slice(None)))
        else:
            self.train_atoms = []

        if (valid_xyz_filename is not None) and os.path.exists(os.path.join(self._root_path, valid_xyz_filename)):
            with open(valid_xyz_filename, 'r') as fi:
                self.valid_atoms = list(read_xyz(fi, index=slice(None)))
        else:
            self.valid_atoms = []

        # 1-2) Initial 구조들 없거나, GNN 없으면 > random 하게 initial generation 만들고, SCF 계산 수행
        # GNN 있는 경우에도 lr_fit 하기 위해선 target potential로 계산된 initial valid_set이 필요함
        N_strucs = len(self.train_atoms + self.valid_atoms)
        _xyz_loaded = N_strucs != 0

        if (not _xyz_loaded) or (not os.path.exists(self.gnn_filename)):
            all_dt = []
            for _ in range(self.n_init):
                new_agent = self.mother_agent.get_random_agent(label=f"{_+1}", label_generation="Gen_0")
                model = new_agent.atoms.copy()
                model.calc = self.target_calc
                model.pbc = True # TODO - check: Currently this is alwasy Ture for PAW
                if not os.path.exists("DFT"):
                    os.mkdir("DFT")
                os.chdir("DFT")

                if not os.path.exists(new_agent.label):
                    os.mkdir(new_agent.label)
                os.chdir(new_agent.label)

                self.add_log(f"Entering the SCF calc of {new_agent.label}", level=0) # TODO: level=3
                if os.path.exists("scf.traj"):
                    try:
                        _ = Trajectory("scf.traj", 'r')
                        model = list(_)[-1] # model is changed to already-calculated structure
                        E = model.get_potential_energy()
                        _.close()
                    except:
                        model = new_agent.atoms.copy()
                        model.calc = self.target_calc
                        model.pbc = True # TODO - check: Currently this is always True for PAW
                        E = model.get_potential_energy()
                        _traj = Trajectory("scf.traj", 'w')
                        _traj.write(model, energy=E)
                        _traj.close()
                else:
                    E = model.get_potential_energy()
                    _traj = Trajectory("scf.traj", 'w')
                    _traj.write(model, energy=E)
                    _traj.close()

                    # TODO: VASP calculator -> SinglePoint Calculator
                    _ = Trajectory("scf.traj", 'r')
                    model = list(_)[-1]  # model is changed to already-calculated structure
                    E = model.get_potential_energy()
                    _.close()

                all_dt.append(model)
                os.chdir(self._root_path)

            train_size = int(self.n_init * train_valid_ratio)
            valid_size = self.n_init - train_size
            _train_atoms, _valid_atoms = random_split(all_dt, [train_size, valid_size])
            self.train_atoms += _train_atoms
            self.valid_atoms += _valid_atoms

        self.add_log(f"Generating Dataset", level=0)  # TODO: level=3

        load_gnn = os.path.exists(self.gnn_filename)
        if not load_gnn:
            # with open(self.train_xyz_filename, 'r') as fi:
            #     _train_atoms_with_EF = list(read_xyz(fi, index=slice(None)))
            #
            # with open(self.valid_xyz_filename, 'r') as fi:
            #     _valid_atoms_with_EF = list(read_xyz(fi, index=slice(None)))
            self.add_log("GNN model not found. Start training!", level=2)

            main_UC(
                xyz_filename = self.train_atoms,
                valid_xyz = self.valid_atoms,
                device = self.device,
                batch_size = self.batch_size,
                valid_batch_size = self.valid_batch_size,
                max_epoch = 2000,
                nonLL_epoch = 10,
                patience = 50,
                ef_ratio = [1, 1],
                gnn_config = gnn_config,
                best_model_filename = self.gnn_filename,
                loss_filename = None)

            self.gnn_model = EquivariantGNN_UC.load(self.gnn_filename, device=self.device, dtype=self.dtype)
            self.add_log("GNN model training done", level=2)

        # 2-2) GNN 이미 있으면 GNN 으로부터 시작하죠
        else:
            self.add_log("GNN model found. Loading GNN...", level=2)
            try:
                self.gnn_model = EquivariantGNN_UC.load(self.gnn_filename, device=self.device, dtype=self.dtype)
            except Exception as e:
                self.add_log("Failed while loading GNN model", level=0)
                raise
            self.add_log("GNN model loaded successfully", level=2)

        self.gnn_calc = GNNUCWrapper(self.gnn_model, device=self.device, dtype=self.dtype)
        self.mother_agent.calc = self.gnn_calc

        self.lr_coef = fit_lr_valid(self.valid_atoms, self.gnn_model, device=self.device)
        self.lcb_tracker = LCB_tracker(gnn_calc = self.gnn_calc, lr_coef = self.lr_coef,
                                       kappa = self.lcb_kappa, optimizer = QuasiNewton)
        self.add_log(f"LCB_tracker set, with coef: {self.lr_coef}", level=2)

        self.waiting_for_target_calc = []
        self.waiting_for_retrain = []

        self.add_log(f"Check restart file", level=0)  # TODO: level=3

        self.restart = "ongoing.pickle"
        if restart is not None:
            self.restart = restart
            if isinstance(restart, str):  # filename
                if os.path.exists(restart):
                    self.add_log(f">>> Read initial generation & lowest minima from {restart} files!\n\n", level=2)
                    self.load_population(restart)  # Define self.current_generation & self.gbest_history
                else:
                    self.add_log(">>> Restart file not found. Starts from the scratch: Gen 0.\n\n", level=1)
                    self.current_gen = None
            else:
                self.add_log(">>> Restart file should be given in str type. Starts from the scratch.\n\n", level=1)
                self.current_gen = None
        else:
            self.current_gen = None

        if self.current_gen is None:
            self.add_log(f"Generation not loaded, random agents generated", level=0)  # TODO: level=3
            self.n_gen = 1
            self.gbest_history = []

            initial_gen_label = "Gen_1" if load_gnn else "Gen_0"
            initial_agents = []
            for i in range(self.population_size):
                initial_agents.append(self.mother_agent.get_random_agent(
                    label = f"{i+1}",
                    label_generation = initial_gen_label
                ))

            if not load_gnn:
                # naive geo-opt, since GNN0 is not robust enough
                self.add_log("Enter generating force_init", level=0) # TODO: level=3
                self.lcb_tracker.update_atoms([agent.atoms for agent in initial_agents], fmax = 1e-1)
                descs = np.array([np.sum(atoms.arrays["node_features_inv"], axis=0) for atoms in self.lcb_tracker.LCBs_atoms_GNN])
                kmeans = KMeans(n_clusters=self.n_force_init, n_init="auto", max_iter=300, tol=1e-4, random_state=None)
                labels = kmeans.fit_predict(descs)

                for i in range(self.n_force_init):
                    mask = (labels==i)
                    model = self.lcb_tracker.LCBs_atoms_GNN[np.argmin(self.lcb_tracker.lcbs[mask])].copy()
                    model.calc = self.target_calc
                    new_agent = self.mother_agent.get_random_agent(
                        label = f"add_geo_opt_{i+1}",
                        label_generation = f"Gen_0"
                    )
                    new_agent.x = model
                    self.waiting_for_target_calc.append(new_agent)

                self.add_log(f"Retrain requested", level=0) # TODO: level=3
                self.re_train(geo_opt=True, max_steps=None) # self.train_atoms, self.valid_atoms, lcb update 까지
                # Full local opt here

                initial_agents = []
                for i in range(self.population_size):
                    initial_agents.append(self.mother_agent.get_random_agent(
                        label = f"{i+1}",
                        label_generation = "Gen_1"
                    ))
            else:
                self.add_log("Skipping force_init, since GNN is loaded", level=2)
                if not _xyz_loaded:
                    # If not loaded, we need to load the geo-opt data to training & validation dataset
                    _added_traj = []
                    for i in range(self.n_force_init):
                        traj_name = os.path.join(self._root_path, "DFT", f"Gen_0_Agent_add_geo_opt_{i+1}", "opt.traj")
                        if os.path.exists(traj_name):
                            _traj = Trajectory(traj_name, 'r')
                            _added_traj += list(_traj)
                            _traj.close()
                        else:
                            self.add_log(f"Gen_0_Agent_add_geo_opt_{i+1}/opt.traj not found. skipping loading", level=2)

                    N = len(_added_traj)
                    if N!=0:
                        train_size = int(self.train_valid_ratio * N)
                        valid_size = N - train_size
                        _t, _v = random_split(_added_traj, [train_size, valid_size])
                        self.train_atoms += _t
                        self.valid_atoms += _v
                    else:
                        self.add_log("None of opt.traj from force_init loaded.", level=2)

                    safe_write_xyz(self.train_xyz_filename, self.train_atoms)
                    safe_write_xyz(self.valid_xyz_filename, self.valid_atoms)
                    # Do we need retrain here? Maybe not - already applied

                lowest_indexes = np.argsort([atoms.get_potential_energy() for atoms in self.train_atoms])[:self.population_size]
                self.lcb_tracker.update_atoms([self.train_atoms[i] for i in lowest_indexes], fmax=self.mother_agent.fmax)

            self.add_log("Initiating Population", level=0) # TODO: level=3
            self.current_gen = Population(agents = initial_agents,
                                          label = f"{self.n_gen}",
                                          maximize_fitness = self.maximize_fitness,
                                          basedir = self._root_path)

        # else:
        #     self.lcb_tracker.update_atoms(self.valid_atoms, fmax = self.mother_agent.fmax)
        self.frozen_agent_index = []
        self.add_log("Initiate Done.", level=2)

    def re_train(self, train_only=False, geo_opt=False, max_steps=None):
        device, dtype = self.device, self.dtype
        if not train_only:
            self.add_log(">>> Start calculation using target potential", level=2)
            self.add_log(f"Total {len(self.waiting_for_target_calc)} structures will be calculated", level=2)
            for agent in self.waiting_for_target_calc:
                model = agent.atoms.copy()
                model.info = {}
                model.arrays = {"numbers":model.arrays["numbers"],
                                "positions":model.arrays["positions"],
                                "tags":model.arrays["tags"],
                                }
                model.pbc = True # TODO-flexibility? -- This is for PAW calculations
                # For PAW. For LCAO or others which enables non-periodic DFT, please ensures that the results is same
                model.calc = self.target_calc
                os.chdir(self._root_path)
                if not os.path.exists("DFT"):
                    os.mkdir("DFT")
                os.chdir("DFT")

                if not os.path.exists(agent.label):
                    os.mkdir(agent.label)
                os.chdir(agent.label)

                if geo_opt:
                    # 이미 있으면 skip
                    if not os.path.exists("opt.traj"):
                        dyn = FIRE(model, logfile=None, trajectory="opt.traj")
                        # 아니 VASP이랑 QuasiNewton이랑 같이 쓰면 계산량 너무 심해짐 왜 그럴까?
                        # MLP에서는 QuasiNewton이랑 같이 쓸 때 젤 빠르던데.. 비싼 DFT랑은 FIRE가 더 잘 어울리나?
                        self.add_log(f"|__ {agent.label} on calculation using target calculator", level=2)
                        if max_steps is not None:
                            dyn.run(fmax=self.mother_agent.fmax, steps = max_steps)
                        else:
                            dyn.run(fmax=self.mother_agent.fmax)
                    else:
                        self.add_log(f"|__ Found opt.traj for {agent.label}. Just use the opt.traj", level=2)
                    _traj = Trajectory("opt.traj", 'r')
                    self.waiting_for_retrain += list(_traj)
                    _traj.close()
                else:
                    self.add_log(f"Geo-opt tag is turned off. SCF calculations will be performed", level=2)
                    model.get_potential_energy()
                    self.waiting_for_retrain.append(model)
                os.chdir(self._root_path)
            self.waiting_for_target_calc = [] # empty the list

        self.add_log(">>> Re-training GNN started", level=2)
        all_dt = self.waiting_for_retrain

        N = len(all_dt)
        train_size = int(N*self.train_valid_ratio)
        valid_size = N-train_size
        _t, _v = random_split(all_dt, [train_size, valid_size])
        self.train_atoms += _t
        self.valid_atoms += _v

        # for atoms in self.train_atoms:
        #     atoms.info = {}
        #     atoms.arrays = {"numbers": atoms.arrays["numbers"], "positions": atoms.arrays["positions"]}
        #
        # for atoms in self.valid_atoms:
        #     atoms.info = {}
        #     atoms.arrays = {"numbers": atoms.arrays["numbers"], "positions": atoms.arrays["positions"]}

        safe_write_xyz(self.train_xyz_filename, self.train_atoms)
        safe_write_xyz(self.valid_xyz_filename, self.valid_atoms)

        main_UC(
            xyz_filename = self.train_atoms,
            valid_xyz = self.valid_atoms,
            device = self.device,
            batch_size = self.batch_size,
            valid_batch_size = self.valid_batch_size,
            max_epoch = 2000,
            nonLL_epoch = 10,
            patience = 50,
            ef_ratio = [1, 1],
            gnn_config = self.gnn_model.gnn_config,
            best_model_filename = self.gnn_filename,
            loss_filename = None)

        self.gnn_model = EquivariantGNN_UC.load(self.gnn_filename, device=self.device, dtype=self.dtype)

        self.gnn_calc = GNNUCWrapper(self.gnn_model, device=device, dtype=dtype)
        self.waiting_for_retrain = []
        self.frozen_agent_index = []

        lr_coef = fit_lr_valid(self.valid_atoms, gnn_model=self.gnn_model, device=device)
        self.lcb_tracker.update(self.gnn_calc, lr_coef, atoms_list = self.valid_atoms, fmax = self.mother_agent.fmax)
        self.lr_coef = lr_coef
        self.add_log(f"LCB_tracker updated, with coef: {self.lr_coef}", level=2)

        safe_write_xyz("lcb_atoms.xyz", self.lcb_tracker.LCBs_atoms_GNN)

        self.mother_agent.calc = self.gnn_calc

        if self.current_gen is not None:
            for agent in self.current_gen:
                agent.calc = self.gnn_calc
                agent._fitness = None

        for agent in self.gbest_history:
            agent.calc = self.gnn_calc
            agent._fitness = None
            agent.evaluate()

    def autorun_gbest_only(self,
                           max_n_gen = None,
                           patience:int = 100,
                           n_calc = 3,
                           n_calc_gen = 10,
                           ):

        device, dtype = self.device, self.dtype
        if max_n_gen is None:
            max_n_gen = np.inf
        gen_no_improve = 0
        gen_no_update_with_lcb_found = 0
        while True:
            self.add_log(f"Gen [{gen_no_improve}/{patience}] | from total [{self.n_gen}/{max_n_gen}]", level=2)
            if self.n_gen > max_n_gen:
                self.add_log("Reached max_n_gen without full convergence.", level=1)
                return False # escape the autorun. Finished properly, but not converged within max_n_gen
            _gbest_renewed = self.evaluate()

            _lcbs = []
            for i, agent in enumerate(self.current_gen.agents):
                if i in self.frozen_agent_index:
                    # We should not add already frozen index
                    lcb_found, lcb = self.lcb_tracker.check(agent.atoms, return_lcb=True)
                    self.add_log(f"*FROZEN-{agent.label}: LCB={lcb}", level=2)
                    continue
                lcb_found, lcb = self.lcb_tracker.check(agent.atoms, return_lcb = True)
                self.add_log(f"{agent.label}: LCB={lcb}, New_LCB {lcb_found}", level=2)
                if lcb_found:
                    self.frozen_agent_index.append(i)
                    _lcbs.append([i, lcb])

            if len(_lcbs) > n_calc:
                self.add_log(f"LCBs found in Gen {self.n_gen} exceeds N_calc", level=1)
                lowest_lcbs = np.argsort([_[1] for _ in _lcbs])[:n_calc]
                for _ in lowest_lcbs:
                    i = _lcbs[_][0]
                    self.waiting_for_target_calc.append(self.current_gen[i])
            else:
                for _ in _lcbs:
                    i = _[0]
                    self.waiting_for_target_calc.append(self.current_gen[i])

            # if self.lcb_tracker.check(agent.atoms):
            #     self.lcb_tracker.add(agent.atoms)
            #     self.waiting_for_target_calc.append(agent)
            #     self.frozen_agent_index.append(i)

            if (len(self.frozen_agent_index) >= n_calc) or gen_no_update_with_lcb_found >= n_calc_gen:
                if len(self.frozen_agent_index) < n_calc:
                    self.add_log(f"Number of frozen agents are less than n_calc:{n_calc}, "
                                 f"but generation without re-training exceeds n_calc_gen:{n_calc_gen}", level=2)
                self.re_train(train_only=False, geo_opt=True, max_steps=5)
                gen_no_update_with_lcb_found = 0
            elif len(self.frozen_agent_index) != 0:
                gen_no_update_with_lcb_found += 1
                # there is something to calculate!!!
            else:
                gen_no_update_with_lcb_found = 0

            _gbest_renewed = self.evaluate()

            gen_no_improve = 0 if _gbest_renewed else (gen_no_improve + 1)
            if gen_no_improve >= patience:
                if len(self.gbest_history)==1:
                    self.add_log("FA couldn't find any other gbest", level=0)
                    return False # finished properly = False -> This means... FA couldn't find any other g_bests
                else:
                    self.add_log(f"End FA in {self.n_gen} generations.", level=2)
                    return True # escape the loop when reached the patience
            self.save_population(self.restart)
            if self.n_gen != max_n_gen:
                self.evolve(frozen_index = self.frozen_agent_index) # self.n_gen += 1 in here


    def autorun_gbest_only_parallel(self,
                                    max_n_gen = None,
                                    patience:int = 100,
                                    call_calculation = 5):
        device, dtype = self.device, self.dtype
        if max_n_gen is None:
            max_n_gen = np.inf
        gen_no_improve = 0
        retrain_tc = os.path.join(self._root_path, "_waiting_for_target_calc")
        retrain_tc = os.path.join(self._root_path, "_waiting_for_retrain")
        if not os.path.exists(retrain_tc):
            os.mkdir(retrain_tc)
        if not os.path.exists(retrain_tced):
            os.mkdir(retrain_tced)

        while True:
            self.add_log(f"Gen [{gen_no_improve}/{patience}] | from total [{self.n_gen}/{max_n_gen}]", level=2)
            if self.n_gen > max_n_gen:
                self.add_log("Reached max_n_gen without full convergence.", level=1)
                return False # escape the autorun. Finished properly, but not converged within max_n_gen
            _gbest_renewed = self.evaluate()
            for i, agent in enumerate(self.current_gen.agents):
                if self.lcb_tracker.check(agent.atoms):
                    self.lcb_tracker.add(agent.atoms)
                    self.waiting_for_target_calc.append(agent)
                    for agent in self.waiting_for_target_calc:
                        fnn = os.path.join(fn, f"{agent.label:s}.vasp")
                        if not os.path.exists(os.path.join(fn, fnn)):
                            model = agent.atoms.copy()
                            # model.calc = None
                            # model.info = {}
                            # model.arrays
                            write_vasp(fnn, agent.atoms, direct=True, sort=False) # sort=False해서 잘 하자!

                    self.frozen_agent_index.append(i)
            if len(self.frozen_agent_index)>call_calculation:
                self.re_train(train_only=False, geo_opt=True, max_steps=5)

            _gbest_renewed = self.evaluate()

            gen_no_improve = 0 if _gbest_renewed else (gen_no_improve + 1)
            if gen_no_improve >= patience:
                if len(self.gbest_history)==1:
                    self.add_log("FA couldn't find any other gbest", level=0)
                    return False # finished properly = False -> This means... FA couldn't find any other g_bests
                else:
                    self.add_log(f"End FA in {self.n_gen} generations.", level=2)
                    return True # escape the loop when reached the patience
            self.save_population(self.restart)
            if self.n_gen != max_n_gen:
                self.evolve(frozen_index = self.frozen_agent_index) # self.n_gen += 1 in here