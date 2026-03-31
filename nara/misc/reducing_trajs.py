from ase.io.trajectory import Trajectory
import glob, os
from tqdm import tqdm
import pickle

total_N_atoms = len(glob.glob("Gen_*.traj"))

all_gens = []
n_gen = 0
processed = 0

pbar = tqdm(total=total_N_atoms, desc="Reading traj files", unit="file")

while True:
	n_gen += 1
	this_gen = []
	all_agent = glob.glob(f"Gen_{n_gen:d}_Agent_*.traj")
	if len(all_agent) == 0:
		break

	for i in range(len(all_agent)):
		this_agent_fn = f"Gen_{n_gen:d}_Agent_{i+1:d}.traj"
		_traj = Trajectory(this_agent_fn, 'r')
		for this_atoms in _traj:
			pass
		_traj.close()
		this_gen.append(this_atoms)
	all_gens.append(this_gen)

	_n = len(this_gen)
	processed += _n
	pbar.update(_n)
	pbar.set_postfix(gen=n_gen, agents=_n, processed=processed)
pbar.close()

with open("all_gens.pickle", 'wb') as fo:
	pickle.dump(all_gens, fo)