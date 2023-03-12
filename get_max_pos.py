import numpy as np

def get_max_pos(runs, grp_size=2500):
    n = int(runs.shape[0] / grp_size)
    max_poses = []
    for i in range(n):
        run_grp = runs[i * grp_size:(i+1) * grp_size]
        grp_mean = run_grp.mean(axis=0)
        max_pos = np.max(grp_mean)
        max_poses.append(max_pos)
    
    max_poses = np.array(max_poses)
    mean, sem = max_poses.mean(), max_poses.std() / np.sqrt(n)
    return mean, sem

xmaxes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
n_pts = len(xmaxes)
n_runs = 5000
tmax = 10000
finite_domain = True
apply_grad = True
x_init = 0
seeds = [1234, 9876]
data_dir = '/home/stef/MarkoRotation/data/'
save_fn = f"max_pos_gradient_{apply_grad}_finite_{finite_domain}_tmax_{tmax}_n_runs_10000_x_init_{x_init}_l0_10_shift_9_gain_0.01.csv"

max_pos_mean_sem = []
for i, xmax in enumerate(xmaxes):
    print(xmax)

    # Load in chemotax data
    runs = []
    for seed in seeds:
        load_fn = f"chemotax_gradient_{apply_grad}_finite_{finite_domain}_tmax_{tmax}_n_runs_{n_runs}_x_init_{x_init}_xmax_{xmax}_l0_10_seed_{seed}_shift_9_gain_0.01.csv"
        this_runs = np.loadtxt(data_dir + load_fn, delimiter=',')
        runs.append(this_runs)
    
    runs = np.vstack(runs) # All trials for this boundary pos
    mean, sem = get_max_pos(runs)
    max_pos_mean_sem.append([xmax, mean, sem]) # Get mean, sem of characteristic time

max_pos_mean_sem = np.array(max_pos_mean_sem)
np.savetxt(data_dir + save_fn, max_pos_mean_sem, delimiter=',')