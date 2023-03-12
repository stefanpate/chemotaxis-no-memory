import numpy as np

'''
Split runs for some condition into
groups, get mean & sem position for each
group, save
'''

def group_mean_sem(runs, grp_size):
    n_grps = int(runs.shape[0] / grp_size)
    grouped_runs = []
    for i in range(n_grps):
        mean = runs[i * grp_size : (i + 1) * grp_size].mean(axis=0)
        sem = runs[i * grp_size : (i + 1) * grp_size].std(axis=0) / np.sqrt(n_grps)
        grouped_runs.append(mean)
        grouped_runs.append(sem)

    grouped_runs = np.array(grouped_runs)
    return grouped_runs

n_runs = 5000
tmax = 10000
finite_domain = True
apply_grad = True
x_init = 0
seeds = [1234, 9876]
grp_size = 2500
data_dir = '/home/stef/MarkoRotation/data/'

# Conditions
xmaxes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

for i, xb in enumerate(xmaxes):
    this_cond = []
    print(xb)

    # Load in data for this condition
    runs = []
    for seed in seeds:
        load_fn = f"chemotax_gradient_{apply_grad}_finite_{finite_domain}_tmax_{tmax}_n_runs_{n_runs}_x_init_{x_init}_xmax_{xb}_l0_10_seed_{seed}_shift_9_gain_0.01.csv"
        this_runs = np.loadtxt(data_dir + load_fn, delimiter=',')
        runs.append(this_runs)

    runs = np.vstack(runs) # All trials for this boundary pos
    grouped_runs = group_mean_sem(runs, grp_size)
    save_fn = f"grouped_mean_sem_pos_gradient_{apply_grad}_finite_{finite_domain}_tmax_{tmax}_n_runs_{n_runs * len(seeds)}_x_init_{x_init}_xmax_{xb}_l0_10_shift_9_gain_0.01.csv"
    np.savetxt(data_dir + save_fn, grouped_runs, delimiter=',')

