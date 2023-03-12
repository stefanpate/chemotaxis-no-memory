import numpy as np


# Load in chemotax data
# xmaxes = [None]
xmaxes = [150, 200, 250]
n_runs = 2500
tmax = 20000
finite_domain = True
apply_grad = True
x_init = 0
seeds = [1234, 9876, 1212, 1218]
n_seeds = len(seeds)
data_dir = '/home/stef/MarkoRotation/data/'
do_squish = False # Use squishy boundary data


for i, xmax in enumerate(xmaxes):
    sum, var = np.zeros(shape=(tmax,)), np.zeros(shape=(tmax,)) # Start w 0 mean, variance for each condition
    print(xmax)
    for j, seed in enumerate(seeds):
        load_fn = f"chemotax_gradient_{apply_grad}_finite_{finite_domain}_tmax_{tmax}_n_runs_{n_runs}_x_init_{x_init}_xmax_{xmax}_l0_10_seed_{seed}_shift_9_gain_0.01.csv"
        if do_squish:
            temp = load_fn.split('_')
            load_fn = temp[0] + '_boundary_squish_' + '_'.join(temp[1:])
        
        this_runs = np.loadtxt(data_dir + load_fn, delimiter=',')
        this_sum = this_runs.sum(axis=0)
        this_var = np.var(this_runs, axis=0)
        sum += this_sum
        var += this_var

    save_fn = f"ave_sem_pos_gradient_{apply_grad}_finite_{finite_domain}_tmax_{tmax}_n_runs_{n_runs * n_seeds}_x_init_{x_init}_xmax_{xmax}_l0_10_shift_9_gain_0.01.csv"
    if do_squish:
        temp = save_fn.split('_')
        save_fn = '_'.join(temp[:3]) + '_boundary_squish_' + '_'.join(temp[3:])
    
    ave_pos= sum / (n_seeds * n_runs)
    sem_pos = np.sqrt(var / (n_seeds * n_runs))
    stats_pos = np.vstack([ave_pos, sem_pos])
    np.savetxt(data_dir + save_fn, stats_pos, delimiter=',')