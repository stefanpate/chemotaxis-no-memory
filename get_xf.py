import numpy as np


xmaxes = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 170]
x_init = 0
apply_grad = True
finite_domain = True
data_dir = '/home/stef/MarkoRotation/data/'
ave_start = -5000
grp_size = 2500
n_grps = 4

xfs = []
for xmax in xmaxes:
    print(xmax)
    if xmax <= 100:
        n_runs = 5000
        tmax = 10000
        seeds = [1234, 9876]
    else:
        n_runs = 2500
        tmax = 20000
        seeds = [1212, 1218, 1234, 9876]
    this_bound = []
    for seed in seeds:
        fn = f"chemotax_boundary_squish_gradient_{apply_grad}_finite_{finite_domain}_tmax_{tmax}_n_runs_{n_runs}_x_init_{x_init}_xmax_{xmax}_l0_10_seed_{seed}_shift_9_gain_0.01.csv"
        this_run = np.loadtxt(data_dir + fn, delimiter=',')
        for i in range(int(n_runs / grp_size)):
            group_mean = this_run[:,ave_start:].mean() # Average over time for every run
            this_bound.append(group_mean)
    
    this_bound = np.array(this_bound)
    mean, sem = this_bound.mean(), this_bound.std() / np.sqrt(n_grps)
            
    xfs.append([xmax, mean, sem]) # Append for each condition

np.savetxt(data_dir + 'final_position_squishy_boundary.csv', xfs, delimiter=',')