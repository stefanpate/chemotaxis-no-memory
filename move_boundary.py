import numpy as np
from helpers import chemotax, chemotax_squish

xmaxes = [20, 30, 40, 50, 60, 70, 80, 90, 100]

n_runs = 5000
tmax = 10000
finite_domain = True
apply_grad = True
x_init = 0
seeds = [9876]
data_dir = '/home/stef/MarkoRotation/data/'
do_squish = True

for xmax in xmaxes:
    print(xmax)
    for seed in seeds:

        if do_squish:
            runs = chemotax_squish(n_runs=n_runs, tmax=tmax, finite_domain=finite_domain, apply_grad=apply_grad, xmax=xmax, x_init=x_init, seed=seed)
            fn = f"chemotax_boundary_squish_gradient_{apply_grad}_finite_{finite_domain}_tmax_{tmax}_n_runs_{n_runs}_x_init_{x_init}_xmax_{xmax}_l0_10_seed_{seed}_shift_9_gain_0.01.csv"
        else:
            runs = chemotax(n_runs=n_runs, tmax=tmax, finite_domain=finite_domain, apply_grad=apply_grad, xmax=xmax, x_init=x_init, seed=seed)
            fn = f"chemotax_gradient_{apply_grad}_finite_{finite_domain}_tmax_{tmax}_n_runs_{n_runs}_x_init_{x_init}_xmax_{xmax}_l0_10_seed_{seed}_shift_9_gain_0.01.csv"
        
        np.savetxt(data_dir + fn, runs, delimiter=',')
