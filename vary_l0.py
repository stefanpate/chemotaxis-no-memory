import numpy as np
from helpers import chemotax, chemotax_squish

l0s = [5, 6, 7, 8, 13, 14, 15]
g = 1e-2
n_runs = 10000
tmax = 1000
finite_domain = False
apply_grad = True
x_init = 0
xmax = None
seeds = [1234]
data_dir = '/home/stef/MarkoRotation/data/'
do_squish = False

for l0 in l0s:
    print(l0)
    for seed in seeds:

        if do_squish:
            runs = chemotax_squish(l0=l0, gain=g, n_runs=n_runs, tmax=tmax, finite_domain=finite_domain, apply_grad=apply_grad, x_init=x_init, seed=seed)
            fn = f"chemotax_boundary_squish_gradient_{apply_grad}_finite_{finite_domain}_tmax_{tmax}_n_runs_{n_runs}_x_init_{x_init}_xmax_{xmax}_l0_{l0}_seed_{seed}_shift_9_gain_{g}.csv"
        else:
            runs = chemotax(l0=l0, gain=g, n_runs=n_runs, tmax=tmax, finite_domain=finite_domain, apply_grad=apply_grad, x_init=x_init, seed=seed)
            fn = f"chemotax_gradient_{apply_grad}_finite_{finite_domain}_tmax_{tmax}_n_runs_{n_runs}_x_init_{x_init}_xmax_{xmax}_l0_{l0}_seed_{seed}_shift_9_gain_{g}.csv"
        
        np.savetxt(data_dir + fn, runs, delimiter=',')
