import numpy as np
from helpers import chemotax3

# Settings
p0 = 0.15
h = 0.7
g = 0.015
x_init = 10
xbs = np.arange(30, 110, 10)
apply_grad = True
finite_domain = True
total_runs = 1e4
grp_size = 2500
n_grps = int(total_runs / grp_size)
data_dir = '/home/stef/MarkoRotation/data/'

for xb in xbs:
    print(xb)
    tf = (2 * xb)**2 + 1000
    # Run in chunks of grp_size, up to total_runs
    for i in range(n_grps):
        print(i)
        # Seed with group number
        grp_runs = chemotax3(p0=p0, h=h, g=g, n_runs=grp_size, tf=tf, x_init=x_init, apply_grad=apply_grad, finite_domain=finite_domain, xb=xb, seed=i)
        mean = grp_runs.mean(axis=0).reshape(1,-1)
        fn = f"chemotax_mean_pos_gradient_{apply_grad}_finite_{finite_domain}_tf_{tf}_n_runs_{grp_size}_xb_{xb}_p0_{p0}_h_{h}_g_{g}_x_init_{x_init}_seed_{i}.csv"
        np.savetxt(data_dir + fn, mean, delimiter=',') # Save group mean, sem
