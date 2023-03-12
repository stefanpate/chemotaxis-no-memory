import numpy as np
from helpers import chemotax3_lt

# Settings
p0 = 0.15
h = 0.7
g = 0.015
x_init = 10
# xbs = np.arange(150, 220, 10)
xbs = [None]
apply_grad = True
finite_domain = False
total_runs = 1e4
grp_size = 2500
n_grps = int(total_runs / grp_size)
data_dir = '/home/stef/MarkoRotation/data/'

for xb in xbs:
    print(xb)
    # tf = (2 * xb)**2 + 1000
    tf = 50000
    # Run in chunks of grp_size, up to total_runs
    for i in range(n_grps):
        print(i)
        # Seed with group number
        mean = chemotax3_lt(p0=p0, h=h, g=g, n_runs=grp_size, tf=tf, x_init=x_init, apply_grad=apply_grad, finite_domain=finite_domain, xb=xb, seed=i)
        fn = f"chemotax_mean_pos_gradient_{apply_grad}_finite_{finite_domain}_tf_{tf}_n_runs_{grp_size}_xb_{xb}_p0_{p0}_h_{h}_g_{g}_x_init_{x_init}_seed_{i}.csv"
        np.savetxt(data_dir + fn, mean, delimiter=',') # Save group mean, sem
