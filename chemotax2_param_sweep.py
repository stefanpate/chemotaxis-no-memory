import numpy as np
from helpers import chemotax2


h = 0.7
x_init = 0
gs = [0.005, 0.01, 0.015, 0.02, 0.025]
p0s = [0.1, 0.15, 0.2]
n_runs = int(1e4)
tmax = int(1e3)
finite_domain = False
apply_grad = True
xmax = None
seed = 1234
data_dir = '/home/stef/MarkoRotation/data/'

for g in gs:
    print(g)
    for p0 in p0s:
        print(p0)
        runs = chemotax2(p0=p0, h=h, g=g, n_runs=n_runs, tmax=tmax, finite_domain=finite_domain, apply_grad=apply_grad, x_init=x_init, seed=seed)
        fn = f"chemotax2_gradient_{apply_grad}_finite_{finite_domain}_tmax_{tmax}_n_runs_{n_runs}_xmax_{xmax}_p0_{p0}_h_{h}_g_{g}_x_init_{x_init}_seed_{seed}.csv"
        np.savetxt(data_dir + fn, runs, delimiter=',')