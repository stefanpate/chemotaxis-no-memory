import numpy as np

p0 = 0.15
h = 0.7
g = 0.015
x_init = 10
xbs = np.arange(50, 110, 10)
apply_grad = True
finite_domain = True
total_runs = 1e4
grp_size = 2500
n_grps = int(total_runs / grp_size)
data_dir = '/home/stef/MarkoRotation/data/'

xb = 100
tf = (2 * xb)**2 + 1000

this_cond = np.zeros(shape=(tf))
for i in range(n_grps):
    fn = f"chemotax2_ave_sem_pos_gradient_{apply_grad}_finite_{finite_domain}_tf_{tf}_n_runs_{grp_size}_xb_{xb}_p0_{p0}_h_{h}_g_{g}_x_init_{x_init}_seed_{i}.csv"
    this_grp = np.loadtxt(data_dir + fn, delimiter=',')
    this_cond.append(this_grp[0,:])

this_cond = np.vstack(this_cond)
this_cond_mean = this_cond.mean(axis=0)
this_cond_sem = this_cond.std(axis=0) / np.sqrt(n_grps)
