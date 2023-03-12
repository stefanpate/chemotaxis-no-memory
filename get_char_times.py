import numpy as np

def get_tc(runs, grp_size=2500, beta=np.e):
    n = int(runs.shape[0] / grp_size)
    tcs = []
    for i in range(n):
        run_grp = runs[i * grp_size:(i+1) * grp_size]
        grp_mean = run_grp.mean(axis=0)
        argmax_pos = np.argmax(grp_mean)
        max_pos = np.max(grp_mean)
        this_tc = np.where(grp_mean[argmax_pos:] <= max_pos / beta)[0][1]
        tcs.append(this_tc)
    
    tcs = np.array(tcs)
    mean, sem = tcs.mean(), tcs.std() / np.sqrt(n)
    return mean, sem

xmaxes = [150, 160, 170, 180, 190, 200]
beta = np.e # e-fold, 2-fold, whatever
n_pts = len(xmaxes)
n_runs = 2500
tmax = 20000
finite_domain = True
apply_grad = True
x_init = 0
seeds = [1234, 9876, 1212, 1218]
data_dir = '/home/stef/MarkoRotation/data/'
save_fn = f"tc_beyond_saturation_beta_e_gradient_{apply_grad}_finite_{finite_domain}_tmax_{tmax}_n_runs_10000_x_init_{x_init}_l0_10_shift_9_gain_0.01.csv"

tc_mean_sem = []
for i, xmax in enumerate(xmaxes):
    print(xmax)

    # Load in chemotax data
    runs = []
    for seed in seeds:
        load_fn = f"chemotax_gradient_{apply_grad}_finite_{finite_domain}_tmax_{tmax}_n_runs_{n_runs}_x_init_{x_init}_xmax_{xmax}_l0_10_seed_{seed}_shift_9_gain_0.01.csv"
        this_runs = np.loadtxt(data_dir + load_fn, delimiter=',')
        runs.append(this_runs)
    
    runs = np.vstack(runs) # All trials for this boundary pos
    mean, sem = get_tc(runs, beta=beta)
    tc_mean_sem.append([xmax, mean, sem]) # Get mean, sem of characteristic time

tc_mean_sem = np.array(tc_mean_sem)
np.savetxt(data_dir + save_fn, tc_mean_sem, delimiter=',')