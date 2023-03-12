import numpy as np

def chemotax(l0=10, n_runs=10000, tmax=10000, x_init=0, apply_grad=False, seed=1234, shift=9, gain=0.01, finite_domain=False, xmax=250):
    '''
    Simulates chemotaxis with or without gradient on a finite domain
    or on an infinite one.
    '''

    rng = np.random.default_rng(seed)

    # Simulate
    runs = []
    for i in range(n_runs):
        this_run = []
        pos = x_init # Initialize position
        direct = rng.choice([-1, 1]) # 50/50 random init direction
        for t in range(tmax):
            this_run.append(pos) # Append position
            pos += direct # Update position
            
            # Select either static pflip or fcn of position
            if apply_grad:
                l = l0 + shift * np.tanh(gain * pos) # Update average run len
                pflip = 1 / l # Flip prob is fcn of position
            else:
                pflip = 1 / l0
            
            # Update direction
            if rng.uniform() < pflip:
                direct *= -1

            # If enforcing finite domain, check if at boundary
            if (finite_domain) & (pos >= xmax):
                direct = -1
            elif (finite_domain) & (pos <= -xmax):
                direct = 1
        
        runs.append(this_run)
    
    return np.array(runs)

def diff_tanh(x, shift=9, gain=0.01):
    return shift * gain * (1 / np.cosh(gain * x))

def lobf(x, m, b):
    return m * x + b

def get_wall_hit_cdf(runs, xmax):
    n_runs = runs.shape[0]
    rows, cols = np.where(runs == xmax)
    sorted_wall_hits = np.array(sorted(list(zip(rows, cols))))
    hit_series = np.zeros(shape=(runs.shape[1],))
    for i in range(n_runs):
        run_mask = sorted_wall_hits[:,0] == i
        row_hits = sorted_wall_hits[:,1][run_mask]
        if len(row_hits) > 0:
            first_hit = row_hits.min()
            hit_series[first_hit] += 1

    return np.cumsum(hit_series) / n_runs

def chemotax_squish(l0=10, n_runs=10000, tmax=10000, x_init=0, apply_grad=False, seed=1234, shift=9, gain=0.01, finite_domain=False, xmax=250):
    '''
    Simulates chemotaxis with or without gradient on a finite domain
    or on an infinite one. Boundary is "squishy". If the agent hits it, 
    it doesn't go anywhere, it doesn't turn around automatically like in
    the reflecting boundary case.
    '''

    rng = np.random.default_rng(seed)

    # Simulate
    runs = []
    for i in range(n_runs):
        squish = False # Whether motion would go past boundary
        this_run = []
        pos = x_init # Initialize position
        direct = rng.choice([-1, 1]) # 50/50 random init direction
        for t in range(tmax):
            this_run.append(pos) # Append position
            if not squish:
                pos += direct # Update position unless at the boundary
            
            # Select either static pflip or fcn of position
            if apply_grad:
                l = l0 + shift * np.tanh(gain * pos) # Update average run len
                pflip = 1 / l # Flip prob is fcn of position
            else:
                pflip = 1 / l0
            
            # Update direction
            if rng.uniform() < pflip:
                direct *= -1

            # If enforcing finite domain, check if at boundary
            if (finite_domain) & (pos >= xmax) & (direct == 1):
                squish = True
            elif (finite_domain) & (pos <= -xmax) & (direct == -1):
                squish = True
            else:
                squish = False
        
        runs.append(this_run)
    
    return np.array(runs)

def conc2pos(c,  g=0.01, l0=10, s=9):
    '''
    Inverse of concentration function
    '''
    return (np.log(s + c - l0) - np.log(s - c + l0)) / (2 * g)


def chemotax2(p0=1e-1, h=1e-1, g=1e-2, n_runs=10000, tf=10000, x_init=0, apply_grad=False, finite_domain=False, xb=250, seed=1234):
    '''
    Simulates chemotaxis with or without gradient on a finite domain
    or on an infinite one. Emphasizes probability of flipping, compared
    to chemotax's run length.
    '''

    rng = np.random.default_rng(seed)

    # Simulate
    runs = []
    for i in range(n_runs):
        this_run = []
        pos = x_init # Initialize position
        direct = rng.choice([-1, 1]) # 50/50 random init direction
        for t in range(tf):
            this_run.append(pos) # Append position
            pos += direct # Update position
            
            # Select either static pflip or fcn of position
            if apply_grad:
                pflip = p0 * (1 - h * np.tanh(g * pos))  # Update pflip
            else:
                pflip = p0
            
            # Update direction
            if rng.uniform() < pflip:
                direct *= -1

            # If enforcing finite domain, check if at boundary
            if (finite_domain) & (pos >= xb):
                direct = -1
            elif (finite_domain) & (pos <= -xb):
                direct = 1
        
        runs.append(this_run)
    
    return np.array(runs)

def chemotax3(p0=1e-1, h=1e-1, g=1e-2, n_runs=10000, tf=10000, x_init=0, apply_grad=False, finite_domain=False, xb=250, seed=1234):
    '''
    Simulates chemotaxis with or without gradient on a finite domain
    or on an infinite one. Emphasizes probability of flipping, compared
    to chemotax's run length.
    '''

    rng = np.random.default_rng(seed)

    # Simulate
    runs = []
    pos = np.zeros(shape=(n_runs,)) + x_init # Initialize position
    direct = rng.choice([-1, 1], size=n_runs) # 50/50 random init direction
    for t in range(tf):
        runs.append(pos) # Append position
        pos = pos + direct # Update position
        
        # Select either static pflip or fcn of position
        if apply_grad:
            pflip = p0 * (1 - h * np.tanh(g * pos))  # Update pflip
        else:
            pflip = p0 * np.ones(shape=(n_runs,))
        
        # Update direction
        do_flip = rng.uniform(size=n_runs) < pflip
        direct[do_flip] *= -1

        # If enforcing finite domain, check if at boundary
        if finite_domain:
            direct[pos >= xb] = -1
            direct[pos <= -xb] = 1
        
    runs = np.vstack(runs).T
    
    return runs

def chemotax3_lt(p0=1e-1, h=1e-1, g=1e-2, n_runs=10000, tf=10000, x_init=0, apply_grad=False, finite_domain=False, xb=250, seed=1234):
    '''
    Simulates chemotaxis with or without gradient on a finite domain
    or on an infinite one. Emphasizes probability of flipping, compared
    to chemotax's run length. Averages every so often to avoid running out
    of memory on long runs.
    '''

    rng = np.random.default_rng(seed)

    # Simulate
    runs = []
    mean = []
    pos = np.zeros(shape=(n_runs,)) + x_init # Initialize position
    direct = rng.choice([-1, 1], size=n_runs) # 50/50 random init direction
    for t in range(tf):
        runs.append(pos) # Append position

        # Save mean, clear runs
        if t % 10000 == 0:
            runs = np.vstack(runs).T # (r x t)
            mean.append(runs.mean(axis=0))
            runs = []
        
        pos = pos + direct # Update position
        
        # Select either static pflip or fcn of position
        if apply_grad:
            pflip = p0 * (1 - h * np.tanh(g * pos))  # Update pflip
        else:
            pflip = p0 * np.ones(shape=(n_runs,))
        
        # Update direction
        do_flip = rng.uniform(size=n_runs) < pflip
        direct[do_flip] *= -1

        # If enforcing finite domain, check if at boundary
        if finite_domain:
            direct[pos >= xb] = -1
            direct[pos <= -xb] = 1
        
    runs = np.vstack(runs).T
    mean.append(runs.mean(axis=0))
    mean = np.hstack(mean).reshape(1, -1)
    
    return mean

def chemotax3_squish(p0=1e-1, h=1e-1, g=1e-2, n_runs=10000, tf=10000, x_init=0, apply_grad=False, finite_domain=False, xb=250, seed=1234):
    '''
    Simulates chemotaxis with or without gradient on a finite domain
    or on an infinite one. Emphasizes probability of flipping, compared
    to chemotax's run length. Boundary is "squishy". If the agent hits it, 
    it doesn't go anywhere, it doesn't turn around automatically like in
    the reflecting boundary case.
    '''

    rng = np.random.default_rng(seed)

    # Simulate
    runs = []
    pos = np.zeros(shape=(n_runs,)) + x_init # Initialize position
    direct = rng.choice([-1, 1], size=n_runs) # 50/50 random init direction
    squish = np.zeros(shape=(n_runs,)).astype(bool)
    for t in range(tf):
        runs.append(pos.copy()) # Append copy of position object, NOT a refernce to it
        pos[~squish] = pos[~squish] + direct[~squish] # Update position where not squished
        
        # Select either static pflip or fcn of position
        if apply_grad:
            pflip = p0 * (1 - h * np.tanh(g * pos))  # Update pflip
        else:
            pflip = p0 * np.ones(shape=(n_runs,))
        
        # Update direction
        do_flip = rng.uniform(size=n_runs) < pflip
        direct[do_flip] *= -1

        # Update squish vector
        if finite_domain:
            squish = np.zeros(shape=(n_runs,)).astype(bool) # Reset squish vector

            # Mark squished bugs
            squish[(pos >= xb) & (direct == 1)] = True
            squish[(pos <= -xb) & (direct == -1)] = True
        
    runs = np.vstack(runs).T
    
    return runs

def chemotax3_squish_ss(p0=1e-1, h=1e-1, g=1e-2, n_runs=10000, tf=10000, x_init=0, apply_grad=False, finite_domain=False, xb=250, seed=1234):
    '''
    Simulates chemotaxis with or without gradient on a finite domain
    or on an infinite one. Emphasizes probability of flipping, compared
    to chemotax's run length. Boundary is "squishy". If the agent hits it, 
    it doesn't go anywhere, it doesn't turn around automatically like in
    the reflecting boundary case. Keeps single state, doesn't store traces.
    '''

    rng = np.random.default_rng(seed)

    # Simulate
    pos = np.zeros(shape=(n_runs,)) + x_init # Initialize position
    direct = rng.choice([-1, 1], size=n_runs) # 50/50 random init direction
    squish = np.zeros(shape=(n_runs,)).astype(bool)
    for t in range(tf):
        pos[~squish] = pos[~squish] + direct[~squish] # Update position where not squished
        
        # Select either static pflip or fcn of position
        if apply_grad:
            pflip = p0 * (1 - h * np.tanh(g * pos))  # Update pflip
        else:
            pflip = p0 * np.ones(shape=(n_runs,))
        
        # Update direction
        do_flip = rng.uniform(size=n_runs) < pflip
        direct[do_flip] *= -1

        # Update squish vector
        if finite_domain:
            squish = np.zeros(shape=(n_runs,)).astype(bool) # Reset squish vector

            # Mark squished bugs
            squish[(pos >= xb) & (direct == 1)] = True
            squish[(pos <= -xb) & (direct == -1)] = True
    
    return pos

def chemotax3_ss(p0=1e-1, h=1e-1, g=1e-2, n_runs=10000, tf=10000, x_init=0, apply_grad=False, finite_domain=False, xb=250, seed=1234):
    '''
    Simulates chemotaxis with or without gradient on a finite domain
    or on an infinite one. Emphasizes probability of flipping, compared
    to chemotax's run length. Keeps single state, doesn't store traces.
    '''

    rng = np.random.default_rng(seed)

    # Simulate
    pos = np.zeros(shape=(n_runs,)) + x_init # Initialize position
    direct = rng.choice([-1, 1], size=n_runs) # 50/50 random init direction
    for t in range(tf):
        pos = pos + direct # Update position
        
        # Select either static pflip or fcn of position
        if apply_grad:
            pflip = p0 * (1 - h * np.tanh(g * pos))  # Update pflip
        else:
            pflip = p0 * np.ones(shape=(n_runs,))
        
        # Update direction
        do_flip = rng.uniform(size=n_runs) < pflip
        direct[do_flip] *= -1

        # If enforcing finite domain, check if at boundary
        if finite_domain:
            direct[pos >= xb] = -1
            direct[pos <= -xb] = 1
    
    return pos