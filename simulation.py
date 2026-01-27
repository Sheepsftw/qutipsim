from main import Simulation, ModelArgument
import numpy as np

from qutip import *

import os

'''
lambda_1s = np.load(os.path.join('data', '1 lambda_1s_noramp_findlambda2.npy'))
time_consts = np.load(os.path.join('data', '1 time_const_noramp_findlambda2.npy'))
lambda_2s = np.load(os.path.join('data', '1 lambda_2s_noramp_findlambda2.npy'))
g2_0 = np.load(os.path.join('data', '1 g2_0_noramp_findlambda2.npy'))

g2_0_short = np.zeros(lambda_1s.size, dtype=np.complex_)

args = ModelArgument()

s = Simulation(N=64, alpha=2, kappa=1, U=0.4, args=args)

ratio = np.linspace(0.9, 1.1, 101)

for index in range(ratio.size):
    err, state = s.run_sim(lambda_1=lambda_1s[0] * ratio[index],
            lambda_2=0,
            time_const=time_consts[0])
    
    # print(err)
    state2 = s.evolve_lambda_3(init_state=state, is_short=True)
    g2_0 = s.check_g2_0(state2)
    print(g2_0)

args.optim_lambda1 = True
args.log_loops=True
'''

# lambdas = s.optimize(init_lambda1=100j, init_lambda2=0, time_const=time_consts[0], max_loops=200)
# print('lambda_1:' + str(lambdas['lambda_1']))


def unit_test_fock_space():
    N1 = 64
    N2 = 100
    alpha = 2.5
    args = ModelArgument()
    args.include_u = True
    s1 = Simulation(N=N1, alpha=2.5, kappa=1, U=0.4, args=args)
    s2 = Simulation(N=N2, alpha=2.5, kappa=1, U=0.4, args=args)
    
    print('s1: ' + str(s1.run_sim(lambda_1=1, lambda_2=1, time_const=0.4)[0]))
    print('s2: ' + str(s2.run_sim(lambda_1=1, lambda_2=1, time_const=0.4)[0]))
    

def unit_test_ramp():
    N = 64
    alpha = 2.5
    kappa = 1
    U = 0.4
    time_const = 0.4
    args = ModelArgument()
    
    # args.lambda_1_rate=0.1
    args.lambda_1_test=0.1
    # args.lambda_2_rate=0.1
    args.lambda_2_test=0.1
    
    
    args.optim_lambda1 = True
    args.ramp_lambda_1 = True
    args.t1_len = 0.1
    args.t2_len = 0.2
    args.lambda_1_t4 = 12 + 0.625j
    args.log_loops = True
    
    s1 = Simulation(N=N, alpha=alpha, kappa=kappa, U=U, args=args)
    params = s1.optimize(init_lambda1=8-1j, init_lambda2=0, time_const=time_const, max_loops=500)
    print(params['lambda_1'])
    
    


def init_error(): 
    N = 100
    alpha = 2
    alphas = np.linspace(0.1, 2, 20)
    args = ModelArgument()
    dist = 0.025
    s = Simulation(N=N,alpha=alpha, kappa=1, U=0.4, args=args)
    
    errs = np.zeros(20)
    g2_0 = np.zeros(20)
    
    for i in range(20):
        curr_alpha = alphas[i]
        
        curr_s = Simulation(N=N,alpha=curr_alpha, kappa=1, U=0.4, args=args)
        
        init_state = coherent(N, curr_alpha + curr_alpha * dist * (1 + 1j))
        final_state = curr_s.evolve_lambda_3(init_state, is_short=True)
        errs[i] = curr_s.coherence_err(init_state)
        g2_0[i] = curr_s.check_g2_0(final_state)
        print('alpha ' + str(curr_alpha) + ': ' + str(g2_0[i]))
        
    
    
    g2_0s = np.zeros([51, 51])
    alpha_err_re = np.linspace(-alpha*dist, alpha*dist, 51)
    alpha_err_im = np.linspace(-alpha*dist, alpha*dist, 51)
    
    for a in range(0, alpha_err_re.size):
        for b in range(0, alpha_err_im.size):
            init_state = coherent(N, alpha + alpha_err_re[a] + (alpha_err_im[b] * 1j))
            final_state = s.evolve_lambda_3(init_state, is_short=True)
            g2_0s[a][b] = s.check_g2_0(final_state)
            print('loop ' + str(a * 51 + b) + ': ' + str(g2_0s[a][b]))
    
    
    np.save(os.path.join('data', '0.025 alphavsg2 relative, init_err_g2_0_short.npy'), g2_0)
    np.save(os.path.join('data', '0.025 alphavserr relative, init_err_g2_0_short.npy'), errs)
    
    np.save(os.path.join('data', '2 rel_alpha_err g2_0 alpha=2 t=1.npy'), g2_0s)


def test_evol():
    N = 120
    alpha = 2
    args = ModelArgument()
    s = Simulation(N=N,alpha=alpha, kappa=1, U=0.044, args=args)
    init_state = coherent(N, alpha)
    final_state = s.evolve_lambda_3(init_state, is_short=True)
    print(final_state)


def lambda_3_vs_n():

    for index in range(13):
        args = ModelArgument()
        alphas = np.linspace(2, 5, 13)
        s = Simulation(N=64, alpha=alphas[index], kappa=1, U=0.4, args=args)
    s.evolve_lambda_3()


def tau_vs_g2_0():
    N = 100
    alpha = 2
    kappa = 1
    U = 0.036
    
    args = ModelArgument()
    args.include_u = True
    
    time_consts = np.logspace(-2, -1, 30)
    
    args.lambda_1_rate=0.1
    args.lambda_1_test=0.1
    args.lambda_2_rate=0.1
    args.lambda_2_test=0.1
    


def check_ramp_lambda2():
    N = 100
    alpha = 2
    kappa = 1
    U = 0.0364
    
    args = ModelArgument()
    time_const = 0.4
    
    args.optim_lambda1 = True
    args.ramp_lambda_1 = True
    args.t1_len = 0.01
    args.t2_len = 0.2
    args.lambda_1_t4 = 11.5 + 0.625j
    args.log_loops = True
    # args.log_alpha = True
    
    args.lambda_1_rate=10
    args.lambda_1_test=0.1
    args.lambda_2_rate=0.1
    args.lambda_2_test=0.1
    
    s1 = Simulation(N=N, alpha=alpha, kappa=kappa, U=U, args=args)
    
    params = s1.optimize(init_lambda1=1+1j, init_lambda2=0, time_const=time_const, max_loops=100)
    
    err, state = s1.run_sim(params['lambda_1'], 0, time_const)
    state2 = s1.evolve_lambda_3(state, is_short=True)
    g2 = s1.check_g2_0(state2)
    print('g2: ' + str(g2))
    
    print('init_lambda_1: ' + str(params['lambda_1']))
    
    args.optim_lambda2 = True
    args.optim_lambda1 = False
    args.include_u = True
    
    s2 = Simulation(N=N, alpha=alpha, kappa=kappa, U=U, args=args)
    
    min_err = 100000
    start_l2 = 0
    
    for i1 in range(-5, 5):
        for i2 in range(-5, 5):
            curr_l2 = 2 * i1 + 2 * i2*1j
            curr_err, state = s2.run_sim(params['lambda_1'], curr_l2, time_const)
            if curr_err < min_err:
                start_l2 = curr_l2
                min_err = curr_err
            print('done: ' + str(i1) + ' ' + str(i2) + 'j')
            print('curr_err: ' + str(curr_err))
    
    params2 = s2.optimize(init_lambda1=params['lambda_1'], init_lambda2=start_l2, time_const=time_const, max_loops=120)
    
    err, state_2 = s2.run_sim(params2['lambda_1'], params['lambda_2'], time_const)
    # err, state_2 = s2.run_sim(8.853155253773249-1.6828108742979953j, -0.49724454304319765+1.0326096811553802j, time_const)
    state2_2 = s2.evolve_lambda_3(state_2, is_short=True)
    g2_2 = s2.check_g2_0(state2_2)
    
    
    args_noramp = ModelArgument()
    
    args_noramp.optim_lambda1 = True
    args_noramp.log_loops = True
    
    args_noramp.lambda_1_rate=10
    args_noramp.lambda_1_test=0.1
    args_noramp.lambda_2_rate=0.1
    args_noramp.lambda_2_test=0.1
    
    s1_noramp = Simulation(N=N, alpha=alpha, kappa=kappa, U=U, args=args_noramp)
    
    params_noramp = s1_noramp.optimize(init_lambda1=1+1j, init_lambda2=0, time_const=time_const, max_loops=100)
    
    args_noramp.optim_lambda2=True
    args_noramp.optim_lambda1=False
    args_noramp.include_u = True
    
    s2_noramp = Simulation(N=N, alpha=alpha, kappa=kappa, U=U, args=args_noramp)
    
    min_err = 100000
    start_l2 = 0

    for i1 in range(-5, 5):
        for i2 in range(-5, 5):
            curr_l2 = 2 * i1 + 2 * i2*1j
            curr_err, state = s2_noramp.run_sim(params_noramp['lambda_1'], curr_l2, time_const)
            if curr_err < min_err:
                start_l2 = curr_l2
                min_err = curr_err
            print('done: ' + str(i1) + ' ' + str(i2) + 'j')
            print('curr_err: ' + str(curr_err))
    
    params2_noramp = s2_noramp.optimize(init_lambda1=params_noramp['lambda_1'], init_lambda2=start_l2, time_const=time_const, max_loops=120)
    
    err, state_2_noramp = s2_noramp.run_sim(params2_noramp['lambda_1'], params_noramp['lambda_2'], time_const)
    state2_2_noramp = s2_noramp.evolve_lambda_3(state_2_noramp, is_short=True)
    g2_2_noramp = s2_noramp.check_g2_0(state2_2_noramp)
    
    print('final_g2_ramp: ' + str(g2_2))
    print('final_g2_noramp: ' + str(g2_2_noramp))
    
    
    # ramping final_g2: (-2.914910334630831+6.017056051443695j) t = 0.4
    # no ramp final_g2: 1.1322828256611266
    
    # ramping final_g2: 1.63 t= 0.04
    # no ramp final_g2: 2.11 
    
    # without optim_lambda1:
    # ramping final_g2: 1.7727622252258162
    # no ramping final_g2: 2.3167140575680563
    # but the error is really low, what's going on??
    
    # lambda_1: 6.48 + 57.4j
    # lambda_2: -1.13 + 0.06j
    
    # lambda_1: 6 + 50j
    # lambda_2: -1 + 0.07j
    
    # U = 0.0364:
    # ramping final_g2: 3.1188995699874624e-09
    # no ramp final_g2: 4.094505099765006e-09
    # lambda_1 requirement relatively unchanged
    # lambda_2_noramp: -0.103 + 0.000495
    # lambda_2_ramp: -0.11 + 0.000505 (at this point just get rid of lambda_2? haha)
    
    # long time:
    # final_g2_ramp: (-0.4146396874617409+0.10928174763431957j)
    # final_g2_noramp: 9.044406430290092e-07
    
    
# last g2_0: (0.09787116235065163+3.898776924633105e-09j)

def lambda_2_error():
    N = 100
    alpha = 2
    U = 0.0364
    kappa = 1
    time_const = 0.1
    
    
    args = ModelArgument()
    
    args.optim_lambda1 = True
    args.log_loops = True
    
    args.lambda_1_rate=10
    args.lambda_1_test=0.1
    
    s = Simulation(N=N, alpha=alpha, kappa=kappa, U=U, args=args)
    
    params = s.optimize(init_lambda1=1+1j, init_lambda2=0, time_const=time_const, max_loops=100)
    
    args2 = ModelArgument()
    
    args2.optim_lambda2 = True
    args2.log_loops = True
    
    args2.lambda_2_rate=1
    args2.lambda_2_test=0.1
    args2.include_u = True
    
    min_err = 100000
    start_l2 = 0
    
    s2 = Simulation(N=N, alpha=alpha, kappa=kappa, U=U, args=args2)

    for i1 in range(-5, 5):
        for i2 in range(-5, 5):
            curr_l2 = 2 * i1 + 2 * i2*1j
            curr_err, state = s2.run_sim(params['lambda_1'], curr_l2, time_const)
            if curr_err < min_err:
                start_l2 = curr_l2
                min_err = curr_err
            print('done: ' + str(i1) + ' ' + str(i2) + 'j')
            print('curr_err: ' + str(curr_err))
    
    params2 = s2.optimize(init_lambda1=params['lambda_1'], init_lambda2=start_l2, time_const=time_const, max_loops=120)
    lambda_2_err = np.linspace(0, 0.1, 25)
    
    g2_0 = np.zeros(lambda_2_err.size)
    
    for index in range(lambda_2_err.size):
        curr_err = lambda_2_err[index]
        curr_lambda_2 = params2['lambda_2'] + curr_err
        
        err, state = s2.run_sim(lambda_1=params['lambda_1'], lambda_2=curr_lambda_2, time_const=time_const)
        
        new_state = s2.evolve_lambda_3(init_state=state, is_short=True)
        g2_0[index] = s2.check_g2_0(new_state)
        
        print('curr_g2: ' + str())
        
    
    np.save(os.path.join('data', 't=0.01 g2_0_lambda2_error.npy'), g2_0)
    
    # time_const = 0.1:
    # lambda_1 = 0.5921069551244873+20.49851442415686j
    # lambda_1 evolution phase: (-0.14559982660380097-0.00044941279750843033j)

'''
for index in range(0, lambda_1s.size):
    print('prev g2_0:' + str(g2_0[index]))

    state = s.run_sim(lambda_1=lambda_1s[index],
                    lambda_2=lambda_2s[index],
                    time_const=time_consts[index],
                    include_u=True)


    state2 = s.evolve_lambda_3(state, is_short=True)
    g2_0_short[index] = s.check_g2_0(state2)'

'''


def lambda_2_error_helper():
    N = 100
    alpha = 2
    U = 0.0364
    kappa = 1
    time_const = 0.4
    
    lambda_1 = (0.6212450111263796+5.494112962929153j)
    lambda_2 = (-0.11416742671335132+0.005940273551506413j)
    
    args2 = ModelArgument()
    
    args2.optim_lambda2 = True
    args2.log_loops = True
    
    args2.lambda_2_rate=1
    args2.lambda_2_test=0.1
    args2.include_u = True
    
    s2 = Simulation(N=N, alpha=alpha, kappa=kappa, U=U, args=args2)
    
    lambda_1_err_re = np.linspace(-0.2, 0.2, 51)
    lambda_1_err_im = np.linspace(-0.2, 0.2, 51)
    
    lambda_2_err_re = np.linspace(-0.06, 0.06, 51)
    lambda_2_err_im = np.linspace(-0.06, 0.06, 51)
    g2_0 = np.zeros([lambda_2_err_re.size, lambda_2_err_im.size])
    
    for index in range(lambda_2_err_re.size):
        for index2 in range(lambda_2_err_im.size):
            curr_err_l1 = lambda_1_err_re[index] + 1j * lambda_1_err_im[index2]
            # curr_err_l2 = lambda_2_err_re[index] + 1j * lambda_2_err_im[index2]
            # curr_lambda_2 = lambda_2 + curr_err_l2
            curr_lambda_1 = lambda_1 + curr_err_l1
            
            err, state = s2.run_sim(lambda_1=curr_lambda_1, lambda_2=lambda_2, time_const=time_const)
            
            new_state = s2.evolve_lambda_3(init_state=state, is_short=True)
            
            print('trace: ' + str(np.trace((new_state * new_state).full())))
            
            g2_0[index][index2] = s2.check_g2_0(new_state)
            
            print('curr_g2: ' + str(g2_0[index][index2]))
    
    # np.save(os.path.join('data', '3 t=0.01 maxerr=0.2 re_im g2_0_lambda1_error.npy'), g2_0)


def tau_error():
    N = 100
    alpha = 2
    U = 0.000364
    kappa = 1
    time_consts = np.linspace(0.1, 2, 20)
    g2_0s = np.zeros(20)
    errs = np.zeros(20)
    
    for index in range(20):
        time_const = time_consts[index]
        
        args = ModelArgument()
        
        args.optim_lambda1 = True
        args.log_loops = True
        # args.log_alpha = True
        
        args.lambda_1_rate=10
        args.lambda_1_test=0.1
        
        
        s1 = Simulation(N=N, alpha=alpha, kappa=kappa, U=U, args=args)
        
        params1 = s1.optimize(init_lambda1=1+1j, init_lambda2=0, time_const=time_const, max_loops=200)
        
        args.lambda_1_test=0.1
        args.lambda_1_rate=0.1
        args.lambda_2_rate=0.1
        args.lambda_2_test=0.1
        args.optim_lambda2=True
        args.optim_lambda1 = False
        args.include_u = True
        
        s2 = Simulation(N=N, alpha=alpha, kappa=kappa, U=U, args=args)
    
        min_err = 100000
        start_l2 = 0
    
        for i1 in range(-5, 5):
            for i2 in range(-5, 5):
                curr_l2 = 2 * i1 + 2 * i2*1j
                curr_err, state = s2.run_sim(params1['lambda_1'], curr_l2, time_const)
                if curr_err < min_err:
                    start_l2 = curr_l2
                    min_err = curr_err
                print('done: ' + str(i1) + ' ' + str(i2) + 'j')
                print('curr_err: ' + str(curr_err))
        
        params2 = s2.optimize(init_lambda1=params1['lambda_1'], init_lambda2=start_l2, time_const=time_const, max_loops=120)
        
        
        args.lambda_1_test = 0.1
        args.lambda_2_test = 0.1
        args.optim_lambda1 = True
        args.optim_lambda2 = True
        
        s3 = Simulation(N=N, alpha=alpha, kappa=kappa, U=U, args=args)
        params3 = s3.optimize(init_lambda1=params2['lambda_1'], init_lambda2=params2['lambda_2'], time_const=time_const, max_loops=120)
        
        
        err, state2 = s3.run_sim(params3['lambda_1'], params3['lambda_2'], time_const)
        errs[index] = err
        
        final_state = s3.evolve_lambda_3(init_state=state2, is_short=True)
        g2_0s[index] = s3.check_g2_0(final_state)
    
    np.save(os.path.join('data', 'fig 3', '2 tau_err time_consts'), time_consts)
    np.save(os.path.join('data', 'fig 3', '2 tau_err g2_0'), g2_0s)
    np.save(os.path.join('data', 'fig 3', '2 tau_err errs'), errs)


import matplotlib.pyplot as plt

def squeezing():
    N = 100
    alpha = 2
    # U = 0.0364
    U = 0.1
    kappa = 1
    time_const = 0.4
    a = destroy(N)
    
    lambda_1 = (0.6212450111263796+5.494112962929153j)
    
    args2 = ModelArgument()
    
    args2.optim_lambda1 = True
    args2.optim_lambda2 = True
    args2.log_loops = True
    
    args2.lambda_1_rate=10
    args2.lambda_1_test=0.1
    args2.lambda_2_rate=1
    args2.lambda_2_test=0.1
    args2.include_u = True
    
    s2 = Simulation(N=N, alpha=alpha, kappa=kappa, U=U, args=args2)
    '''
    min_err = 100000
    start_l2 = 0
    
    for i1 in range(-5, 5):
        for i2 in range(-5, 5):
            curr_l2 = 2 * i1 + 2 * i2*1j
            curr_err, state = s2.run_sim(lambda_1, curr_l2, time_const)
            if curr_err < min_err:
                start_l2 = curr_l2
                min_err = curr_err
            print('done: ' + str(i1) + ' ' + str(i2) + 'j')
            print('curr_err: ' + str(curr_err))
    
    lambdas = s2.optimize(init_lambda1=lambda_1, init_lambda2=start_l2, time_const=time_const, max_loops=120)
    
    err, state = s2.run_sim(lambda_1=lambda_1, lambda_2=lambdas['lambda_2'], time_const=time_const)
    
    # err, state = s2.run_sim(lambda_1=lambda_1, lambda_2=0, time_const=time_const)
    '''
    # print(new_state)
    xvec = np.linspace(-5, 5, 100)
    # w = wigner(state, xvec, xvec)
    
    # np.save(os.path.join('data', 'squeezed state with l1l2 U=0.2 t=0.4.npy'), w)
    w = np.load(os.path.join('data', 'squeezed state U=0.2.npy'))
    w2 = np.load(os.path.join('data', 'squeezed state with l1l2 U=0.2 t=0.4.npy'))
    
    # w = wigner(coherent_dm(N, 2), xvec, xvec)
    
    plt.figure()
    
    plot = plt.contourf(xvec, xvec, np.abs(w2), 100)
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.colorbar(plot)

    plt.show()
    
    
    
    
    


# test_evol()
# unit_test_ramp()
# check_ramp_lambda2()
# init_error()
# lambda_2_error()
# test = np.load(os.path.join('data', 't=0.01 g2_0_lambda2_error.npy'))
# print(test)
# lambda_2_error_helper()
# squeezing()
tau_error()