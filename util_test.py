from util import Simulation, ModelArgument
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


def init_error(): 
    N = 64
    alpha = 3
    args = ModelArgument()
    s = Simulation(N=N,alpha=alpha, kappa=1, U=0.4, args=args)
    alpha_err_re = np.linspace(-0.05, 0.05, 25)
    alpha_err_im = np.linspace(-0.05, 0.05, 25)
    g2_0 = np.zeros([25, 25], dtype=np.complex_)
    for a in range(0, 25):
        for b in range(0, 25):
            init_state = coherent(N, alpha + alpha_err_re[a] + (alpha_err_im[b] * 1j))
            final_state = s.evolve_lambda_3(init_state, is_short=True)
            g2_0[a][b] = s.check_g2_0(final_state)
            print('loop ' + str((a-1) * 25 + b) + ': ' + str(g2_0[a][b]))
    
    np.save(os.path.join('data', '1 alpha3, init_err_g2_0_short.npy'), g2_0)



def lambda_3_vs_n():

    for index in range(13):
        args = ModelArgument()
        alphas = np.linspace(2, 5, 13)
        s = Simulation(N=64, alpha=alphas[index], kappa=1, U=0.4, args=args)
    s.evolve_lambda_3()

# last g2_0: (0.09787116235065163+3.898776924633105e-09j)

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

init_error()
