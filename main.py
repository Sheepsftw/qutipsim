import numpy as np
import matplotlib.pyplot as plt

from qutip import *
from qutip import QobjEvo

from enum import Enum

import sys
import os


# auxiliary class for holding all the simulation parameters
class ModelArgument:
    def __init__(self):
        self.optim_lambda1=False 
        self.optim_lambda2=False
        self.lambda_1_rate=1
        self.lambda_1_test=1
        self.lambda_2_rate=1
        self.lambda_2_test=1
        self.t1_len=0
        self.t2_len=0
        self.lambda_1_t4=0
        self.log_alpha=False 
        self.ramp_lambda_1=False 
        self.include_u=False
        self.log_loops=False
        self.cross_phase_modulation=False
        self.two_photon_loss=False


# main class with all the functions
class Simulation:
    def __init__(self, N, alpha, kappa, U, args: ModelArgument):
        self.args = args

        self.N = N
        self.a = destroy(N)
        self.alpha = alpha
        self.kappa = kappa
        self.U = U

        self.lambda_3_t = 2 * self.U * alpha
        self.psi0 = fock(N, 0)
        if U != 0:
            self.delta_b = -1 * (np.absolute(self.lambda_3_t) ** 2) / self.U 
        else:
            self.delta_b = -10
    

    def coherence_err(self, state):
        a = self.a
        alpha = self.alpha

        avg_alpha = expect(a, state)
        error1 = avg_alpha - alpha
        error2 = expect(a.dag() * a, state) - alpha * np.conj(alpha)
        error3 = expect(a * a, state) - (alpha ** 2)
        error4 = expect(a * a * a, state) - (alpha ** 3)
        # alternatively, 
        # return 100 * np.abs(error1) + np.abs(error2) + np.abs(error3)
        return 100 * np.abs(error1) + np.abs(error2) + np.abs(error3) + np.abs(error4)
    
    
    @staticmethod
    def lambda_1_coeff(t, args):
        if args['ramp_lambda_1']:
            if t < args['t1_len']:
                return args['lambda_1_t1_const'] * t
            elif t > args['t1_len'] + args['t2_len']:
                return args['slope'] * t + args['intercept']
            return 1

        return 1
    
    
    # not sure if this is ok
    @staticmethod
    def lambda_2_coeff(t, args):
        return (1 / args['time_const']) * t
        # return 2 * t


    @staticmethod
    def lambda_fun(t, args):
        a = destroy(args['N'])
        U = args['U']
        delta_b = args['delta_b']

        lambda_1 = args['lambda_1']
        lambda_2 = args['lambda_2']

        H0 = U * a.dag() * a.dag() * a * a + delta_b * a.dag() * a
        H1 = lambda_1 * a.dag() + (lambda_1 * a.dag()).dag()
        H2 = lambda_2 * a.dag() * a.dag() + (lambda_2 * a.dag() * a.dag()).dag()

        if args['ramp_lambda_1']:
            if t < args['t1_len']:
                H1 = H1 * args['lambda_1_t1_const'] * t
            elif t > args['t1_len'] + args['t2_len']:
                H1 = H1 * args['slope'] * t + args['intercept']

        if args['include_u']:
            H2 = H2 * (1 / args['time_const']) * t
        else:
            H0 = delta_b * a.dag() * a
            H2 = H2 * 0
        
        return H0 + H1 + H2

    
    def check_g2_0(self, state):
        a = self.a

        g2_0 = expect(a.dag() * a.dag() * a * a, state) / (expect(a.dag() * a, state) ** 2)
        return g2_0


    def evolve_lambda_3(self, init_state, is_short=False):
        N = self.N
        a = self.a
        kappa = self.kappa
        lambda_3_t = self.lambda_3_t
        alpha = self.alpha

        r_val = 1
        U = self.U
        lambda_1 = lambda_3_t * (-r_val + (np.absolute(lambda_3_t) ** 2) / (2 * (U ** 2)) + 1j * kappa / (4 * U))
        lambda_2 =  -1 * (lambda_3_t ** 2) / (4 * U)

        # TODO: use highest <n> in short-time evol, not ss
        times2 = np.logspace(-3, 3, 1000, base=10.0) / kappa

        # TODO: check time, 0.5*kappa or 1*kappa
        if is_short:
            times2 = np.logspace(-3, 0, 1000, base=10.0) / kappa

        delta_b = -1 * (np.absolute(lambda_3_t) ** 2) / U

        lambda_1 = lambda_3_t * (-r_val + (np.absolute(lambda_3_t) ** 2) / (2 * (U ** 2)) + 1j * kappa / (4 * U))
        lambda_2 =  -1 * (lambda_3_t ** 2) / (4 * U)

        drive_term = lambda_1 * a.dag() + lambda_2 * a.dag() * a.dag()

        new_H = U * a.dag() * a.dag() * a * a + delta_b * a.dag() * a + \
                    (drive_term + drive_term.dag())

        #TODO: progress-bar: text
        final_result = mesolve(new_H, init_state, times2, [kappa * a], [], options={'atol': '1e-15', 'rtol': '1e-15', 'max_step': '0.001', 'nsteps': 80000})
        state2 = displace(N, -alpha) * final_result.states[-1] * displace(N, alpha)
        
        # testing purposes
        '''
        max = 0
        for i in range(1000):
            state2 = displace(N, -alpha) * final_result.states[i] * displace(N, alpha)
            
            nexpect = expect(a.dag() * a, state2)
            print(str(i) + ': ' + str(nexpect))
            if nexpect > max:
                max = nexpect
        
        return max
        '''
        return state2


    def run_sim(self, lambda_1, lambda_2, time_const):

        t1_len = self.args.t1_len
        t2_len = self.args.t2_len
        lambda_1_t4 = self.args.lambda_1_t4
        log_alpha = self.args.log_alpha
        ramp_lambda_1 = self.args.ramp_lambda_1
        include_u = self.args.include_u

        N = self.N
        a = self.a
        U = self.U
        delta_b = self.delta_b
        psi0 = self.psi0
        kappa = self.kappa

        times = np.linspace(0, time_const, 10000)
        
        # this code was broken due to a bug in QuTiP. The bug has since been fixed.
        '''
        H0 = U * a.dag() * a.dag() * a * a + delta_b * a.dag() * a
        if not include_u:
            H0 = delta_b * a.dag() * a
        
        H1 = lambda_1 * a.dag() + (lambda_1 * a.dag()).dag()
        H2 = lambda_2 * a.dag() * a.dag() + (lambda_2 * a.dag() * a.dag()).dag()

        # for some reason, [H0, [H1, self.lambda_1_coeff], H2] and [H0, [H1, self.lambda_1_coeff], [H2, self.lambda_2_coeff]]
        # have different behaviors even when H2 = 0.

        H = [H0, [H1, self.lambda_1_coeff], [H2, self.lambda_2_coeff]]
        if not include_u: 
            H = [H0, [H1, self.lambda_1_coeff]]
        
        # H = [H0 + H2, [H1, self.lambda_1_coeff]]
        # H = H0 + H1'
        '''

        args = {'N': N,
                'U': U,
                'delta_b': delta_b,
                'lambda_1': lambda_1,
                'lambda_2': lambda_2,
            
                'ramp_lambda_1': ramp_lambda_1,
                'include_u': include_u,}
        
        if ramp_lambda_1:
            slope = 0
            if np.abs(lambda_1) != 0:
                slope = (1 - (lambda_1_t4 / lambda_1)) / (t1_len + t2_len - time_const)
            intercept = 1 - slope * (t1_len + t2_len)
            ramp_args = {'t1_len': t1_len,
                         't2_len': t2_len,
                         'lambda_1_t1_const': 1 / t1_len,
                         'slope': slope,
                         'intercept': intercept}
            args.update(ramp_args)
        
        if include_u:
            time_const_arg = {'time_const': time_const}
            args.update(time_const_arg)

        H = QobjEvo(self.lambda_fun, args=args)

        result = mesolve(H, psi0, times, [kappa * a], [], args=args, options={'atol': '1e-15', 'rtol': '1e-15', 'max_step': '0.001'})
        if log_alpha:
            print('alpha: ' + str(expect(a, result.states[-1])))
        # return result.states[-1]
        # TODO: should return as dict for consistency
        return self.coherence_err(state=result.states[-1]), result.states[-1]
      
    
    def optimize(self, 
                 init_lambda1, 
                 init_lambda2,
                 time_const, 
                 max_loops):
        
        if not self.args.optim_lambda1 and not self.args.optim_lambda2:
            return {}
        
        lambda_1 = init_lambda1
        lambda_2 = init_lambda2
        
        lambda_1_test = self.args.lambda_1_test
        lambda_2_test = self.args.lambda_2_test

        err, final_state = self.run_sim(lambda_1=init_lambda1, 
                           lambda_2=init_lambda2, 
                           time_const=time_const)
        num_loops = 0

        while num_loops < max_loops:

            new_lambda_1 = lambda_1
            new_lambda_2 = lambda_2

            if self.args.optim_lambda1:

                test_lambda_1_re = lambda_1 + lambda_1_test
                test_lambda_1_im = lambda_1 + lambda_1_test * 1j

                err_lambda_1_re, final_state_re = self.run_sim(
                    test_lambda_1_re,
                    lambda_2,
                    time_const)
                
                err_lambda_1_im, final_state_im = self.run_sim(
                    test_lambda_1_im,
                    lambda_2,
                    time_const)
                
                grad_lambda_1_re = (err_lambda_1_re - err)
                grad_lambda_1_im = (err_lambda_1_im - err)

                new_lambda_1 = lambda_1 - grad_lambda_1_re * self.args.lambda_1_rate - grad_lambda_1_im * self.args.lambda_1_rate * 1j
            
            if self.args.optim_lambda2:
                test_lambda_2_re = lambda_2 + lambda_2_test
                test_lambda_2_im = lambda_2 + lambda_2_test * 1j

                err_lambda_2_re, final_state_re = self.run_sim(
                    lambda_1,
                    test_lambda_2_re,
                    time_const)

                err_lambda_2_im, final_state_im = self.run_sim(
                    lambda_1,
                    test_lambda_2_im,
                    time_const)
                
                grad_lambda_2_re = (err_lambda_2_re - err)
                grad_lambda_2_im = (err_lambda_2_im - err)

                new_lambda_2 = lambda_2 - grad_lambda_2_re * self.args.lambda_2_rate - grad_lambda_2_im * self.args.lambda_2_rate * 1j
            
            new_err, final_state = self.run_sim(new_lambda_1, 
                                   new_lambda_2, 
                                   time_const)

            if self.args.log_loops:
                if self.args.optim_lambda1:
                    print('lambda_1: ' + str(new_lambda_1))
                if self.args.optim_lambda2:
                    print('lambda_2: ' + str(new_lambda_2))
                print('err ' + str(num_loops) + ': ' + str(new_err))
            
            if new_err > err:
                lambda_1_test = lambda_1_test / 2
                lambda_2_test = lambda_2_test / 2
            else:
                lambda_1 = new_lambda_1
                lambda_2 = new_lambda_2
                err = new_err
            
            num_loops = num_loops + 1
        
        return {'lambda_1': lambda_1, 'lambda_2': lambda_2}
            

            


        