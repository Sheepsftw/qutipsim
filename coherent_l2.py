import numpy as np
import matplotlib.pyplot as plt

from qutip import *

from enum import Enum

import sys

np.set_printoptions(threshold=sys.maxsize)

# Goal: find lambda2 given lambda 1.

# sim parameters
ramp_lambda_1 = False

lambda_1 = -1.33 + 6.81j
time_const = 0.4

tolerance = 0.13

delta_b = 1
# delta_b = -1 * (np.absolute(lambda_3_t) ** 2) / U # + 4 * U * alpha ** 2?


N = 64
kappa = 1
lambda_3_t = 2 * kappa


U = 0.4 * kappa


alpha = 2.5

a = destroy(N)

def coherence_err(state):
    error1 = expect(a, state) - alpha
    error2 = expect(a.dag() * a, state) - alpha * np.conj(alpha)
    error3 = expect(a * a, state) - (alpha ** 2)
    return 100 * np.abs(error1) + np.abs(error2) + np.abs(error3)


psi0 = fock(N, 0)
temp = coherent(N, alpha)


# TODO: do not hardcode
lambda_1_t4 = 12 + 0.625j

times = np.linspace(0, time_const, 100)


t1_len = 0.1
t3_len = 0.1
# t2_len = 0.0576
t2_len = time_const - (t1_len + t3_len)
lambda_1_t1_const = 1 / t1_len
# lambda_1_t1_const = 0
# TODO: update this constant as lambda_1 changes
lambda_1_t3_const = ((lambda_1 - lambda_1_t4)/ lambda_1) / t3_len


def lambda_1_coeff(t, args):
    if ramp_lambda_1:
        if t < t1_len:
            return lambda_1_t1_const * t
        elif t > t1_len + t2_len:
            return (1 - lambda_1_t3_const) * (t - (t1_len + t2_len))
        return 1
    
    return 1
    

# why is this 2?
lambda_2_const = 2

def lambda_2_coeff(t, args):
    return lambda_2_const * t


# TODO: organize for lambda_1 vs lambda_2 optimization
def run_sim(lambda_1, lambda_2, time_const, log_alpha=False):
    times = np.linspace(0, time_const, 100)
    H0 = U * a.dag() * a.dag() * a * a + delta_b * a.dag() * a
    H1 = lambda_1 * a.dag() + (lambda_1 * a.dag()).dag()
    H2 = lambda_2 * a.dag() * a.dag() + (lambda_2 * a.dag() * a.dag()).dag()

    H = [H0, [H1, lambda_1_coeff], [H2, lambda_2_coeff]]

    result = mesolve(H, psi0, times, [kappa * a], [])
    if log_alpha:
        print('alpha: ' + str(expect(a, result.states[-1])))
    return coherence_err(result.states[-1])


'''
# sweep to find a good starting point for lambda2.
# can skip this if we already have it.
start_l2 = 0
min_err = 100

for i1 in range(-10, 10):
    for i2 in range(-10, 10):
        curr_l2 = i1 + i2*1j
        curr_err = run_sim(lambda_1, curr_l2, time_const)
        if curr_err < min_err:
            start_l2 = curr_l2
            min_err = curr_err
        print('done: ' + str(i1) + ' ' + str(i2) + 'j')
        
lambda_2 = start_l2
'''

lambda_2 = -2.66 - 0.53j

lambda_2_test_re = 0.0005
lambda_2_test_im = 0.0005

lambda_2_step_re = 1
lambda_2_step_im = 1

err = run_sim(lambda_1, lambda_2, time_const)

while err > tolerance:

    test_lambda_2_re = lambda_2 + lambda_2_test_re
    test_lambda_2_im = lambda_2 + lambda_2_test_im * 1j

    err_lambda_2_re = run_sim(lambda_1, test_lambda_2_re, time_const)
    err_lambda_2_im = run_sim(lambda_1, test_lambda_2_im, time_const)

    grad_lambda_2_re = (err_lambda_2_re - err)
    grad_lambda_2_im = (err_lambda_2_im - err)

    print('time_const: ' + str(time_const))
    print('lambda_2: ' + str(lambda_2))

    # print('lambda_2_re_grad: ' + str(grad_lambda_2_re))
    # print('lambda_2_im_grad: ' + str(grad_lambda_2_im))
    
    lambda_2 = lambda_2 - grad_lambda_2_re * lambda_2_step_re - grad_lambda_2_im * lambda_2_step_im * 1j

    new_err = run_sim(lambda_1, lambda_2, time_const, True)

    print('err ' + str(time_const) + ': ' + str(new_err))

    err = new_err
    
print('lambda_2: ' + str(lambda_2))

xvec = np.linspace(-10, 10, 200)
times = np.linspace(0, time_const, 100)
H0 = U * a.dag() * a.dag() * a * a + delta_b * a.dag() * a
H1 = lambda_1 * a.dag() + (lambda_1 * a.dag()).dag()
H2 = lambda_2 * a.dag() * a.dag() + (lambda_2 * a.dag() * a.dag()).dag()
H = [H0, [H1, lambda_1_coeff], [H2, lambda_2_coeff]]

result = mesolve(H, psi0, times, [kappa * a], [])
print('alpha: ' + str(expect(a, result.states[-1])))
w = wigner(result.states[-1], xvec, xvec)
# w = wigner(coherent_dm(N, alpha), xvec, xvec)
# print('alpha: ' + str(alpha))
print(coherence_err(result.states[-1]))

plt.figure()

plot = plt.contourf(xvec, xvec, w, 100)
plt.xlabel('Real')
plt.ylabel('Imag')
plt.colorbar(plot)

plt.show()

new_psi0 = result.states[-1]

r_val = 1
U = 0.4 * kappa
lambda_1 = lambda_3_t * (-r_val + (np.absolute(lambda_3_t) ** 2) / (2 * (U ** 2)) + 1j * kappa / (4 * U)) + 20j
lambda_2 =  -1 * (lambda_3_t ** 2) / (4 * U)

times2 = np.logspace(-3, 3, 1000, base=10.0) / kappa

delta_b = -1 * (np.absolute(lambda_3_t) ** 2) / U

lambda_1 = lambda_3_t * (-r_val + (np.absolute(lambda_3_t) ** 2) / (2 * (U ** 2)) + 1j * kappa / (4 * U))
lambda_2 =  -1 * (lambda_3_t ** 2) / (4 * U)

drive_term = lambda_1 * a.dag() + lambda_2 * a.dag() * a.dag()

new_H = U * a.dag() * a.dag() * a * a + delta_b * a.dag() * a + \
            (drive_term + drive_term.dag())

final_result = mesolve(new_H, new_psi0, times2, [kappa * a], [], options={'nsteps': 80000, 'progress_bar': 'text'})

final_alpha = expect(a, final_result.states[-1])
state2 = displace(N, -final_alpha) * final_result.states[-1] * displace(N, final_alpha)
# state2 = displace(N, -alpha) * final_result.states[-1] * displace(N, alpha)

print(expect(a.dag() * a.dag() * a * a, state2))

g2_0 = expect(a.dag() * a.dag() * a * a, state2) / (expect(a.dag() * a, state2) ** 2)

print('g2_0: ' + str(g2_0))
print('<n>: ' + str(expect(a.dag() * a, state2)))

w = wigner(state2, xvec, xvec)

plt.figure()

plot = plt.contourf(xvec, xvec, w, 100)
plt.xlabel('Real')
plt.ylabel('Imag')
plt.colorbar(plot)

plt.show()