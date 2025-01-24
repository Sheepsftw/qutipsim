import numpy as np
import matplotlib.pyplot as plt

from qutip import *

from enum import Enum

import sys

np.set_printoptions(threshold=sys.maxsize)

# Goal: find lambda1 given sim 

# sim parameters

ramp_lambda_1 = False
time_const = 0.1

tolerance = 0.16


N = 64
kappa = 1
lambda_3_t = 2 * kappa

U = 0
# U = 0.4 * kappa

alpha = 2.5

a = destroy(N)

def coherence_err(state):
    avg_alpha = expect(a, state)
    error1 = avg_alpha - alpha
    error2 = expect(a.dag() * a, state) - alpha * np.conj(alpha)
    error3 = expect(a * a, state) - (alpha ** 2)
    # return 100 * np.abs(error1) + np.abs(error2) + np.abs(error3)
    return 100 * np.abs(error1) + np.abs(error2) + np.abs(error3)




psi0 = fock(N, 0)
temp = coherent(N, alpha)

# TODO: check if we can extend time_const and lower lambda
lambda_1 = -1.33 + 6.81j
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
    

    
lambda_2_const = 2

def lambda_2_coeff(t, args):
    return lambda_2_const * t


lambda_1_test_re = 0.05
lambda_1_test_im = 0.05

lambda_2_test_re = 0.0005
lambda_2_test_im = 0.0005

lambda_1_step_re = 1
lambda_1_step_im = 1

lambda_2_step_re = 1
lambda_2_step_im = 1

# delta_b = -1 * (np.absolute(lambda_3_t) ** 2) / U # + 4 * U * alpha ** 2?
delta_b = -10


# TODO: organize for lambda_1 vs lambda_2 optimization
def run_sim(lambda_1, lambda_2, time_const, log_alpha=False):
    times = np.linspace(0, time_const, 100)
    H0 = U * a.dag() * a.dag() * a * a + delta_b * a.dag() * a
    H1 = lambda_1 * a.dag() + (lambda_1 * a.dag()).dag()
    H2 = lambda_2 * a.dag() * a.dag() + (lambda_2 * a.dag() * a.dag()).dag()

    H = [H0, [H1, lambda_1_coeff]]
    # H = [H0, [H1, lambda_1_coeff], [H2, lambda_2_coeff]]

    result = mesolve(H, psi0, times, [kappa * a], [])
    if log_alpha:
        print('alpha: ' + str(expect(a, result.states[-1])))
    return coherence_err(result.states[-1])


start_l1 = 0
min_err = 100
for i1 in range(-10, 10):
    for i2 in range(-10, 10):
        curr_l1 = i1 + i2*1j
        curr_err = run_sim(curr_l1, 0, time_const)
        if curr_err < min_err:
            start_l1 = curr_l1
            min_err = curr_err
        print('done: ' + str(i1) + ' ' + str(i2) + 'j')
        
lambda_1 = start_l1

# lambda_1 = -7.866843215085964+12.73304724105866j

'''
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
# lambda_2 = -2.66 - 0.53j
lambda_2 = -0.3j


err = run_sim(lambda_1, lambda_2, time_const)

while err > tolerance:

    test_lambda_1_re = lambda_1 + lambda_1_test_re
    test_lambda_1_im = lambda_1 + lambda_1_test_im * 1j

    # test_lambda_2_re = lambda_2 + lambda_2_test_re
    # test_lambda_2_im = lambda_2 + lambda_2_test_im * 1j

    err_lambda_1_re = run_sim(test_lambda_1_re, lambda_2, time_const)
    err_lambda_1_im = run_sim(test_lambda_1_im, lambda_2, time_const)

    # err_lambda_2_re = run_sim(lambda_1, test_lambda_2_re, time_const)
    # err_lambda_2_im = run_sim(lambda_1, test_lambda_2_im, time_const)

    # grad_lambda_2_re = (err_lambda_2_re - err)
    # grad_lambda_2_im = (err_lambda_2_im - err)

    print('time_const: ' + str(time_const))
    print('lambda_1: ' + str(lambda_1))
    # print('lambda_2: ' + str(lambda_2))

    # print('lambda_2_re_grad: ' + str(grad_lambda_2_re))
    # print('lambda_2_im_grad: ' + str(grad_lambda_2_im))
    grad_lambda_1_re = (err_lambda_1_re - err)
    grad_lambda_1_im = (err_lambda_1_im - err)

    lambda_1 = lambda_1 - grad_lambda_1_re * lambda_1_step_re - grad_lambda_1_im * lambda_1_step_im * 1j
    # lambda_2 = lambda_2 - grad_lambda_2_re * lambda_2_step_re - grad_lambda_2_im * lambda_2_step_im * 1j

    new_err = run_sim(lambda_1, lambda_2, time_const, True)

    print('err ' + str(time_const) + ': ' + str(new_err))

    err = new_err
    

xvec = np.linspace(-10, 10, 200)
times = np.linspace(0, time_const, 100)
# H0 = a.dag() * a
H0 = U * a.dag() * a.dag() * a * a + delta_b * a.dag() * a
H1 = lambda_1 * a.dag() + (lambda_1 * a.dag()).dag()
H2 = lambda_2 * a.dag() * a.dag() + (lambda_2 * a.dag() * a.dag()).dag()
H = [H0, [H1, lambda_1_coeff]]
# H = [H0, [H1, lambda_1_coeff], [H2, lambda_2_coeff]]

result = mesolve(H, psi0, times, [kappa * a], [])
print('alpha: ' + str(expect(a, result.states[-1])))

w = wigner(result.states[-1], xvec, xvec)
# w = wigner(coherent_dm(N, alpha), xvec, xvec)
# print('alpha: ' + str(alpha))

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
print('final_alpha: ' + str(final_alpha))
state2 = displace(N, -final_alpha) * final_result.states[-1] * displace(N, final_alpha)

g2_0 = expect(a.dag() * a * a.dag() * a, state2) / (expect(a.dag() * a, state2) ** 2)
print('g2_0: ' + str(g2_0))
w = wigner(final_result.states[-1], xvec, xvec)
# w = wigner(coherent_dm(N, alpha), xvec, xvec)
# print('alpha: ' + str(alpha))

plt.figure()

plot = plt.contourf(xvec, xvec, w, 100)
plt.xlabel('Real')
plt.ylabel('Imag')
plt.colorbar(plot)

plt.show()




'''
amp_err = 0
r = 1 + amp_err

delta_b = -1 * (np.absolute(lambda_3_t) ** 2) / U

lambda_1 = lambda_3_t * (-r + (np.absolute(lambda_3_t) ** 2) / (2 * (U ** 2)) + 1j * kappa / (4 * U)) + 20j
lambda_2 =  -1 * (lambda_3_t ** 2) / (4 * U)
lambda_1_t = lambda_1 + alpha * delta_b + 2 * np.conjugate(alpha) * lambda_2 + 2 * U * (np.absolute(alpha) ** 2) * alpha - 0.5 * 1j * kappa * alpha
lambda_2_t = lambda_2 + U * alpha * alpha
'''