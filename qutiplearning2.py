'''
Code to compare the ME solver with the MC solver.
'''

import numpy as np
import matplotlib.pyplot as plt

from qutip import *

# two-level system
N = 2

# cavity mode operator
a = destroy(N)

# dissipation term
kappa = 0.5

# Kerr non-linearity
U = 0.4 * kappa

# desired drive amplitude
lambda_3_t = 2 * kappa

# displacement parameter
alpha = lambda_3_t / (2 * U)

# solving the master equation
alpha_b = lambda_3_t / (2 * U)
psi0 = fock(N, 0)
# psi0 = coherent(N, alpha_b)
# times = np.logspace(-3, 3, 500, base=10.0) * kappa
times = np.linspace(0, 100, 500) * kappa

kappa_2 = 0.2

amp_err = 0.1

# tuned parameter, drive mismatch causes this to not be exactly 1
r = 1 + amp_err

c = 1
# escape rate
gamma_esc = c * (np.absolute(lambda_3_t * amp_err) ** 2) / kappa

# Equation 3, zero drive amplitude mismatch
block = lambda_3_t * a.dag() * (a.dag() * a - 1)
H_target = (block + block.dag()) + U * a.dag() * a.dag() * a * a

# result_mc = mcsolve(H_target, psi0, times, [kappa * a], [num(N)], ntraj=1000)
# result_me = mesolve(H_target, psi0, times, [kappa * a], [num(N)])


delta_b = -1 * (np.absolute(lambda_3_t) ** 2) / U

lambda_1 = lambda_3_t * (-r + (np.absolute(lambda_3_t) ** 2) / (2 * (U ** 2)) + 1j * kappa / (4 * U))
lambda_2 =  -1 * (lambda_3_t ** 2) / (4 * U)
lambda_1_t = lambda_1 + alpha * delta_b + 2 * np.conjugate(alpha) * lambda_2 + 2 * U * (np.absolute(alpha) ** 2) * alpha - 0.5 * 1j * kappa * alpha
lambda_2_t = lambda_2 + U * alpha * alpha

delta_t = delta_b + 4 * U * (np.absolute(alpha_b) ** 2)

drive_term = lambda_1_t * a.dag() + lambda_2_t * a.dag() * a.dag() + \
            lambda_3_t * a.dag() * a.dag() * a

H_disp = U * a.dag() * a.dag() * a * a + delta_t * a.dag() * a + \
            (drive_term + drive_term.dag())

result2_mc = mcsolve(H_disp, psi0, times, [kappa * a], [num(N)], ntraj=1000)
result2_me = mesolve(H_disp, psi0, times, [kappa * a], [num(N)])

print(H_target)
print(H_disp)

plt.figure()

# plt.plot(times, result_mc.expect[0])
plt.plot(times, result2_mc.expect[0])
# plt.plot(times, result_me.expect[0])
plt.plot(times, result2_me.expect[0])

# plt.xscale('log')

plt.xlabel('Time')
plt.ylabel('Expectation values')
# plt.legend(['ideal_mc', 'mismatch_mc', 'ideal_me', 'mismatch_me'])
plt.legend(['mismatch_mc', 'mismatch_me'])

plt.show()