import numpy as np
import matplotlib.pyplot as plt

from qutip import *

# two-level system
N = 32

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

# relative amplitude error
amp_err = 0

# tuned parameter, drive mismatch causes this to not be exactly 1
r = 1 + amp_err

psi0 = fock(N, 0)

#times = np.linspace(0, 8, 500) * kappa
times = np.logspace(-3, 3, 1000, base=10.0) * kappa

delta_b = -1 * (np.absolute(lambda_3_t) ** 2) / U

lambda_1 = lambda_3_t * (-r + (np.absolute(lambda_3_t) ** 2) / (2 * (U ** 2)) + 1j * kappa / (4 * U))
lambda_2 =  -1 * (lambda_3_t ** 2) / (4 * U)
lambda_1_t = lambda_1 + alpha * delta_b + 2 * np.conjugate(alpha) * lambda_2 + 2 * U * (np.absolute(alpha) ** 2) * alpha - 0.5 * 1j * kappa * alpha
lambda_2_t = lambda_2 + U * alpha * alpha

delta_t = delta_b + 4 * U * (np.absolute(alpha_b) ** 2)

drive_term = lambda_1 * a.dag() + lambda_2 * a.dag() * a.dag() 

H_rwa = U * a.dag() * a.dag() * a * a + delta_t * a.dag() * a + \
            (drive_term + drive_term.dag())

drive_term_disp = lambda_1_t * a.dag() + lambda_2_t * a.dag() * a.dag() + \
            lambda_3_t * a.dag() * a.dag() * a

H_disp = U * a.dag() * a.dag() * a * a + delta_t * a.dag() * a + \
            (drive_term_disp + drive_term_disp.dag())


D = displace(N, alpha_b)

block = lambda_3_t * a.dag() * (a.dag() * a - r)
H_target = (block + block.dag()) + U * a.dag() * a.dag() * a * a

result_ss = mesolve(H_target, psi0, times, [kappa * a], [], options={'nsteps': 10000})
print(expect(num(N), result_ss.states[-1]))

rwa_ss = steadystate(H_rwa, [kappa * (a + alpha_b)])
target_ss = steadystate(H_target, [kappa * a])

print(H_disp - H_target)

print(expect(num(N), rwa_ss))
print(expect(num(N), target_ss))


'''
# sanity check: check if H_disp is equal to H_target with the chosen parameters

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

print(H_disp - H_target)
'''

# psi = fock_dm(N, 0)
# disp = coherent_dm(N, alpha_b)
# print(disp)
# print(D * psi * D.dag())

# print(fexpt_rwa)
# print(fexpt_target)


'''

print(expect(a.dag() * a, rwa_ss))


drive_term = lambda_1_t * a.dag() + lambda_2_t * a.dag() * a.dag() + \
            lambda_3_t * a.dag() * a.dag() * a

H_disp = U * a.dag() * a.dag() * a * a + delta_t * a.dag() * a + \
            (drive_term + drive_term.dag())



times = np.linspace(0, (1/lambda_3_t), 500)

step2 = mesolve(H_target, psi0, times, [kappa * a], [])
psi_s2 = step2.final_state
'''