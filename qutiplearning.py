'''
Code to (attempt to) simulate behavior of a Kerr resonator with coherent
one- and two-photon drives, both chosen to realize an effective
non-linear single-photon drive.
Everything is currently being done in the rotating frame.

Issues: 
Timescale to reach steady-state seems a little too long
Lowering drive mismatch causes disproportionately large oscillations 
'''

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
psi0 = fock(N, 0)
# psi0 = coherent(N, alpha_b)
times = np.logspace(-3, 3, 1000, base=10.0) * kappa
times1 = np.logspace(-3, 0, 500, base=10.0) * kappa
times2 = np.logspace(0, 3, 500, base=10.0) * kappa

kappa_2 = 0.2
diss = lindblad_dissipator(a)

# relative amplitude error
amp_err = 0.01

# tuned parameter, drive mismatch causes this to not be exactly 1
r = 1 + amp_err

c = 1
# escape rate
gamma_esc = c * (np.absolute(lambda_3_t * amp_err) ** 2) / kappa

# Equation 3, zero drive amplitude mismatch
block = lambda_3_t * a.dag() * (a.dag() * a - 1)
H_target = (block + block.dag()) + U * a.dag() * a.dag() * a * a

result = mesolve(H_target, psi0, times, [kappa * a], [num(N)])

# drive amplitudes
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

# dimensionless c
c = 1

# escape rate
gamma_esc = c * (np.absolute(lambda_3_t * amp_err) ** 2) / kappa

result2 = mesolve(H_disp, psi0, times1, [np.sqrt(gamma_esc) * a.dag(), kappa * a], [num(N)], options={'nsteps': 10000})

# plug in result from result_temp into result3
result_temp = mesolve(H_disp, psi0, times1, [np.sqrt(gamma_esc) * a.dag(), kappa * a], [], options={'nsteps': 10000})
result3 = mesolve(H_disp, result_temp.states[-1], times2, [np.sqrt(gamma_esc) * a.dag()], [num(N)], options={'nsteps': 10000})

'''
drive_term_rwa = lambda_1 * a.dag() + lambda_2 * a.dag() * a.dag()
H_rwa = U * a.dag() * a.dag() * a * a + delta_b * a.dag() * a + (drive_term_rwa + drive_term_rwa.dag())

psi0_rwa = coherent(N, alpha_b)
result3 = mesolve(H_rwa, psi0_rwa, times, [kappa *  a], [num(N)], options={'nsteps': 100000})
'''

result_g = mesolve(H_disp, psi0, times, [kappa * a], [], options={'nsteps': 10000})

# steady state solver
rho_ss = steadystate(H_disp, [kappa * a])
fexpt = expect(num(N), rho_ss)

final = result_g.states[-1]
one = fock(N, 1)
print(one.dag() * final * one)

g2_0 = expect(a.dag() * a.dag() * a * a, final) / (expect(a.dag() * a, final) ** 2)
print('non-linearity: ' + str(U / kappa))
print('intracavity correlation: ' + str(g2_0))


plt.figure()

# plt.plot(times, result.expect[0])
# plt.plot(times, np.append(result2.expect[0], result3.expect[0]))
# plt.plot(times, result2.expect[0])

plt.axhline(y=fexpt, color='r', ls='dashed', lw=1.5)

plt.xscale('log')
# plt.yscale('log')

plt.xlabel('Time')
plt.ylabel('Expectation values')
plt.legend(['ideal', 'mismatch_diss', 'ss'])

plt.show()
