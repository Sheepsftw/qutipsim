'''
Code to demonstrate the effects of drive mismatch on the 
instantaneous second-order correlation g2_0.
The result should roughly match Figure 3C.
'''

import numpy as np
import matplotlib.pyplot as plt

from qutip import *

# Fock space limit
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

times = np.linspace(0, 2 / kappa, 1000, dtype=float)
times_kappa = np.array(times * kappa, dtype=float)

amp_errs = np.array([0.1, 0.056, 0.032, 0.018, 0.01, 0.0056])

# tuned parameter, drive mismatch causes this to not be exactly 1
r_vals = 1 + amp_errs

plt.figure()

for i in range(6): 
    # H params
    delta_b = -1 * (np.absolute(lambda_3_t) ** 2) / U

    lambda_1 = lambda_3_t * (-r_vals[i] + (np.absolute(lambda_3_t) ** 2) / (2 * (U ** 2)) + 1j * kappa / (4 * U))
    lambda_2 =  -1 * (lambda_3_t ** 2) / (4 * U)
    lambda_1_t = lambda_1 + alpha * delta_b + 2 * np.conjugate(alpha) * lambda_2 + 2 * U * (np.absolute(alpha) ** 2) * alpha - 0.5 * 1j * kappa * alpha
    lambda_2_t = lambda_2 + U * alpha * alpha

    delta_t = delta_b + 4 * U * (np.absolute(alpha_b) ** 2)

    drive_term = lambda_1_t * a.dag() + lambda_2_t * a.dag() * a.dag() + \
                lambda_3_t * a.dag() * a.dag() * a

    H_disp = U * a.dag() * a.dag() * a * a + delta_t * a.dag() * a + \
                (drive_term + drive_term.dag())
    
    result = mesolve(H_disp, psi0, times, [kappa * a], [a.dag() * a.dag() * a * a, (a.dag() * a)])

    g2_0 = result.expect[0] / (result.expect[1] ** 2)

    plt.plot(times_kappa, g2_0)
    # maybe cut off the starting few values?
    # plt.plot(times[2:], g2_0[2:])

plt.xlabel('Time kappa*t')
plt.ylabel('g2_0')

plt.yscale('log')
plt.legend(['0.1', '0.056', '0.032', '0.018', '0.01', '0.0056'])
plt.show()

