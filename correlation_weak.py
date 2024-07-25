'''
Code to demonstrate instantaneous second-order correlation g2_0
in the regime lambda3 <= kappa.
The result should roughly match Figure 4B.
'''

import numpy as np
import matplotlib.pyplot as plt

from qutip import *

# two-level system
N = 128

# cavity mode operator
a = destroy(N)

# dissipation term
kappa = 1

# Kerr non-linearity
U = 0.075 * kappa

# desired drive amplitude
lambda_3_t = np.array([0.5, 0.33, 0.22, 0.15, 0.10]) * kappa

# displacement parameter
alpha = lambda_3_t / (2 * U)

# solving the master equation
alpha_b = lambda_3_t / (2 * U)
psi0 = fock(N, 0)

times = np.logspace(-2, 2, 1000, base=10.0) / kappa

# relative amplitude error
amp_err = 0.01

# tuned parameter, drive mismatch causes this to not be exactly 1
r = 1 + amp_err

psi0 = fock(N, 0)

# for use in plotting
opacities = [0.2, 0.4, 0.6, 0.8, 1]

plt.figure()

for i in range(5):
    # H params
    delta_b = -1 * (np.absolute(lambda_3_t[i]) ** 2) / U

    lambda_1 = lambda_3_t[i] * (-r + (np.absolute(lambda_3_t[i]) ** 2) / (2 * (U ** 2)) + 1j * kappa / (4 * U))
    lambda_2 =  -1 * (lambda_3_t[i] ** 2) / (4 * U)
    lambda_1_t = lambda_1 + alpha[i] * delta_b + 2 * np.conjugate(alpha[i]) * lambda_2 + 2 * U * (np.absolute(alpha[i]) ** 2) * alpha[i] - 0.5 * 1j * kappa * alpha[i]
    lambda_2_t = lambda_2 + U * alpha[i] * alpha[i]

    delta_t = delta_b + 4 * U * (np.absolute(alpha_b[i]) ** 2)

    drive_term = lambda_1_t * a.dag() + lambda_2_t * a.dag() * a.dag() + \
                lambda_3_t[i] * a.dag() * a.dag() * a

    H_disp = U * a.dag() * a.dag() * a * a + delta_t * a.dag() * a + \
                (drive_term + drive_term.dag())
    
    result_disp = mesolve(H_disp, psi0, times, [kappa * a], [a.dag() * a.dag() * a * a, (a.dag() * a)], options={'nsteps': 40000})

    g2_0 = result_disp.expect[0] / (result_disp.expect[1] ** 2)

    plt.plot(times * kappa, g2_0)

plt.xlabel('Time kappa*t')
plt.ylabel('g2_0')

plt.xlim([10 ** -1, 10 ** 2])
plt.ylim([10 ** -5, 10 ** 1])

plt.xscale('log')
plt.yscale('log')
plt.legend(['0.5', '0.33', '0.22', '0.15', '0.1'])

plt.show()