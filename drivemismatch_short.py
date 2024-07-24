'''
Code to demonstrate the effects of drive mismatch on short-term <n>.
The result should roughly match Figure 3B.
'''

import numpy as np
import matplotlib.pyplot as plt

from qutip import *

# Fock space limit
N = 64

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
amp_errs = np.array([0.1, 0.056, 0.032, 0.018, 0.01, 0.0056])

# tuned parameter, drive mismatch causes this to not be exactly 1
r_vals = 1 + amp_errs

# initial state
psi0 = fock(N, 0)

times = np.linspace(0, 2 / kappa, 1000)

# for use in plotting
opacities = [0.25, 0.4, 0.55, 0.7, 0.85, 1]

plt.figure()

for index in range(6): 
    # H params
    delta_b = -1 * (np.absolute(lambda_3_t) ** 2) / U

    lambda_1 = lambda_3_t * (-r_vals[index] + (np.absolute(lambda_3_t) ** 2) / (2 * (U ** 2)) + 1j * kappa / (4 * U))
    lambda_2 =  -1 * (lambda_3_t ** 2) / (4 * U)
    lambda_1_t = lambda_1 + alpha * delta_b + 2 * np.conjugate(alpha) * lambda_2 + 2 * U * (np.absolute(alpha) ** 2) * alpha - 0.5 * 1j * kappa * alpha
    lambda_2_t = lambda_2 + U * alpha * alpha

    delta_t = delta_b + 4 * U * (np.absolute(alpha_b) ** 2)

    drive_term = lambda_1_t * a.dag() + lambda_2_t * a.dag() * a.dag() + \
                lambda_3_t * a.dag() * a.dag() * a

    H_disp = U * a.dag() * a.dag() * a * a + delta_t * a.dag() * a + \
                (drive_term + drive_term.dag())
    
    result_disp = mesolve(H_disp, psi0, times, [kappa * a], [num(N)], options={'nsteps': 400000})

    disp_ss = steadystate(H_disp, [kappa * a])
    print(expect(num(N), disp_ss))

    plt.plot(times * kappa, result_disp.expect[0], color='g', alpha=opacities[index])


# Equation 3, zero drive amplitude mismatch
block = lambda_3_t * a.dag() * (a.dag() * a - 1)
H_target = (block + block.dag()) + U * a.dag() * a.dag() * a * a

result_target = mesolve(H_target, psi0, times, [kappa * a], [num(N)], options={'nsteps': 100000})


disp_ss = steadystate(H_target, [kappa * a])
expt = expect(num(N), disp_ss)
plt.axhline(y=expt, color='r', ls='dashed', lw=1.5)

plt.plot(times * kappa, result_target.expect[0])

plt.xlim([0, 2])
plt.ylim([0, 1])

plt.xlabel('Time kappa*t')
plt.ylabel('<n>')
plt.legend(['0.1', '0.056', '0.032', '0.018', '0.01', '0.0056', 'ss_target', 'H_target'])

plt.show()