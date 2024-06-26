import numpy as np
import matplotlib.pyplot as plt

from qutip import *

# two-level system
N = 2

# cavity mode operator
a = destroy(N)

# Kerr non-linearity
U = 0.5

# displacement parameter
alpha = 0.5

# tuned parameter, drive mismatch causes this to not be exactly 1
r = 1

# desired drive amplitude
lambda_3_t = 2 * U * alpha

# Equation 3
block = lambda_3_t * a.dag() * (a.dag() * a - r)
H_target = (block + block.dag()) + U * a.dag() * a.dag() * a * a

# solving the master equation
alpha_b = lambda_3_t / (2 * U)
psi0 = coherent(N, alpha_b)
times = np.logspace(-3, 3, 500, base=10.0)

# dissipation term
kappa = 0.5
diss = lindblad_dissipator(a)

result = mesolve(H_target, psi0, times, [kappa * diss], [a.dag() * a])

# relative amplitude error
amp_err = 0.1

# drive amplitudes
lambda_1 = 1.0
lambda_2 =  -1 * (lambda_3_t ** 2) / (4 * U)
lambda_1_t = -1 * lambda_3_t * (1 + amp_err)
lambda_2_t = lambda_2 + U * alpha * alpha
delta_b = -1 * (abs(lambda_3_t) ** 2) / U


H_RWA = U * a.dag() * a.dag() * a * a + delta_b * a.dag() * a + \
        lambda_1_t * a.dag() + lambda_2 * a.dag() * a.dag() + \
        a * np.conjugate(lambda_1_t) + a * a * np.conjugate(lambda_2_t)

delta_t = delta_b + 4 * U * (abs(alpha_b) ** 2)

drive_term = lambda_1_t * a.dag() + lambda_2_t * a.dag() * a.dag() + \
            lambda_3_t * a.dag() * a.dag() * a

print(drive_term)

psi0 = coherent(N, alpha_b)

H_disp = U * a.dag() * a.dag() * a * a + delta_t * a.dag() * a + \
            (drive_term + drive_term.dag())

diss2 = lindblad_dissipator(a + alpha_b)
result2 = mesolve(H_disp, psi0, times, [kappa * diss2], [a.dag() * a])

plt.figure()

plt.plot(times, result.expect[0])
plt.plot(times, result2.expect[0])

plt.xscale('log')

plt.xlabel('Time')
plt.ylabel('Expectation values')
plt.legend('cavity photon number')

plt.show()