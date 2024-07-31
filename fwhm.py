import numpy as np
import matplotlib.pyplot as plt

from qutip import *

# two-level system
N = 128

# cavity mode operator
a = destroy(N)

# dissipation term
kappa = 1

# desired drive amplitude
lambda_3_t = 2 * kappa

plt.figure()

i1_len = 50

fwhm = np.empty(i1_len)

# Kerr non-linearity
# U = 0.5 * lambda_3_t
U_vals = np.linspace(0.15, 0.5, i1_len) * np.array(lambda_3_t)
# U_vals = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] * np.array(lambda_3_t)
for i1 in range(len(U_vals)):
    U = U_vals[i1]

    # solving the master equation
    alpha = lambda_3_t / (2 * U)

    x_len = 100

    # relative amplitude error
    amp_err1 = np.logspace(-20, -0.5, x_len, base=10.0)
    amp_err2 = np.logspace(-0.5, -20, x_len, base=10.0) * -1
    amp_err = np.append(amp_err2, amp_err1)

    # tuned parameter, drive mismatch causes this to not be exactly 1
    r_vals = 1 + amp_err

    n_ss = np.empty([2*x_len])

    for i2 in range(2*x_len):
        r = r_vals[i2]
        delta_b = -1 * (np.absolute(lambda_3_t) ** 2) / U

        lambda_1 = lambda_3_t * (-r + (np.absolute(lambda_3_t) ** 2) / (2 * (U ** 2)) + 1j * kappa / (4 * U))
        lambda_2 =  -1 * (lambda_3_t ** 2) / (4 * U)
        lambda_1_t = lambda_1 + alpha * delta_b + 2 * np.conjugate(alpha) * lambda_2 + 2 * U * (np.absolute(alpha) ** 2) * alpha - 0.5 * 1j * kappa * alpha
        lambda_2_t = lambda_2 + U * alpha * alpha

        delta_t = delta_b + 4 * U * (np.absolute(alpha) ** 2)

        drive_term_disp = lambda_1_t * a.dag() + lambda_2_t * a.dag() * a.dag() + \
                    lambda_3_t * a.dag() * a.dag() * a

        H_disp = U * a.dag() * a.dag() * a * a + delta_t * a.dag() * a + \
                    (drive_term_disp + drive_term_disp.dag())

        disp_ss = steadystate(H_disp, [kappa * a])
        n_ss[i2] = expect(num(N), disp_ss)

    approx_max = n_ss[0]
    approx_min = 0.5
    half_min = (approx_max - approx_min) / 2 + 0.5

    idx = np.nonzero(n_ss < half_min)[0]

    fwhm[i1] = amp_err[idx[-1]] - amp_err[idx[0] - 1]
    print('U: ' + str(U) + ', FWHM: ' + str(fwhm[i1]))

plt.yscale('log')

plt.xlabel('U/lambda_3_t')
plt.ylabel('FWHM, amp_err vs <n>')

plt.plot(U_vals/np.array(lambda_3_t), fwhm)
# plt.show()
plt.savefig('fwhm.png')



    



