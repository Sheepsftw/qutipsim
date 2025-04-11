import numpy as np
from matplotlib import pyplot as plt
import numpy.polynomial.polynomial as poly

import os

from qutip import *

kappa = 1


X = np.load(os.path.join('data', '1 alpha_err_re.npy'))
Y = np.load(os.path.join('data', '1 alpha_err_im.npy'))


lambda2s = np.load(os.path.join('data', '1 lambda_2s_noramp_findlambda2.npy'))

temp1 = np.load(os.path.join('data', '1 ramp_lambda1_g2_0s.npy'))
temp2 = np.load(os.path.join('data', '1 ramp_lambda1_times.npy'))

# 0.001: 0.1259889931744587-1.2105739233172566e-05j
# 0.0001: 0.1259760763190976+1.9640621655143902e-06j


# print('lambda1: ' + str(lambda1s[50]))
# print('g2_0: ' + str(g2_0_1[50]))
'''

lambda1sd = lambda1s - lambda1s[50]
lambda1sr = np.real(lambda1s)

magnitudes = np.abs(lambda1sd)
magnitudes2 = np.abs(lambda1s)
ratio = magnitudes / np.abs(lambda1s[50])


coefs = poly.polyfit(alphas, g2_0, 2)
ffit = poly.polyval(alphas, coefs)

fig = plt.figure()
ax = plt.axes(projection='3d')

contour_height = np.ones(np.shape(g2_0)) * 0.01

ax.set_xlabel(r'$Re(\alpha - \alpha_0)$')
ax.set_ylabel(r'$Im(\alpha - \alpha_0)$')
ax.set_zlabel(r'$g^{2}(0)$')
# ax.plot_surface(X, Y, g2_0)
ax.contour(X, Y, g2_0, [0.01], lw=3, linestyles='solid')

plt.show()


'''

def plot_alpha_err_vs_g2_0():
    alpha_err_re = np.linspace(-0.05, 0.05, 25)[1:]
    alpha_err_im = np.linspace(-0.05, 0.05, 25)[1:]
    
    x, y = np.meshgrid(alpha_err_re, alpha_err_im)
    
    # alpha_err_re = np.load(os.path.join('data', '1 alpha_err_re.npy'))
    # alpha_err_im = np.load(os.path.join('data', '1 alpha_err_im.npy'))
    
    g2_0 = np.load(os.path.join('data', '1 alpha3, init_err_g2_0.npy'))[1:, 1:]
    print(np.shape(g2_0))
    
    fig, ax = plt.subplots()
    
    im = ax.pcolormesh(x, y, np.real(g2_0), cmap='hot')
    
    cbar = plt.colorbar(im)
    cbar.set_label('g2_0')
    
    ax.set_xlabel('alpha_err_re')
    ax.set_ylabel('alpha_err_im')
    
    plt.show()

def old_plot_alpha_err_vs_g2_0():
    
    alpha_err_re = np.load(os.path.join('data', '1 alpha_err_re.npy'))
    alpha_err_im = np.load(os.path.join('data', '1 alpha_err_im.npy'))
    
    g2_0 = np.load(os.path.join('data', '1 g2_0 t10.npy'))
    print(np.shape(g2_0))
    
    fig, ax = plt.subplots()
    
    im = ax.pcolormesh(alpha_err_re, alpha_err_im, np.real(g2_0), cmap='hot')
    
    cbar = plt.colorbar(im)
    cbar.set_label('g2_0')
    
    ax.set_xlabel('alpha_err_re')
    ax.set_ylabel('alpha_err_im')
    
    plt.show()

def plot_U_vs_power_vs_photon_num():
    
    lambda_3s = np.load(os.path.join('data', '1 lambda3_n_vs_lambda3.npy'))
    photon_maxs = np.load(os.path.join('data', '1 photon_maxs_n_vs_lambda3.npy'))
    Us = np.logspace(-3, -1, 100, base=10.0) * 4.4 * kappa
    h_bar_omega = 1.27e-19
    
    # print(photon_maxs)
    
    # lambda_3s = np.array([0.1, 0.33, 0.5, 1, 2])
    opacities = [0.25, 0.4, 0.55, 0.85, 1]
    
    P = np.zeros([lambda_3s.size, Us.size])
    
    for index in range(lambda_3s.size):
        det_kap_ratio = (np.abs(lambda_3s[index]) ** 2) / Us

        lambda_1s = lambda_3s[index] * (-1 + (np.absolute(lambda_3s[index]) ** 2) / (2 * (Us ** 2)) + 1j * kappa / (4 * Us))
        
        # P = (np.abs(lambda_1s) ** 2) * h_bar_omega * (det_kap_ratio ** 2) * 1e+8
        P[index] = (np.abs(lambda_1s) ** 2) * h_bar_omega * (det_kap_ratio ** 2) * 1e+8 # maybe
    
        # plt.plot((0.044 * 1e-2) / Us, P, color='g', alpha=opacities[index])
    
    print(P[0][0])
    
    x, y = np.meshgrid((0.044 * 1e-2) / Us, photon_maxs)
    fig, ax = plt.subplots()
    
    im = ax.pcolormesh(x, y, P, cmap='hot', norm='log')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    cbar = plt.colorbar(im)
    cbar.set_label('lambda_1 (W)')
    
    ax.set_xlabel('V (um^3)')
    ax.set_ylabel('max(<n>)')
    
    '''
    plt.xlabel('V (um^3)')
    plt.ylabel('lambda_1 (W)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['6x10^-3', '0.065', '0.144', '0.46', '0.75']
    '''

    plt.show()

def plot_lambda3_vs_n():
    lambda_3s = np.linspace(0.01 , 1, 100)
    
    return


def plot_g2_0_ramp():
    time_consts = np.load(os.path.join('data', '1 ramp_lambda1_times.npy'))
    g2_0s = np.load(os.path.join('data', '1 ramp_lambda1_g2_0s.npy'))

    fig = plt.figure()
    ax = plt.gca()
    ax.set_ylim([0, 1])

    plt.plot(time_consts, np.real(g2_0s))

    plt.xlabel('ramp_times')
    plt.ylabel('g2_0_ramp')

    plt.show()




def plot_g2_0_long():
    time_consts = np.load(os.path.join('data', '1 time_const_noramp_findlambda2.npy'))
    g2_0_long = np.load(os.path.join('data', '1 g2_0_noramp_findlambda2.npy'))

    fig = plt.figure()

    plt.plot(time_consts * 10, np.real(g2_0_long))

    plt.xlabel('ramp_times (ns)')
    plt.ylabel('g2_0_short')

    plt.show()


def plot_g2_0_short():
    time_consts = np.load(os.path.join('data', '1 time_const_noramp_findlambda2.npy'))
    g2_0_short = np.load(os.path.join('data', '1short g2_0_noramp_findlambda2.npy'))

    fig = plt.figure()

    plt.plot(time_consts * 10, np.real(g2_0_short))

    plt.xlabel('ramp_times (ns)')
    plt.ylabel('g2_0_short')

    plt.show()


plot_alpha_err_vs_g2_0()