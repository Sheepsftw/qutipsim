import numpy as np
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy.polynomial.polynomial as poly

import os

from qutip import *

kappa = 1


def calculate_lambda_2_power():
    alpha = 2
    U = 0.0364
    Ukappa = 4.4e+6
    lambda_3 = 2 * U * alpha
    kappa = 1.21e+8
    delta = (np.abs(lambda_3) ** 2) / U
    hbar_omega = 1.27e-19 
    # omega_a? omega_b?
    hbar = 1.05e-34
    
    lambda_2 = (lambda_3 ** 2) / (4 * U)
    print(lambda_2)
    
    beta = 1e-3
    
    P = lambda_2 / ((2 * Ukappa) * (1 / (((kappa / 2) - delta) ** 2)) * (1 / (hbar_omega))) / beta
    print(P)
    
    
def calculate_lambda_1_power():
    alpha = 2
    kappa = 1.21e+8
    # lambda_1 = 200 * kappa
    
    lambda_3s = np.load(os.path.join('data', '1 lambda3_n_vs_lambda3.npy'))
    
    
    
    # U = 4.4e+6
    U = 0.0364
    hbar_omega = 1.27e-19
    lambda_3 = 2 * U * alpha 
    
    lambda_1 = lambda_3 * (-1 + (lambda_3 ** 2)/ (2 * (U ** 2) + 1j /(4 * U)))
    print(lambda_1)
    
    # lambda_1 = 100 * alpha
    
    det_kap_ratio =  4 * (alpha ** 2) * 0.0364
    print(det_kap_ratio)
    P0 = hbar_omega * (lambda_1 ** 2) * kappa
    P = P0 * (det_kap_ratio ** 2)
    print(P)
    

# Figure 2 plot
def plot_U_vs_power_vs_photon_num():
    
    fontsize = 20
    legendsize = 14
    
    kappa = 1
    
    lambda_3s = np.load(os.path.join('data', 'fig 2', '1 lambda3_n_vs_lambda3.npy'))
    photon_maxs = np.load(os.path.join('data', 'fig 2', '1 photon_maxs_n_vs_lambda3.npy'))
    Us = np.logspace(-3, -1, 100, base=10.0) * 3.64 * kappa 
    h_bar_omega = 1.27e-19
    
    
    # lambda_3s = np.array([0.1, 0.33, 0.5, 1, 2])
    opacities = [0.25, 0.4, 0.55, 0.85, 1]
    
    PQ = np.zeros([lambda_3s.size, Us.size])
    PV = np.zeros([lambda_3s.size, Us.size])
    
    for index in range(lambda_3s.size):
        det_kap_ratio = (np.abs(lambda_3s[index]) ** 2) / Us

        lambda_1s = lambda_3s[index] * (-1 + (np.absolute(lambda_3s[index]) ** 2) / (2 * (Us ** 2)) + 1j * kappa / (4 * Us))
        
        Qfac  = 10 * Us / 0.364
        
        # P = (np.abs(lambda_1s) ** 2) * h_bar_omega * (det_kap_ratio ** 2) * 1e+8
        PQ[index] = (np.abs(lambda_1s) ** 2) * h_bar_omega * (det_kap_ratio ** 2) * 1e+8 / Qfac
        
        PV[index] = (np.abs(lambda_1s) ** 2) * h_bar_omega * (det_kap_ratio ** 2) * 1e+8 # maybe
    
        # plt.plot((0.044 * 1e-2) / Us, P, color='g', alpha=opacities[index])
    
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12.8, 9.6)
    
    xV, yV = np.meshgrid((0.0364 * 1e-2) / Us, photon_maxs)
    
    
    im = ax[0][0].pcolormesh(xV, yV, PV, cmap='hot', norm='log')
    ax[0][0].set_xscale('log')
    ax[0][0].set_yscale('log')
    
    ax[0][0].axvline(x=1e-2, color='b', linestyle='dashed')
    ax[0][0].axhline(y=0.013, color='b', linestyle='dashed')
    
    cbar = plt.colorbar(im, ax=ax[0][0])
    cbar.set_ticks([1e-27, 1e-19, 1e-11, 1e-3, 1e+5])
    cbar.set_ticklabels([r'$10^{-27}$', r'$10^{-19}$', r'$10^{-11}$', r'$10^{-3}$', r'$10^{5}$'], fontsize=fontsize)
        
    cbar.set_label(r'$P$ (W)', fontsize=fontsize)
    
    ax[0][0].set_xlabel(r'$V_{eff} \,\, (\mu \mathrm{m}^3$)', fontsize=fontsize)
    ax[0][0].set_ylabel(r'$\langle n \rangle$', fontsize=fontsize)
    
    ax[0][0].tick_params(axis='both', which='major', labelsize=fontsize)
    
    
    xQ, yQ = np.meshgrid(1e+8 * Us / 0.364, photon_maxs)
    
    im = ax[1][0].pcolormesh(xQ, yQ, PQ, cmap='hot', norm='log')
    ax[1][0].set_xscale('log')
    ax[1][0].set_yscale('log')
    
    cbar = plt.colorbar(im, ax=ax[1][0])
    cbar.set_label(r'$P$ (W)', fontsize=fontsize)
    cbar.set_ticks([1e-27, 1e-19, 1e-11, 1e-3, 1e+5])
    cbar.set_ticklabels([r'$10^{-27}$', r'$10^{-19}$', r'$10^{-11}$', r'$10^{-3}$', r'$10^{5}$'], fontsize=fontsize)
    
    ax[1][0].set_xlabel(r'Q', fontsize=fontsize)
    ax[1][0].set_ylabel(r'$\langle n \rangle$', fontsize=fontsize)
    
    ax[1][0].tick_params(axis='both', which='major', labelsize=fontsize)
    
    ax[1][0].axvline(x=1e+7, color='b', linestyle='dashed')
    ax[1][0].axhline(y=0.013, color='b', linestyle='dashed')
    
    indices_to_choose = [49, 59, 69, 79, 89]
    
    ind_photon_nums = np.round(np.array([photon_maxs[i] for i in indices_to_choose]), 3)
    
    for i in range(5):
        index = indices_to_choose[i]
        det_kap_ratio = (np.abs(lambda_3s[index]) ** 2) / Us

        lambda_1s = lambda_3s[index] * (-1 + (np.absolute(lambda_3s[index]) ** 2) / (2 * (Us ** 2)) + 1j * kappa / (4 * Us))
        
        Qfac  = 10 * Us / 0.364
        PQ_curr = (np.abs(lambda_1s) ** 2) * h_bar_omega * (det_kap_ratio ** 2) * 1e+8 / Qfac

        PV_curr = (np.abs(lambda_1s) ** 2) * h_bar_omega * (det_kap_ratio ** 2) * 1e+8 # maybe
        ax[0][1].plot((0.044 * 1e-2) / Us, PV_curr, color='g', alpha=opacities[i])
        ax[1][1].plot(1e+8 * Us / 0.364, PQ_curr, color='g', alpha=opacities[i])
    
    ax[0][1].set_xlabel(r'$V_{eff} \,\, (\mu \mathrm{m}^3$)', fontsize=fontsize)
    #ax[0][1].set_ylabel(r'$\Lambda_1$ (W)')
    
    ax[0][1].set_xscale('log')
    ax[0][1].set_yscale('log')
    ax[0][1].legend(ind_photon_nums, title=r'$\langle n \rangle$', frameon=False, fontsize=legendsize)
    
    ax[0][1].tick_params(axis='both', which='major', labelsize=fontsize)
    
    ax[0][1].set_yticks([1e-19, 1e-13, 1e-7, 1e-1, 1e+5])
    ax[0][1].set_yticklabels([r'$10^{-19}$', r'$10^{-13}$', r'$10^{-7}$', r'$10^{-1}$', r'$10^{5}$'])
    
    ax[1][1].set_xlabel(r'Q', fontsize=fontsize)
    
    # ax[1][1].set_ylabel(r'$\Lambda_1$ (W)')
    
    ax[1][1].set_xscale('log')
    ax[1][1].set_yscale('log')
    ax[1][1].legend(ind_photon_nums, title=r'$\langle n \rangle$', frameon=False, fontsize=legendsize)
    
    ax[1][1].set_yticks([1e-19, 1e-13, 1e-7, 1e-1, 1e+5])
    ax[1][1].set_yticklabels([r'$10^{-19}$', r'$10^{-13}$', r'$10^{-7}$', r'$10^{-1}$', r'$10^{5}$'])
    ax[1][1].tick_params(axis='both', which='major', labelsize=fontsize)
    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'Q and V vs n vs Lambda_1 fontsize.png'), bbox_inches='tight')
    plt.show()
    

# Figure 3 plot
def combined_figure_3():
        
    state1 = np.load(os.path.join('data', 'fig 3', 'squeezed state U=0.npy'))
    state2 = np.load(os.path.join('data', 'fig 3', 'squeezed state U=0.2.npy'))
    state3 = np.load(os.path.join('data', 'fig 3', 'squeezed state with l1l2 U=0.2 t=0.4.npy'))
    
    xvec = np.linspace(-5, 5, 100)
    
    fig, ax = plt.subplots(2,3)
    fig.set_size_inches(19.2, 9.6)
    
    ax[0][0].contourf(xvec, xvec, np.abs(state1), 100)
    
    ax[0][0].set_xlabel(r'x')
    ax[0][0].set_ylabel(r'p')
    
    ax[0][1].contourf(xvec, xvec, np.abs(state2), 100)
    
    ax[0][1].set_xlabel(r'x')
    ax[0][1].set_ylabel(r'p')
    
    ax[0][2].contourf(xvec, xvec, np.abs(state3), 100)
    ax[0][2].set_xlabel(r'x')
    ax[0][2].set_ylabel(r'p')
    
    
    alpha_err_re = np.linspace(-0.025, 0.025, 51)
    alpha_err_im = np.linspace(-0.025, 0.025, 51)
    alpha_x, alpha_y = np.meshgrid(alpha_err_re, alpha_err_im)
    
    g2_0 = np.load(os.path.join('data', 'fig 3', '2 rel_alpha_err g2_0 alpha=2 t=1.npy'))
    
    fig.set_size_inches(19.2, 7.2)
    
    
    im = ax[1][0].pcolormesh(alpha_x, alpha_y, np.real(g2_0), cmap='hot')
    
    cbar = plt.colorbar(im)
    cbar.set_label(r'$g^{(2)}(0)$')
    
    ax[1][0].set_xlabel(r'$Re(\Delta\alpha)$')
    ax[1][0].set_ylabel(r'$Im(\Delta\alpha)$')
    
    
    alphas = np.linspace(0.1, 2, 20)
    g2_0_2 = np.load(os.path.join('data', 'fig 3', '0.025 alphavsg2 relative, init_err_g2_0_short.npy'))
    errs_alphas = np.load(os.path.join('data', 'fig 3', '0.025 alphavserr relative, init_err_g2_0_short.npy'))

    ln1_1 = ax[1][1].plot(alphas, g2_0_2, color='g', label=r'$g^{(2)}(0)$')
    ax[1][1].set_xlabel(r'$\alpha$')
    ax[1][1].set_ylabel(r'$g^{(2)}(0)$')
    
    ax1_2 = ax[1][1].twinx()
    ln1_2 = ax1_2.plot(alphas, errs_alphas, label='Loss')
    ax1_2.set_ylabel(r'$\mathrm{Loss} (\mathrm{a.u.})$')
    ax1_2.set_ylim([-1, 9])
    
    tot1 = ln1_1 + ln1_2
    labels = [l.get_label() for l in tot1]
    
    ax[1][1].legend(tot1, labels)
    
    
    time_consts = np.load(os.path.join('data', 'fig 3', '1 time_const_noramp_findlambda2.npy'))
    g2_0_short = np.load(os.path.join('data', 'fig 3', '1 short g2_0_noramp_findlambda2.npy'))

    errs = np.load(os.path.join('data', 'fig 3', '2 coherence_err no_ramp_find_lambda2.npy'))

    ln2_1 = ax[1][2].plot(time_consts * 10, np.real(g2_0_short), color='g', label=r'$g^{(2)}(0)$')
    ax2_2 = ax[1][2].twinx()
    ln2_2 = ax2_2.plot(time_consts * 10, np.real(errs), label='Loss')
    
    tot2 = ln2_2 + ln2_1
    labels = [l.get_label() for l in tot2]

    ax[1][2].set_xlabel(r'$\tau (\mathrm{ns})$')
    ax[1][2].set_ylabel(r'$g^{(2)}(0)$')
    ax2_2.set_ylabel(r'$\mathrm{Loss} (\mathrm{a.u.})$')
    
    ax2_2.set_ylim([-0.001, 0.013])
    
    ax[1][2].legend(tot2, labels)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join('images', 'combined fig 3.png'), bbox_inches='tight')
    plt.show()

    
# Figure 4 plot
def plot_combined_error():
    
    fontsize = 16
    
    lambda_1 = (0.6212450111263796+5.494112962929153j)
    lambda_2 = (-0.11416742671335132+0.005940273551506413j)
    
    # relative error
    errs_re_1 = np.linspace(-0.2, 0.2, 51) / np.abs(lambda_1)
    errs_im_1 = np.linspace(-0.2, 0.2, 51) / np.abs(lambda_1)
    g2_0s_1 = np.load(os.path.join('data', 'fig 4', '3 t=0.01 maxerr=0.2 re_im g2_0_lambda1_error.npy'))
    
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(6.4, 4.8)
    
    x1, y1 = np.meshgrid(errs_re_1, errs_im_1)
    im1 = ax[0].pcolormesh(x1, y1, g2_0s_1, cmap='hot', norm='log')
    cbar = plt.colorbar(im1, ax=ax[0])
    cbar.set_label(r'$g^{(2)}(0)$', fontsize=fontsize)
    
    cbar.set_ticks([1, 1e-1, 1e-2, 1e-3, 1e-4])
    cbar.set_ticklabels([r'$10^{-0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$'], fontsize=fontsize)
    
    ax[0].set_xlabel(r'$Re(\Delta\Lambda_1)$', fontsize=fontsize)
    ax[0].set_ylabel(r'$Im(\Delta\Lambda_1)$', fontsize=fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
    
    
    # relative error
    errs_re_2 = np.linspace(-0.06, 0.06, 51) / np.abs(lambda_2)
    errs_im_2 = np.linspace(-0.06, 0.06, 51) / np.abs(lambda_2)
    g2_0s_2 = np.load(os.path.join('data', 'fig 4', '2 t=0.01 maxerr=0.06 re_im g2_0_lambda2_error.npy'))
    
    x2, y2 = np.meshgrid(errs_re_2, errs_im_2)
    im2 = ax[1].pcolormesh(x2, y2, g2_0s_2, cmap='hot', norm='log')
    cbar = plt.colorbar(im2, ax=ax[1])
    cbar.set_label(r'$g^{(2)}(0)$', fontsize=fontsize)
    cbar.set_ticks([1, 1e-1, 1e-2, 1e-3])
    cbar.set_ticklabels([r'$10^{-0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$'], fontsize=fontsize)
    
    ax[1].set_xlabel(r'$Re(\Delta\Lambda_2)$', fontsize=fontsize)
    ax[1].set_ylabel(r'$Im(\Delta\Lambda_2)$', fontsize=fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
    
    
    lambda_1_evo = 1.0192 + 1j
    lambda_2_evo = -0.1456
    
    amp_errs = np.linspace(0, 0.5, 100) / np.abs(lambda_1_evo)
    amp2_err = np.linspace(0, 0.005, 100) / np.abs(lambda_2_evo)
    g2_0s_3 = np.load(os.path.join('data', 'fig 4', 't=1-kappa g2_0 lambda1andlambda2err.npy'))
    
    ax[2].set_xlabel(r'$\Delta\Lambda_1$', fontsize=fontsize)
    ax[2].set_ylabel(r'$\Delta\Lambda_2$', fontsize=fontsize)
    ax[2].tick_params(axis='both', which='major', labelsize=fontsize)

    x2, y2 = np.meshgrid(amp_errs, amp2_err)

    im2 = ax[2].pcolormesh(x2, y2, g2_0s_3, cmap='hot', norm='log')
    cbar = plt.colorbar(im2, ax=ax[2])
    cbar.set_label(r'$g^{(2)}(0)$', fontsize=fontsize)
    cbar.set_ticks([1, 1e-1, 1e-2, 1e-3, 1e-4])
    cbar.set_ticklabels([r'$10^{-0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$'], fontsize=fontsize)
    
    fig.set_size_inches(19.2, 4.8)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'combined_errs fontsize.png'), bbox_inches='tight')
    
    plt.show()
    

    

# plot_U_vs_power_vs_photon_num()
# calculate_lambda_2_power()
# calculate_lambda_1_power()
plot_combined_error()
# combined_figure_3()










# Below are unused


def plot_alpha_err_vs_g2_0():
    
    alpha_err_re = np.load(os.path.join('data', '1 alpha_err_re.npy'))
    alpha_err_im = np.load(os.path.join('data', '1 alpha_err_im.npy'))
    
    g2_0 = np.load(os.path.join('data', '1 g2_0 t10.npy'))
    
    fig = plt.figure()
    
    ax0 = fig.add_subplot(2, 1, 1)
    fig.set_size_inches(6.4, 7.2)
    
    im = ax0.pcolormesh(alpha_err_re, alpha_err_im, np.real(g2_0), cmap='hot')
    
    cbar = plt.colorbar(im)
    cbar.set_label(r'$g^{(2)}(0)$')
    
    ax0.set_xlabel(r'$Re(\Delta\alpha)$')
    ax0.set_ylabel(r'$Im(\Delta\alpha)$')
    
    alphas = np.linspace(0.1, 2, 20)
    g2_0_2 = np.load(os.path.join('data', '0.025 alphavsg2 relative, init_err_g2_0_short.npy'))
    errs_alphas = np.load(os.path.join('data', '0.025 alphavserr relative, init_err_g2_0_short.npy'))
    
    ax1 = fig.add_subplot(2, 2, 3)

    ln1_1 = ax1.plot(alphas, g2_0_2, color='g', label=r'$g^{(2)}(0)$')
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$g^{(2)}(0)$')
    
    ax1_2 = ax1.twinx()
    ln1_2 = ax1_2.plot(alphas, errs_alphas, label='Loss')
    ax1_2.set_ylabel(r'$\mathrm{Loss} (\mathrm{a.u.})$')
    ax1_2.set_ylim([-1, 9])
    
    tot1 = ln1_1 + ln1_2
    labels = [l.get_label() for l in tot1]
    
    ax1.legend(tot1, labels)
    
    ax2 = fig.add_subplot(2, 2, 4)
    
    time_consts = np.load(os.path.join('data', '1 time_const_noramp_findlambda2.npy'))
    g2_0_short = np.load(os.path.join('data', '1short g2_0_noramp_findlambda2.npy'))

    errs = np.load(os.path.join('data', '2 coherence_err no_ramp_find_lambda2.npy'))

    ln2_1 = ax2.plot(time_consts * 10, np.real(g2_0_short), color='g', label=r'$g^{(2)}(0)$')
    ax2_2 = ax2.twinx()
    ln2_2 = ax2_2.plot(time_consts * 10, np.real(errs), label='Loss')
    
    tot2 = ln2_2 + ln2_1
    labels = [l.get_label() for l in tot2]

    ax2.set_xlabel(r'$\tau (\mathrm{ns})$')
    ax2.set_ylabel(r'$g^{(2)}(0)$')
    ax2_2.set_ylabel(r'$\mathrm{Loss} (\mathrm{a.u.})$')
    
    ax2_2.set_ylim([-0.001, 0.013])
    
    ax2.legend(tot2, labels)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join('images', 'alpha_err vs g2_0.png'), bbox_inches='tight')
    plt.show()


def plot_g2_0_short():
    time_consts = np.load(os.path.join('data', '1 time_const_noramp_findlambda2.npy'))
    g2_0_short = np.load(os.path.join('data', '1short g2_0_noramp_findlambda2.npy'))

    errs = np.load(os.path.join('data', '2 coherence_err no_ramp_find_lambda2.npy'))
    g2_0_long_2 = np.load(os.path.join('data', '2 g2_0_noramp_findlambda2.npy'))

    fig, ax = plt.subplots()

    ln1 = ax.plot(time_consts * 10, np.real(g2_0_short), color='g', label=r'$g^{(2)}(0)$')
    ax2 = ax.twinx()
    ln2 = ax2.plot(time_consts * 10, np.real(errs), label='Loss')
    
    tot = ln2 + ln1
    labels = [l.get_label() for l in tot]

    plt.xlabel(r'$\tau (\mathrm{ns})$')
    ax.set_ylabel(r'$g^{(2)}(0)$')
    ax2.set_ylabel(r'$\mathrm{Loss} (\mathrm{a.u.}$')
    
    ax2.set_ylim([-0.001, 0.013])
    
    ax.legend(tot, labels)
    
    plt.show()
    
    
def plot_alpha_err_percent():
    
    alphas = np.linspace(0.1, 2, 20)
    g2_0 = np.load(os.path.join('data', '1 alphavsg2 relative, init_err_g2_0_short.npy'))
    
    fig = plt.figure()

    plt.plot(alphas, g2_0)

    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$g^{(2)}(0)$')

    plt.show()


def plot_alpha_growth_vs_g2_0_growth():
    alphas = np.linspace(0.1, 2, 20)
    g2_0s = np.load(os.path.join('data', '1 alphavsg2, init_err_g2_0_short.npy'))
    
    fig, ax = plt.subplots()
    
    plt.plot(alphas, g2_0s)
    
    ax.set_xlabel('alpha')
    ax.set_ylabel('g2_0')
    
    plt.show()


def unused_plot_alpha_err_vs_g2_0():
    alpha_err_re = np.linspace(-0.1, 0.1, 51)
    alpha_err_im = np.linspace(-0.1, 0.1, 51)
    
    x, y = np.meshgrid(alpha_err_re, alpha_err_im)
    
    # alpha_err_re = np.load(os.path.join('data', '1 alpha_err_re.npy'))
    # alpha_err_im = np.load(os.path.join('data', '1 alpha_err_im.npy'))
    
    g2_0_alpha3 = np.load(os.path.join('data', '1 alpha5, init_err_g2_0_short.npy'))
    
    fig, ax = plt.subplots()
    
    im = ax.pcolormesh(x, y, np.real(g2_0_alpha3), cmap='hot')
    
    cbar = plt.colorbar(im)
    cbar.set_label('g2_0')
    
    ax.set_xlabel(r'$Re(\delta\alpha)$')
    ax.set_ylabel('alpha_err_im')
    
    plt.show()
    

def squeezed_state():
    state1 = np.load(os.path.join('data', 'squeezed state U=0.npy'))
    state2 = np.load(os.path.join('data', 'squeezed state U=0.2.npy'))
    state3 = np.load(os.path.join('data', 'squeezed state with l1l2 U=0.2 t=0.4.npy'))
    
    xvec = np.linspace(-5, 5, 100)
    
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(19.2, 4.8)
    
    ax[0].contourf(xvec, xvec, np.abs(state1), 100)
    
    ax[0].set_xlabel(r'x')
    ax[0].set_ylabel(r'p')
    
    ax[1].contourf(xvec, xvec, np.abs(state2), 100)
    
    ax[1].set_xlabel(r'x')
    ax[1].set_ylabel(r'p')
    
    ax[2].contourf(xvec, xvec, np.abs(state3), 100)
    ax[2].set_xlabel(r'x')
    ax[2].set_ylabel(r'p')
    
    
    plt.show()


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
    errs = np.load(os.path.join('data', '2 coherence_err no_ramp_find_lambda2.npy'))
    g2_0_long_2 = np.load(os.path.join('data', '2 g2_0_noramp_findlambda2.npy'))

    fig, ax = plt.subplots()

    ln1 = ax.plot(time_consts * 10, np.real(g2_0_long), color='g', label=r'$g^{(2)}(0)$')
    ax2 = ax.twinx()
    ln2 = ax2.plot(time_consts * 10, errs, label='Loss')
    
    tot = ln1 + ln2
    labels = [l.get_label() for l in tot]

    plt.xlabel(r'$\tau (\mathrm{ns})$')
    ax.set_ylabel(r'$g^{(2)}(0)$')
    ax2.set_ylabel(r'Loss')
    
    ax.legend(tot, labels)

    plt.show()


def plot_lambda2_error():
    # Lambda2 = -0.114 + 0.006j
    errs_re = np.linspace(-0.06, 0.06, 51)
    errs_im = np.linspace(-0.06, 0.06, 51)
    g2_0s = np.load(os.path.join('data', '2 t=0.01 maxerr=0.06 re_im g2_0_lambda2_error.npy'))
    
    errs2 = np.linspace(0, 0.06, 50)
    g2_0s2 = np.load(os.path.join('data', 't=0.01 maxerr=0.06 g2_0_lambda2_error.npy'))
    
    x, y = np.meshgrid(errs_re, errs_im)
    
    fig, ax = plt.subplots()
    
    im = ax.pcolormesh(x, y, g2_0s, cmap='hot', norm='log')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$g^{(2)}(0)$')
    
    ax.set_xlabel(r'$Re(\Delta\Lambda_2)$')
    ax.set_ylabel(r'$Im(\Delta\Lambda_2)$')
    
    
    # ax.plot(errs2, g2_0s2)
    
    # ax.set_xlabel(r'$\Delta\Lambda_2$')
    # ax.set_ylabel(r'$g^{(2)}(0)$')
    
    plt.show()


def plot_lambda1_error():
        # Lambda2 = -0.114 + 0.006j
    errs_re = np.linspace(-0.2, 0.2, 51)
    errs_im = np.linspace(-0.2, 0.2, 51)
    g2_0s = np.load(os.path.join('data', '3 t=0.01 maxerr=0.2 re_im g2_0_lambda1_error.npy'))
    
    x, y = np.meshgrid(errs_re, errs_im)
    
    fig, ax = plt.subplots()
    
    im = ax.pcolormesh(x, y, g2_0s, cmap='hot', norm='log')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$g^{(2)}(0)$')
    
    ax.set_xlabel(r'$Re(\Delta\Lambda_2)$')
    ax.set_ylabel(r'$Im(\Delta\Lambda_2)$')
    
    plt.show()
    
    
def plot_lambda_1_and_2_err():
    
    amp_errs = np.linspace(0, 0.5, 100)
    amp2_err= np.linspace(0, 0.005, 100)
    g2_0 = np.load(os.path.join('data', 't=1-kappa g2_0 lambda1andlambda2err.npy'))
    
    fig, ax = plt.subplots()

    ax.set_xlabel(r'$\Delta\Lambda_1$')
    ax.set_ylabel(r'$\Delta\Lambda_2$')

    x, y = np.meshgrid(amp_errs, amp2_err)

    im = ax.pcolormesh(x, y, g2_0, cmap='hot', norm='log')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$g^{(2)}(0)$')
    
    plt.show()
    
    
def test_data_load():
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