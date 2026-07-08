import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from Floquet_perturbation_theory import *
from helper_function import *
from utils import *
from scipy.optimize import root

def main():
    s = Simulation()
    N_t = 100
    tg = 250
    tlist = np.linspace(0, tg, N_t)
    sigma = 0.15

    # compute cos params
    cos_amp, cos_wd = compute_cos_params(s, tg)
    heff_params = s.heff(cos_wd, cos_amp)
    Delta = heff_params[2][2] - heff_params[0][0]
    # rrr follows from the heff response to amplitude, as both Omega_ab and Omega_bc are linear rrr is constant for all amp- and wd values
    rrr = heff_params[1][2] / heff_params[0][1] # compute the RabiRateRatio (RRR) from the two rabirates in cos dirve case
    rrr, Delta

    t = sp.symbols("t", real=True)
    popt_real = [ 1.59629893e+00, 1.25000077e+02, 3.74433146e+01, -5.60979110e-03]
    popt_imag = [ 4.00236190e-02, 1.24975571e+02,  4.48865078e+01, -2.19686977e-02, 3.75630992e-03,  5.80637185e-05]
    popt_freq = [2.10515838e-02, 1.25000139e+02, 2.66358933e+01, 4.43802950e+00]
    A, t0, sigma, c = popt_real
    amp_sym_real = (c + A*sp.exp(-(t-t0)**2/(2*sigma**2)))
    A, t0, sigma, omega, phi, c = popt_imag
    amp_sym = amp_sym_real + 1j*(c + A * sp.sin(omega*(t-t0)+phi) * sp.exp(-(t-t0)**2/(2*sigma**2)))
    A, t0, sigma, c = popt_freq
    freq_sym = c + A*sp.exp(-(t-t0)**2/(2*sigma**2))
    freq_func = sp.lambdify(t, freq_sym, "numpy")
    amp_func = sp.lambdify(t, amp_sym, "numpy")


    initial_state = np.zeros(27)
    initial_state[s.state_a] = 1
    psi_t_rHeff2_rW0_num, lUnum = Psi_t_FloquetPerturb_dynamic(5,0,
        freq_func,
        amp_func,
        s.resonances, s.E_array,s.V1_dressed_array,initial_state,tlist,num = True,)


    # each column in the unitary represents one initial state U(row;column)
    def plot_states(states, names=None):
        _, axes = plt.subplots(1,2, figsize = (14,4))
        for i, psi in enumerate(states): 
            title = '' if names is None else names[i]

            axes[i].plot(tlist, np.abs(psi[s.state_a, :]) ** 2, "--",label="|001> a (rH=2, rW=0)",)
            axes[i].plot(tlist, np.abs(psi[s.state_b, :]) ** 2, "--",label="|100> b (rH=2, rW=0)",)
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Population')
            axes[i].set_title(title)
            axes[i].legend()
            # plot the leakage population on a log-scale
            axc = axes[i].twinx()
            axc.semilogy(tlist, np.abs(psi[s.state_c, :]) ** 2, "--",label="|100> (rH=2, rW=0)",color = 'green')
            # axc.semilogy(tlist, np.abs(lU[:,2,i])**2, label='population c (log)', color='green')
            axc.legend()
            axc.plot()
    plot_states([psi_t_rHeff2_rW0_num], ['numerical'])
    plt.show()

    for i in range(len(tlist)):
        print("t: ",tlist[i],  " psit: ", psi_t_rHeff2_rW0_num[:, i])



    print("Simulation beendet")

if __name__ == "__main__":
    main()