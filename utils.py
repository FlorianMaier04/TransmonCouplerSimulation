import sys
import numpy as np
import sympy as sp
from qutip import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from Floquet_perturbation_theory import *
from helper_function import *
from IPython.display import display
from scipy.linalg import logm
from scipy.optimize import fsolve
import scipy.special


class Simulation:
    @property 
    def gt(self):
        return self._gt

    @gt.setter
    def gt(self, value):
        self._gt = value
        self.setup_modulation(self.modulation_mode, self.modulation_args)

    def __init__(self, dim_q1=3, dim_q2=3, dim_c=3,
                 w1=3.83, w2=3.11, wc=4.29,
                 alpha1=-0.205, alpha2=-0.216, alphac=-0.161,
                 g1c=0.115, g2c=0.110, g12=0.015):
        """        
        Args:
            dim_q1, dim_q2, dim_c: Hilbert space dimensions
            w1, w2, wc: Frequencies (in GHz, will be converted to rad/ns)
            alpha1, alpha2, alphac: Anharmonicities (in GHz, will be converted to rad/ns)
            g1c, g2c, g12: Coupling strengths (in GHz, will be converted to rad/ns)
            amp_sweep: Array of amplitudes to sweep (default: 0.001 to 0.05, 10 points)
            fre_sweep: Array of frequency detunings to sweep (default: -0.1 to 0.1, 10 points)
        """
        self.dim_q1, self.dim_q2, self.dim_c = dim_q1, dim_q2, dim_c
        self.d = self.dim_q1 * self.dim_c * self.dim_q2
        self.state_b = dim_q2 * dim_c  # state |100>
        self.state_a = 1  # state |001>
        self.state_c = dim_q2  # state |010>
        self.state_d = 2 # state |002> # this state is at same energy as state_a at the resonant frequency
        self.state_e = 1*(dim_c*dim_q2)+1*(dim_q2)+1 # |111>
        self.order = 3
        self.resonances = {self.state_a: 0, self.state_b: 1, self.state_c: 2}  # E_state_a - wd = E_state_b
        self.sd = len(self.resonances)
        # Convert frequencies and couplings to rad/ns frequencies are given in GHz
        self.w1_num = w1 * 2 * np.pi
        self.alpha1_num = alpha1 * 2 * np.pi
        self.w2_num = w2 * 2 * np.pi
        self.alpha2_num = alpha2 * 2 * np.pi
        self.wc_num = wc * 2 * np.pi
        self.alphac_num = alphac * 2 * np.pi
        self.g1c_num = g1c * 2 * np.pi
        self.g2c_num = g2c * 2 * np.pi
        self.g12_num = g12 * 2 * np.pi
        self._setup_operators()
        self._setup_hamiltonian()

    def _setup_operators(self):
        """Initialize creation/annihilation operators."""
        self.a_q1 = tensor(destroy(self.dim_q1), qeye(self.dim_c), qeye(self.dim_q2))
        self.a_q2 = tensor(qeye(self.dim_q1), qeye(self.dim_c), destroy(self.dim_q2))
        self.a_qc = tensor(qeye(self.dim_q1), destroy(self.dim_c), qeye(self.dim_q2))
    def _setup_hamiltonian(self):
        """Setup Hamiltonian and transform to dressed basis."""
        Hq1 = self.w1_num * self.a_q1.dag() @ self.a_q1 + \
              (self.alpha1_num / 2) * self.a_q1.dag() @ self.a_q1.dag() @ self.a_q1 @ self.a_q1
        Hq2 = self.w2_num * self.a_q2.dag() @ self.a_q2 + \
              (self.alpha2_num / 2) * self.a_q2.dag() @ self.a_q2.dag() @ self.a_q2 @ self.a_q2
        Hc = self.wc_num * self.a_qc.dag() @ self.a_qc + \
             (self.alphac_num / 2) * self.a_qc.dag() @ self.a_qc.dag() @ self.a_qc @ self.a_qc
        V0 = (self.g1c_num * ((self.a_q1 - self.a_q1.dag()) @ (self.a_qc - self.a_qc.dag())) +
              self.g2c_num * ((self.a_q2 - self.a_q2.dag()) @ (self.a_qc - self.a_qc.dag())) +
              self.g12_num * ((self.a_q1 - self.a_q1.dag()) @ (self.a_q2 - self.a_q2.dag())))
        H0 = Hq1 + Hq2 + Hc + V0
        V1 = self.a_qc.dag() @ self.a_qc
        # Diagonalize H0
        evals, evecs = H0.eigenstates()
        self.sorted_evals, self.sorted_evecs = SortedFRFSpectrum(evals, evecs, self.dim_q1, self.dim_c, self.dim_q2)
        self.E_array = self.sorted_evals.reshape(self.d)
        self.E_states = self.sorted_evecs.reshape(self.d)
        self.t_interaction_picture = Qobj(
            np.column_stack([
                self.sorted_evecs[i, j, k].full()
                for i in range(self.dim_q1)
                for j in range(self.dim_c)
                for k in range(self.dim_q2)
            ]),
            dims=[[self.dim_q1, self.dim_c, self.dim_q2], [self.dim_q1, self.dim_c, self.dim_q2]],
        )
        self.H0_dressed = self.t_interaction_picture.dag() @ H0 @ self.t_interaction_picture
        self.V1_dressed = self.t_interaction_picture.dag() @ V1 @ self.t_interaction_picture
        self.V1_dressed_array = self.V1_dressed.full().reshape(self.d, self.d).real
        # Resonant frequency
        self.res_freq_static = (self.sorted_evals[1, 0, 0] - self.sorted_evals[0, 0, 1]) / (2 * np.pi)

    def heff_element(self, i, j, wd, amp):
        element = Heff_Floquet_summed(self.order, i, j, wd, self.resonances, self.E_array, (amp/2 * (1+0j))*self.V1_dressed_array, V0=None)
        return element
    
    def resonant_condition(self, fre, amp, order, i, f, ):
        fre = float(np.atleast_1d(fre)[0])
        resonances = {self.state_a:0, self.state_b:1}
        delta_i = Heff_Floquet_summed(
            self.order,
            i,
            i,
            fre,
            resonances,
            self.E_array,
            amp / 2 * self.V1_dressed_array,
            V0=None,
            analytics=False,
        )
        delta_f = Heff_Floquet_summed(
            order,
            f,
            f,
            fre,
            resonances,
            self.E_array,
            amp / 2 * self.V1_dressed_array,
            V0=None,
            analytics=False,
        )
        diff = delta_f - delta_i
        return float(np.real(diff))

    def find_resonance(self, amp):
        wd_initial_guess = self.res_freq_static * 2*np.pi
        resonant_wd_solution = fsolve(self.resonant_condition, wd_initial_guess, 
            args=(amp, self.order, self.state_a, self.state_b,),)
        return resonant_wd_solution[0]

    def heff(self, wd, amp):
        heff = np.zeros((self.sd, self.sd), dtype=complex)
        states = [*self.resonances.keys()]
        for idx_a, state_a in enumerate(states):
            for idx_b, state_b in enumerate(states):
                heff[idx_a,idx_b] = self.heff_element(state_a, state_b, wd, amp)
        return heff
    
    def extract_fidelity(self, lh, tlist):
        U_ideal = np.array([[0, 1j, 0], [1j, 0, 0], [0, 0, 1]], dtype=complex)
        U_subsys = np.zeros((self.sd, self.sd), dtype=complex)
        for col_idx, _ in enumerate(self.resonances): 
            phi0 = qt.basis(self.sd, col_idx)
            result = mesolve(
                lh,
                phi0,
                tlist,
                e_ops=[],
            )
            final_state = result.states[-1]
            # Extract subsystem components to form column of unitary
            for row_idx, j_state in enumerate(self.resonances):
                U_subsys[row_idx, col_idx] = final_state.full()[row_idx, 0]
        # print_matrix(U_subsys)
        fidelity = qutip.process_fidelity(Qobj(U_subsys), Qobj(U_ideal))
        return fidelity

    def get_state_name(self, idx):
        match idx:
            case self.state_a:
                return 'a'
            case self.state_b:
                return 'b'
            case self.state_c:
                return 'c'
            case self.state_d:
                return 'd'

def sigma_x_ij(i, j, d):
    ei = basis(d, i)
    ej = basis(d, j)
    return ej*ei.dag() + ei*ej.dag()

def sigma_y_ij(i, j, d):
    ei = basis(d, i)
    ej = basis(d, j)
    return -1j*ei*ej.dag() + 1j*ej*ei.dag()

def roh(index, pos):
    val = 1 if pos else -1
    ax, ay, az  = 0,0,0
    match index:
        case 0: ax = val
        case 1: ay = val
        case 2: az = val
    return Qobj(1/2 * np.array([[1+az, ax-ay*1j,0],[ax+1j*ay, 1-az, 0], [0,0,2]]))

def compute_phi0(s, wd, amp, psi0): 
    rW = 2
    W = W_Floquet_elements(rW, wd, s.resonances, s.E_array, amp/2 * s.V1_dressed_array,
                        None, ref_state = None, analytics = False)
    # express phi0 in the resonant subspace, assuming psi0 is no superposition and lies in the res. subspace
    phi0 = qt.zero_ket(s.sd)
    projections = [None] * s.sd
    for idx, a in enumerate(s.resonances.keys()):
        dic = W[rW, psi0, a]
        n_psi0 = s.resonances[psi0]
        if n_psi0 in dic: c = np.conj(dic[n_psi0])
        else: c = 0
        # if(c!=0): print("idx: ", a, " c: ", c, " richtiger idx: ", idx)
        phi0 += c * qt.basis(s.sd, idx)
        projections[idx] = qt.basis(s.sd, idx).proj()
        # projections[idx] = c * np.conj(c) * qt.basis(simulator.sd, idx).proj()
    if psi0 == s.state_a:
        phi0 = qt.basis(s.sd, 0)
    if psi0 == s.state_b:
      phi0 = qt.basis(s.sd, 1)
    if psi0 == s.state_c:
        phi0 = qt.basis(s.sd, 2)
    # phi0 = phi0.unit()
    # print("phi0: ", phi0)
    return phi0, projections

def find_optimal_gate_time(wd, amp, s, heff):
    Omega_ab = heff[0, 1]
    tg,_,_ = find_optimal_time(wd,amp,s.order,
        s.state_b,s.state_a,s.state_c,s.E_array,s.V1_dressed_array, int(np.pi/(2*Omega_ab.real)),)
    tlist = np.arange(0, tg, 0.01)
    return tg, tlist

def compute_t_dependency(s, heff, tg, use_drag = True, use_gauss = True):
    H = np.zeros((s.sd, s.sd), dtype=complex)
    # V = [None] * (int(scipy.special.binom(s.sd, 2))*2)
    Vx = [None] * (s.sd - 1)
    Vy = [None] * (s.sd - 1)
    lambda1 = heff[0, 1]*2
    delta0 = heff[0][0]
    Delta = - 2* heff[1][1] + heff[2][2] + delta0
    epsilon_x, epsilon_y = pulse_functions_gauss(tg, 1.0*Delta, lambda1)
    if not use_drag:
        epsilon_y = lambda t,args=None: 0   
    if not use_gauss:
        epsilon_x = lambda t,args=None: lambda1 
    for i, _ in enumerate(s.resonances):
        # diagonal element
        delta = heff[i][i] - delta0
        H[i][i] = delta
        if i+1 == s.sd: continue
        lambda_i = heff[i][i+1].real * 2 / lambda1
        coeff_img = heff[i][i+1].imag
        Vx[i] = [sigma_x_ij(i, i+1, s.sd) * lambda_i * 1/2, epsilon_x]
        Vy[i] = [sigma_y_ij(i, i+1, s.sd) * lambda_i * 1/2, epsilon_y]
        # for j, _ in enumerate(s.resonances):
        #     if j==i or i>j: continue
        #     coeff_real = heff[i][j].real
        #     coeff_img = heff[i][j].imag
        #     V[(i+j-1)*2] = [sigma_x_ij(i, j, s.sd) * coeff_real, epsilon_x]
        #     V[(i+j-1)*2+1] = [sigma_y_ij(i, j, s.sd) * coeff_img, epsilon_y]
            # print("V number: ", ((i+j-1)*2))
            # display(V[(i+j-1)*2])
    return [Qobj(H), *Vx, *Vy]
    # return [Qobj(H), *Vx]

def pulse_functions_gauss(tg, Delta, lambda1):
    sigma = 1.0 * tg
    amp = tg * (np.sqrt(2*np.pi)*sigma*scipy.special.erf(tg/(2*np.sqrt(2)*sigma)) -
                np.exp(-tg**2/(8*sigma**2)) * tg)**-1
    B = np.exp(-tg**2 / (8 * sigma**2)) * amp
    epsilon_x = lambda t, args=None: lambda1 * max(0, amp * np.exp(-(t - tg / 2)**2 / (2 * sigma**2)) - B)
    dt = 1e-6
    epsilon_y = lambda t, args=None: -(
        epsilon_x(t + dt) - epsilon_x(t - dt)
    ) / (2 * dt) * 1/Delta
    # epsilon_x = lambda t, args=None: 1 if t < tg else 0 
    return epsilon_x, epsilon_y

def plot_pulse_functions(tgate, amp, tlist=None):
    if tlist is None:
        tlist = np.linspace(0, tgate, 1000)
    epsilon_x, epsilon_y = pulse_functions_gauss(tgate, 1)
    x_vals = np.array([epsilon_x(t) for t in tlist])
    y_vals = np.array([epsilon_y(t) for t in tlist])

    plt.figure(figsize=(8, 5))
    plt.plot(tlist, x_vals, label=r'$\epsilon_x$', linewidth=2)
    plt.plot(tlist, y_vals, label=r'$\epsilon_y$', linewidth=2)
    plt.xlabel(r'Time (ns)')
    plt.ylabel(r'Pulse amplitude')
    plt.title(r'Pulse functions $\epsilon_x$ and $\epsilon_y$ vs. time')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_matrix(A, d=3, th=1e-4):
    f = lambda x: "0" if abs(x) < th else (f"{x:.{d}e}" if abs(x) < 10**(-d) or abs(x) >= 1e4 else f"{x:.{d}f}")
    s = lambda z: (
        f(z.real) if abs(z.imag) < th else
        f"{f(z.imag)}j" if abs(z.real) < th else
        f"{f(z.real)} {'+' if z.imag >= 0 else '-'} {f(abs(z.imag))}j"
    )
    rows = [[s(z) for z in row] for row in A]
    w = [max(len(r[j]) for r in rows) for j in range(len(rows[0]))]
    print('\n'.join('[ ' + '  '.join(x.rjust(wi) for x, wi in zip(r, w)) + ' ]' for r in rows))

config = {
    'initial_state': 'b',  # 'a' or 'b' for initial state selection
    'amplitude': 0.2,  # in GHz
    'detune': 0.002, # in GHz # 223 # 0.00389
}
if __name__ == "__main__":
    s = Simulation()
    amp = config['amplitude'] * 2 * np.pi
    detune = config['detune'] * 2 * np.pi
    # detune = 0
    wd = s.find_resonance(amp) + detune

    heff = s.heff(wd, amp)
    tg, tlist = find_optimal_gate_time(wd, amp, s, heff)

    Omega_ab = heff[0, 1]
    tg,f_optimal,U_optimal = find_optimal_time(wd,amp,s.order,
        s.state_b,s.state_a,s.state_c,s.E_array,s.V1_dressed_array, int(np.pi/(2*Omega_ab.real)),)
    print("f_optimal: ", f_optimal)
    print_matrix(U_optimal)

    print("tg")
    lh = compute_t_dependency(s, heff, tg, use_drag=True, use_gauss = True)
    f = s.extract_fidelity(lh, tlist)
    print(" simulated fidelity: ", (f*100),"%")