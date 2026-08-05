import numpy as np
import sympy as sp
from qutip import *
from Floquet_perturbation_theory import *
from helper_function import *
from scipy.optimize import fsolve
from utils_math import *
from scipy.integrate import quad
from scipy.optimize import root_scalar

class Simulation:
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
        self.D = self.dim_q1 * self.dim_c * self.dim_q2
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
        self.a_q1 = tensor(destroy(self.dim_q1), qeye(self.dim_c), qeye(self.dim_q2))
        self.a_q2 = tensor(qeye(self.dim_q1), qeye(self.dim_c), destroy(self.dim_q2))
        self.a_qc = tensor(qeye(self.dim_q1), destroy(self.dim_c), qeye(self.dim_q2))

    def _setup_hamiltonian(self):
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
        # V1 = (self.a_qc.dag() @ self.a_qc.dag() @ self.a_qc @ self.a_qc) + (self.a_qc.dag() @ self.a_qc)
        # Diagonalize H0
        evals, evecs = H0.eigenstates()
        self.sorted_evals, self.sorted_evecs = SortedFRFSpectrum(evals, evecs, self.dim_q1, self.dim_c, self.dim_q2)
        self.E_array = self.sorted_evals.reshape(self.D)
        self.E_states = self.sorted_evecs.reshape(self.D)
        self.dressed_base_states = [basis(27, idx) for idx in range(0,27)]
        self.H0 = H0 # be sure to use H0 when working in the normal basis
        self.V1 = V1 # be sure to use V1 when working in the normal basis (not dressed)
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
        self.V1_dressed_array = self.V1_dressed.full().reshape(self.D, self.D).real
        # Resonant frequency
        self.res_freq_static = (self.sorted_evals[1, 0, 0] - self.sorted_evals[0, 0, 1]) / (2 * np.pi)

    def resonant_condition(self, fre, amp, order, i, f, use_c):
        fre = float(np.atleast_1d(fre)[0])
        resonances = {self.state_a:0, self.state_b:1, self.state_c:2} if use_c else {self.state_a:0, self.state_b:1}
        
        delta_i = Heff_Floquet_summed(self.order,i,i,
            fre, resonances,self.E_array,amp / 2 * self.V1_dressed_array,V0=None,analytics=False,)
        delta_f = Heff_Floquet_summed(order,f,f,
            fre, resonances,self.E_array,amp / 2 * self.V1_dressed_array,V0=None,analytics=False,)
        diff = delta_f - delta_i
        return float(np.real(diff))

    def find_resonance(self, amp, use_c = False):
        wd_initial_guess = self.res_freq_static * 2*np.pi
        resonant_wd_solution = fsolve(self.resonant_condition, wd_initial_guess, 
            args=(amp, self.order, self.state_a, self.state_b, use_c),)
        return resonant_wd_solution[0]

    def heff_element(self, i, j, wd, amp):
        V_posharm = (amp/2) * self.V1_dressed_array
        V0 = None
        element = Heff_Floquet_summed(self.order, i, j, wd, self.resonances, self.E_array, V_posharm, V0=V0)
        return element

    def heff(self, wd, amp, dwd=0, da=0, t=0):
        heff = np.zeros((self.sd, self.sd), dtype=complex)
        states = [*self.resonances.keys()]
        for idx_a, state_a in enumerate(states):
            for idx_b, state_b in enumerate(states):
                heff[idx_a,idx_b] = self.heff_element(state_a, state_b, wd, amp)
        return heff
    
def _is_symbolic(value):
    return isinstance(value, sp.Basic)


def _sigma_op(i, j, d, kind='x', symbolic=False):
    if symbolic:
        op = sp.zeros(d, d)
        if kind == 'x':
            op[i, j] = 1
            op[j, i] = 1
        else:
            op[i, j] = -sp.I
            op[j, i] = sp.I
        return op
    return sigma_x_ij(i, j, d) if kind == 'x' else sigma_y_ij(i, j, d)

def h_target(Delta, rrr, tg, sigma_r, pulse='gauss'):
    symbolic = any(_is_symbolic(v) for v in (Delta, rrr, tg, sigma_r))
    if symbolic:
        return h_target_symbolic(Delta, rrr, tg, sigma_r, pulse)
    return h_target_numeric(Delta, rrr, tg, sigma_r, pulse)

def h_target_symbolic(Delta, rrr, tg, sigma_r, pulse='gauss'):
    t = sp.Symbol('t')
    H = sp.diag(0, 0, Delta)
    delta1_mat = sp.diag(0, 1, 0)
    epsilonx, epsilony, delta1 = pulse_functions(tg, Delta, rrr, sigma_r, pulse)

    H_total = H + delta1_mat * delta1(t)

    for i in range(2):
        lambda_i = 1 if i == 0 else rrr
        H_total += _sigma_op(i, i + 1, 3, symbolic=True) * lambda_i * sp.Rational(1, 2) * epsilonx(t)
        H_total += _sigma_op(i, i + 1, 3, kind='y', symbolic=True) * lambda_i * sp.Rational(1, 2) * epsilony(t)

    return H_total, (epsilonx(t), epsilony(t), delta1(t))

def h_target_numeric(Delta, rrr, tg, sigma_r, pulse='gauss'):

    H = Qobj(np.diag([0, 0, Delta]))
    delta1_mat = Qobj(np.diag([0, 1, 0]))
    epsilonx, epsilony, delta1 = pulse_functions(tg, Delta, rrr, sigma_r, pulse)

    Vx, Vy = [], []

    for i in range(2):
        lambda_i = 1 if i == 0 else rrr
        Vx.append([Qobj(_sigma_op(i, i + 1, 3, symbolic=False)) * lambda_i / 2, epsilonx])
        Vy.append([Qobj(_sigma_op(i, i + 1, 3, kind='y', symbolic=False)) * lambda_i / 2, epsilony])

    return [H, [delta1_mat, delta1], *Vx, *Vy], (epsilonx, epsilony, delta1)

def pulse_functions(tg, Delta, rrr, sigma_r=0.3, pulse='gauss'):

    symbolic = any(_is_symbolic(v) for v in (tg, Delta, rrr, sigma_r))
    sigma = sigma_r * tg
    t = sp.Symbol('t') if symbolic else None
    xp = sp if symbolic else np
    pi = sp.pi if symbolic else np.pi

    if pulse == 'gauss':
        B_ratio = xp.exp(-tg**2 / (8 * sigma**2))
        def pulse_shape(t, A):
            return A * (xp.exp(-(t - tg / 2)**2 / (2 * sigma**2)) - B_ratio)
        def pulse_derivative(t, A):
            return -A * (t - tg / 2) * xp.exp(-(t - tg / 2)**2 / (2 * sigma**2)) / sigma**2
    elif pulse == 'tanh':
        def pulse_shape(t, A):
            return A * xp.tanh(t / sigma) * xp.tanh((tg - t) / sigma)
        def pulse_derivative(t, A):
            x, y = t / sigma, (tg - t) / sigma
            return A / sigma * (xp.tanh(y) / xp.cosh(x)**2 - xp.tanh(x) / xp.cosh(y)**2)
    else:
        raise ValueError(f"Unsupported pulse '{pulse}'. Use 'gauss' or 'tanh'.")

    if symbolic:
        t_prime = sp.symbols("t^{\\prime}")
        A = pi / sp.Integral(pulse_shape(t_prime, 1), (t_prime, 0, tg))
    else:
        A = root_scalar(lambda A: quad(lambda t: pulse_shape(t, A), 0, tg)[0] - pi, bracket=[0, 5], method='brentq').root

    def ep_pi(t, args=None):
        return pulse_shape(t, A)

    def ep_pi_tder(t, args=None):
        return pulse_derivative(t, A)

    def delta1(t, args=None):
        ep = ep_pi(t)
        return ((rrr**2 - 4) * ep**2 / (4 * Delta) - (rrr**4 - 7 * rrr**2 + 12) * ep**4 / (16 * Delta**3))

    def epsilonx(t, args=None):
        ep = ep_pi(t)
        return ep + (rrr**2 - 4) * ep**3 / (8 * Delta**2) - (13 * rrr**4 - 76 * rrr**2 + 112) * ep**5 / (128 * Delta**4)

    def epsilony(t, args=None):
        ep, ep_dot = ep_pi(t), ep_pi_tder(t)
        return -ep_dot / Delta + 33 * (rrr**2 - 2) * ep**2 * ep_dot / (24 * Delta**3)

    return epsilonx, epsilony, delta1

def compute_cos_params(s, tg, use_c=False):
    epsilonx = np.pi/(2*tg)
    amp_min, amp_max = 0.001, 1.5*2*np.pi
    f = lambda A: np.abs(s.heff_element(s.state_a, s.state_b, s.find_resonance(A, use_c = use_c), A)) - epsilonx # epsilonx * 1 = Omega_ab
    sol = root_scalar(f, bracket=[amp_min, amp_max])
    return sol.root, s.find_resonance(sol.root, use_c = use_c)