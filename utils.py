import numpy as np
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

    def resonant_condition(self, fre, amp, order, i, f, use_c):
        fre = float(np.atleast_1d(fre)[0])
        resonances = {self.state_a:0, self.state_b:1} if not use_c else {self.state_a:0, self.state_b:1, self.state_c:2} 
        
        delta_i = Heff_Floquet_summed(self.order,i,i,
            fre,resonances,self.E_array,amp / 2 * self.V1_dressed_array,V0=None,analytics=False,)
        delta_f = Heff_Floquet_summed(order,f,f,
            fre,resonances,self.E_array,amp / 2 * self.V1_dressed_array,V0=None,analytics=False,)
        diff = delta_f - delta_i
        return float(np.real(diff))

    def find_resonance(self, amp, use_c = True):
        wd_initial_guess = self.res_freq_static * 2*np.pi
        resonant_wd_solution = fsolve(self.resonant_condition, wd_initial_guess, 
            args=(amp, self.order, self.state_a, self.state_b, use_c),)
        return resonant_wd_solution[0]

    def heff_element(self, i, j, wd, amp):
        V_posharm = (amp/2) * self.V1_dressed_array
        V0 = None
        element = Heff_Floquet_summed(self.order, i, j, wd, self.resonances, self.E_array, V_posHarm=V_posharm, V0=V0)
        return element

    def heff(self, wd, amp):
        heff = np.zeros((self.sd, self.sd), dtype=complex)
        states = [*self.resonances.keys()]
        for idx_a, state_a in enumerate(states):
            for idx_b, state_b in enumerate(states):
                heff[idx_a,idx_b] = self.heff_element(state_a, state_b, wd, amp)
        return heff
    
    # converts 0(a),1(b),2(c), index in the used indexing system
    def get_associated_index(self, idx):
        match idx:
            case 0: return self.state_a
            case 1: return self.state_b
            case 2: return self.state_c

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

def h_target(Delta, rrr, tg, use_drag=True, sigma_r=0.5):
    H = np.zeros((3, 3), dtype=complex)
    H[0][0] = 0
    H[1][1] = 0
    H[2][2] = Delta
    Vx = [None] * (2)
    Vy = [None] * (2)
    epsilonx, epsilony, delta1 = pulse_functions_gauss(tg, Delta, rrr, use_drag = use_drag, sigma_ratio=sigma_r)
    delta1_mat = [[0,0,0], [0,1,0], [0,0,0]]
    for i in range(0,2):
        lambda_i = 1 if not i == 1 else rrr
        Vx[i] = [sigma_x_ij(i, i+1, 3) * lambda_i * 1/2, epsilonx]
        Vy[i] = [sigma_y_ij(i, i+1, 3) * lambda_i * 1/2, epsilony]
    args = epsilonx, epsilony, delta1
    return [Qobj(H), [Qobj(delta1_mat), delta1], *Vx, *Vy], args
# define a function to change the list representation of h_target to one time dependant matrix function
def sum_list(ht):
    def f(t):
        result = ht[0]
        for i in range(1, len(ht)):
            result = result + ht[i][0] * ht[i][1](t)
        return result
    return f

def compute_t_dependency(heff, tg):
    lambda_param = heff[0][1] / heff[1][2]
    Delta = -(heff[0][0] - heff[1][1])/2
    detune = heff[0][0] - heff[1][1]
    return h_target(Delta, lambda_param, tg, delta_a = heff[2][2], delta_c = heff[0][0], detune = detune)

def pulse_functions_gauss(tg, Delta, rrr, sigma_ratio=0.3, use_drag=True):
    sigma = tg * sigma_ratio
    B_ratio = np.exp(-tg**2 / (8 * sigma**2)) # B = B_ratio * A

    gauss = lambda t: np.exp(-(t - tg / 2)**2 / (2 * sigma**2))
    ep_pi_s = lambda t, A: (gauss(t) - B_ratio) * A
    if(use_drag):
        delta1_s = lambda t, A: ((rrr**2-4) * ep_pi_s(t, A)**2/(4*Delta) -
                    (rrr**4 - 7*rrr**2 + 12)*ep_pi_s(t, A)**4 / (16*Delta**3))
    else: delta1_s = lambda t,A, args=None: 0
    # find the pulse-amplitude
    def pulse_area(A):
        # f = lambda t: np.sqrt(delta1_s(t, A)**2 + 1/4*ep_pi_s(t, A)**2)
        f = lambda t: ep_pi_s(t, A)/2
        # f = lambda t: (ep_pi(t, A)*np.sqrt(1-ep_pi(t,A)**2*
        #                                    ((rrr**2-4)/(4*Delta)-ep_pi(t,A)**2*(rrr**4-7*rrr**2+12)/(16*Delta**3))))
        # f = lambda t: np.sqrt(delta_detune**2 + A**2 * (np.exp(-(t - tg/2)**2 / (2*sigma**2)) - B_ratio)**2)
        I, _ = quad(f, 0, tg)
        return I
    f = lambda A: pulse_area(A) - np.pi/2
    sol = root_scalar(
        f,
        bracket=[0, 2],
        method='brentq'
    )
    A = sol.root
    ep_pi = lambda t, args=None: ep_pi_s(t, A)
    delta1 = lambda t, args=None: delta1_s(t, A)
    if use_drag:
        def ep_pi_tder(t, args=None):
            return -A * (t - tg / 2) * gauss(t) / sigma**2
        # ep_pi_tder = lambda t: -A/sigma**2 * (t-tg/2) * np.exp(-(t-tg/2)**2/(2*sigma**2))
        epsilonx = lambda t, args=None: (ep_pi(t) + 
                            (rrr**2 - 4)*ep_pi(t)**3 / (8*Delta**2) - 
                            (13*rrr**4 - 76*rrr**2 + 112)*ep_pi(t)**5/(128*Delta**4))
        epsilony = lambda t, args=None: (-ep_pi_tder(t)/Delta +
                            33*(rrr**2 - 2) * ep_pi(t)**2 * ep_pi_tder(t) / (24*Delta**3))
    else: 
        epsilonx = lambda t, args=None: ep_pi(t)
        epsilony = lambda t, args=None: 0
        delta1 = lambda t,args=None: 0
    return epsilonx, epsilony, delta1

def compute_cos_params(s, tg):
    epsilonx = np.pi/(2*tg)
    amp_min, amp_max = 0.001, 1.5*2*np.pi
    f = lambda A: np.abs(s.heff_element(s.state_a, s.state_b, s.find_resonance(A), A)) - epsilonx # epsilonx * 1 = Omega_ab
    sol = root_scalar(f, bracket=[amp_min, amp_max])
    return sol.root, s.find_resonance(sol.root)

if __name__ == "__main__":
    print("start")
    s = Simulation()
    # gaussian without drag
    tg = 250
    tlist_fid = np.linspace(0, tg, 25000)
    cos_amp, cos_wd = compute_cos_params(s, tg)
    print("cos_wd: ", cos_wd/(2*np.pi), " cos_amp: ", cos_amp/(2*np.pi))
    gate_time, fid, _ = find_optimal_time(
        cos_wd, cos_amp, 3,
        s.state_a, s.state_b, s.state_c,
        s.E_array, s.V1_dressed_array,
        tg
    )
    print("gate time: ", gate_time, " fidelity: ", fid)

    lh = [s.H0_dressed, [s.V1_dressed, lambda t, args: cos_amp * np.cos(cos_wd * t)]]
    f_cos = extract_fidelity(lh, tlist_fid, sim=s)
    fpop_cos = extract_pop_fid(lh, tlist_fid, s=s, plot=True)
    # print("cos: ", fpop_cos)
    print(f"infidelity cos: {(1-f_cos):.2e} pop: {(1-fpop_cos['iswap_fidelity']):.2e}")