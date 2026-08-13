import numpy as np
import sympy as sp
from qutip import *
from Floquet_perturbation_theory import *
from helper_function import *
from scipy.optimize import fsolve
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
        def get_index_for_state(n_q1, n_c, n_q2):
            """Gibt die Energie E für den Zustand |n_q1, n_c, n_q2> aus E_array zurück."""
            return n_q1 * (dim_c * dim_q2) + n_c * dim_q2 + n_q2
        self.state_b = get_index_for_state(1,0,0)  # state |100>
        self.state_a = get_index_for_state(0,0,1)  # state |001>
        self.state_c =  get_index_for_state(0,1,0)  # state |010>
        self.state_000 =  get_index_for_state(0,0,0) # state |002> # this state is at same energy as state_a at the resonant frequency
        self.state_101 =  get_index_for_state(1,0,1)
        self.U_ideal = np.array([
            [1,  0,  0, 0],
            [0,  0, 1j, 0],
            [0, 1j,  0, 0],
            [0,  0,  0, 1]
        ], dtype=complex)
        self.comp_indices = [self.state_000, self.state_a, self.state_b, self.state_101]
        self.order = 3
        self.resonances = {self.state_a: 0, self.state_b: 1, self.state_c: 2}  # E_state_a - wd = E_state_b
        self.d_res = len(self.resonances)
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
        heff = np.zeros((self.d_res, self.d_res), dtype=complex)
        states = [*self.resonances.keys()]
        for idx_a, state_a in enumerate(states):
            for idx_b, state_b in enumerate(states):
                heff[idx_a,idx_b] = self.heff_element(state_a, state_b, wd, amp)
        return heff
    
def compute_cos_params(s, tg, use_c=False):
    epsilonx = np.pi/(2*tg)
    amp_min, amp_max = 0.001, 1.5*2*np.pi
    f = lambda A: np.abs(s.heff_element(s.state_a, s.state_b, s.find_resonance(A, use_c = use_c), A)) - epsilonx # epsilonx * 1 = Omega_ab
    sol = root_scalar(f, bracket=[amp_min, amp_max])
    return s.find_resonance(sol.root, use_c = use_c), sol.root

def column_evolution(lh, tlist, j, debug=False, s=None):
    options = {"progress_bar": "tqdm", "nsteps":100000} if debug else None

    if s is None:
        resonant_states = [basis(3, i) for i in range(3)]
    else:
        # Order resonant state indices by their assigned order/value
        state_indices = list(s.resonances.keys())

        # Try to determine the Hamiltonian dimension by probing the provided `lh`.
        Hdim = None
        try:
            # `lh` may be a time-dependent list like [H_func] or a callable, or a Qobj
            if isinstance(lh, (list, tuple)):
                h0 = lh[0](tlist[0]) if callable(lh[0]) else lh[0]
            elif callable(lh):
                h0 = lh(tlist[0])
            else:
                h0 = lh

            # Qobj or ndarray: get first dimension
            Hdim = h0.shape[0] if hasattr(h0, 'shape') else None
        except Exception:
            Hdim = None

        # If the Hamiltonian lives in the reduced resonant subspace (Hdim == number of resonant states)
        # then use basis vectors of that reduced space. Otherwise fall back to full eigenstates.
        if Hdim is not None and Hdim == len(state_indices):
            resonant_states = [basis(Hdim, idx) for idx in range(Hdim)]
        else:
            resonant_states = [s.E_states[k] for k in state_indices]

    if j >= len(resonant_states):
        raise ValueError(f"Column index j={j} out of range.")

    result = mesolve(lh, resonant_states[j], tlist, e_ops=[], options=options)

    return np.array([
        [state.overlap(psi_t) for psi_t in result.states]
        for state in resonant_states
    ])