from qutip import *
import numpy as np
from tqdm import tqdm
from qutip import *

def fourier_coeffs(f, wd, N=100, M=10000):
    T = 2*np.pi / wd
    t = np.linspace(0, T, M, endpoint=False)
    dt = t[1] - t[0]
    ft = f(t)  # shape: (M, D, D)
    coeffs = []
    for n in range(0, N):
        # Berechne V_n = (1/T) * integral_0^T V(t) * exp(-i*n*wd*t) dt
        exp_term = np.exp(-1j*n*wd*t)  # shape: (M,)
        integrand = ft * exp_term[:, None, None]  # shape: (M, D, D)
        coeff = (1/T) * np.sum(integrand, axis=0) * dt  # shape: (D, D)
        coeffs.append(coeff.real) # we only insert real pulse shapes
    return np.array(coeffs)

def sigma_x_ij(i, j, d):
    ei = basis(d, i)
    ej = basis(d, j)
    return ej*ei.dag() + ei*ej.dag()

def sigma_y_ij(i, j, d):
    ei = basis(d, i)
    ej = basis(d, j)
    return -1j*ei*ej.dag() + 1j*ej*ei.dag()

def extract_fidelity(lh, tlist, sd = 3):
    U_ideal = np.array([[0, 1j, 0], [1j, 0, 0], [0, 0, 1]], dtype=complex)
    U_subsys = np.zeros((sd, sd), dtype=complex)
    for col_idx in range(0,sd): 
        phi0 = basis(sd, col_idx)
        result = mesolve(
            lh,
            phi0,
            tlist,
            e_ops=[],
        )
        final_state = result.states[-1]
        # Extract subsystem components to form column of unitary
        for row_idx in range(0,sd):
            U_subsys[row_idx, col_idx] = final_state.full()[row_idx, 0]
    # print_matrix(U_subsys)
    fidelity = qutip.process_fidelity(Qobj(U_subsys), Qobj(U_ideal))
    return fidelity

def extract_pop_fid(lh, tlist, sd=3):
    """
    Extract population fidelity by checking if population is exchanged for important initial states.
    For iSWAP-like behavior: |0⟩ ↔ |1⟩ (lowest two states)
    
    Parameters:
    -----------
    lh : Qobj
        Liouvillian or Hamiltonian for time evolution
    tlist : array
        Time list for evolution
    sd : int
        System dimension (default: 3 for qutrit system)
    
    Returns:
    --------
    dict : Contains population exchanges for each initial state
        - 'pop_0_to_1': Population in state 1 after starting from state 0
        - 'pop_1_to_0': Population in state 0 after starting from state 1
        - 'pop_2_final': Population in state 2 for additional test case (if sd > 2)
        - 'iswap_fidelity': Average fidelity of population exchange
    """
    
    # Define initial states for the two important states and one additional test case
    initial_states = {
        'state_0': basis(sd, 0),  # |0⟩
        'state_1': basis(sd, 1),  # |1⟩
    }
    
    # Add additional test case state if system has more than 2 levels
    if sd > 2:
        initial_states['state_2'] = basis(sd, 2)  # |2⟩
    
    results = {}
    
    for state_name, initial_state in initial_states.items():
        # Run time evolution
        result = mesolve(
            lh,
            initial_state,
            tlist,
            e_ops=[],
        )
        
        final_state = result.states[-1]
        final_state_vec = final_state.full().flatten()
        
        # Calculate populations in computational basis
        populations = np.abs(final_state_vec)**2
        
        results[f'initial_{state_name}'] = state_name
        
        # Store populations for all levels
        for i in range(min(sd, len(populations))):
            results[f'pop_{i}_{state_name}'] = populations[i]
    
    # Check iSWAP exchange conditions for the two lowest states
    pop_0_to_1 = results.get('pop_1_state_0', 0)  # Should be close to 1 for perfect iSWAP
    pop_1_to_0 = results.get('pop_0_state_1', 0)  # Should be close to 1 for perfect iSWAP
    
    # Additional test case check (if sd > 2): state |2⟩ should remain in |2⟩
    pop_2_final = results.get('pop_2_state_2', 0) if sd > 2 else None
    
    # Calculate fidelity based on iSWAP conditions
    iswap_fidelity = (pop_0_to_1 + pop_1_to_0) / 2.0
    
    # Store results
    results['pop_0_to_1'] = pop_0_to_1
    results['pop_1_to_0'] = pop_1_to_0
    if pop_2_final is not None:
        results['pop_2_final'] = pop_2_final
    results['iswap_fidelity'] = iswap_fidelity
    
    return results

def partial_derivative_heff_element(s, i, j, amp, wd, var='amp_real', h=1e-4):
    """
    Calculate partial derivative of heff element (i,j) with respect to amplitude or frequency.
    Uses finite differences for numerical differentiation.
    
    Parameters:
    -----------
    s : Simulation object
        The simulation object with heff(wd, amp) method
    i, j : int
        Row and column indices of the heff element
    amp : complex
        Current amplitude (complex value)
    wd : float
        Current angular frequency (real value)
    var : str
        Variable to differentiate with respect to:
        - 'amp_real': real part of amplitude
        - 'amp_imag': imaginary part of amplitude
        - 'wd': angular frequency
    h : float
        Step size for finite differences (default: 1e-4)
        
    Returns:
    --------
    complex
        Partial derivative d(heff[i,j])/dvar
        
    Example:
    --------
    >>> s = Simulation()
    >>> amp = 0.2 * 2*np.pi
    >>> wd = 0.71 * 2*np.pi
    >>> d_amp_real = partial_derivative_heff_element(s, 0, 1, amp, wd, var='amp_real')
    >>> d_wd = partial_derivative_heff_element(s, 1, 1, amp, wd, var='wd')
    """
    
    if var == 'amp_real':
        heff_plus = s.heff(wd, amp + h)
        heff_minus = s.heff(wd, amp - h)
    elif var == 'amp_imag':
        heff_plus = s.heff(wd, amp + 1j*h)
        heff_minus = s.heff(wd, amp - 1j*h)
    elif var == 'wd':
        heff_plus = s.heff(wd + h, amp)
        heff_minus = s.heff(wd - h, amp)
    else:
        raise ValueError("var must be 'amp_real', 'amp_imag', or 'wd'")
    
    derivative = (heff_plus[i][j] - heff_minus[i][j]) / (2 * h)
    return derivative
