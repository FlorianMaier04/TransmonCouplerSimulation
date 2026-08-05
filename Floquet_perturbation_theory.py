"""
**General resonant Floquet perturbation theory for multiphoton processes**

Module containing the main functions to compute effective Hamiltonian and dynamics for multiphoton processes in a resonantly driven systems. 
The main functions are (see their respective help documentation for details):

- `Heff_Floquet()`: to compute the matrix elements of the effective Hamiltonian at a given order.

- `Heff_Floquet_summed()`: to compute the sum of lowest orders of the matrix elements of the effective Hamiltonian.

- `W_Floquet()`: to compute the non-zero matrix elements at a given order of the transformation `W` mapping the eigenstates of Heff in the degenerate Sambe subspace to the eigenstates of the full Sambe Hamiltonian in the full Sambe space.

- `dynamics_eff()`: to compute the effective evolution of the wavefunction obtained from an effective Hamiltonian and operator `W`. The function takes as input the outputs of the functions `Heff_Floquet_matrix_summed()` and `W_Floquet_elements` (see their respective help documentation).
"""

import numpy as np
import itertools
import sympy as sy
import copy
from fractions import Fraction
import os
import warnings

## Folder containing the .txt files of DPT coefficients
folder_DPT = os.path.join(os.path.dirname(__file__), 'DPT')

def Heff_Floquet(r, b, a, wd, resonances, E, V_posHarm, V0=None, ref_state=None, printProcesses=False, analytics=False, check_validity=False):
    """
    Effective Hamiltonian for multi-photon resonances of a periodically driven Hamiltonian.

    Return the matrix element <b|Heff^{(r)}|a> of the effective Hamiltonian at order r (no sum of lower orders) between two resonant states a and b.

    The time-dependent Hamiltonian must be decomposed as
        H(t) = H0 + V(t)
             = H0 + V0 + sum_{p>=1} ( V_p e^{-i p wd t} + V_{-p} e^{i p wd t} )
        Where 
            H0 = sum_{k=0}^{D-1} E[k] |k><k| is the un-driven Hamiltonian of eigen-energies E[k] and dimension D.
            V0 is an (optional) static perturbation.
            V_p are the drive harmonics.

    Parameters
    --------
    r : integer
        order of perturbation theory
    b, a : integer
        states of the matrix element <b|Heff|a>
    wd : float or SymPy symbol
        drive frequency
    resonances : dict or list of tuples
        dictionary of resonance orders of the states {a:na, b:nb, ...}. Such that E[a]-na*wd ~ E[b]-nb*wd. Can also give a list of pairs (state, order): [(a,na), (b,nb), ...].
    E : numpy array
        H0 spectrum. list or np.array of shape (D), with D Hilbert space dimension. Can be made of SymPy symbol.
    V_posHarm : list 
        list [V_1, V_2, V_3, ...] of the positive harmonics of the drive operator expressed in the eigenbasis |k> of H0. Each V_posHarm[p] is shape (D,D). Can also give only one harmonics V_1 (then array of shape (D,D)). Can be made of SymPy symbols.
    V0 : numpy array (optional)
        Constant drive operator perturbation. Shape (D,D). Can be made of SymPy symbols.
    ref_state : int (optional)
        reference state from which we define the energy detuning E[a] = E[ref_state] + resonances[a]*wd + Delta[a]. If None, pick the lowest energy resonant state (smallest a).
    printProcesses : bool (optional)
        if True, print the amplitude of each non-zero contribution and its type of Feynmann diagram (harmonics, virtual states, and DPT coefficient).
    analytics : bool (optional)
        if True, forces the use of symolic calculation using the module SymPy.
    
    Returns
    --------
    Hba : float or SymPy symbol
        matrix element <b| H^{(r)} |a> of the r-th order effective Hamiltonian. Real number if a=b (diagonal element).
    """
    if type(resonances) == list and type(resonances[0]) in (list, tuple):
        ## If we give a list of state and order (a, n) for resonances, we translate into a dictionnary
        resonances = {k:n for (k,n) in resonances}
    assert a in resonances.keys() and b in resonances.keys(), "a and b must be states of the degenerate subspace."
    ## Copy arrays to prevent array modification outside the function
    E = np.array(E)
    V_posHarm = np.array(V_posHarm)
    ## Force analytics if one of the data is SymPy symbol
    if E.dtype == 'O' or V_posHarm.dtype == 'O' or np.array(wd).dtype == 'O':
        analytics = True
    ## Hilbert space dimension
    D = len(E)
    ## If analytics, convert energy and V0 to symbol to manipulate the detuning
    dtype_E = sy.Symbol if analytics else float
    dtype_V = sy.Symbol if analytics else complex
    E = np.array(E, dtype=dtype_E)
    V_posHarm = np.array(V_posHarm, dtype=dtype_V)
    V0 = np.zeros((D, D), dtype=dtype_E) if V0 is None else np.array(V0, dtype=dtype_E)
    ## Choose a reference state for the detuning, if it is not given
    if ref_state is None:
        ref_state = min(resonances.keys())
    ## Put detuning in perturbation and set energy of final state to resonance. By def delta[k] = E[k] - E[ref_state] - (resonances[k] - resonances[ref_state])*wd
    for k in resonances.keys():
        if k != ref_state:
            V0[k, k] = V0[k, k] + E[k] - E[ref_state] - (resonances[k] - resonances[ref_state])*wd
            E[k] = E[ref_state] + (resonances[k] - resonances[ref_state])*wd
    ## Transform V_posHarm to list of 1 element if we give only one harmonics
    if type(V_posHarm) == np.ndarray and V_posHarm.shape == (D, D):
        V_posHarm = [V_posHarm]
    ## Number of harmonics (max number of photon absorbed or emit at each step)
    nHarm = len(V_posHarm)
    Vdag = [V_posHarm[p].T.conj() for p in range(nHarm)]
    def V(p):
        """
        Drive harmonics
        V(0) = V0
        V(1) = V_1 = Vpos[0]  ;  V(2) = V_2 = Vpos[1]  ;  
        V(-2) = V_{-2} = V_2^dag = VposDag[-1] ;
        etc"""
        if np.abs(p) > nHarm:
            return np.zeros((D, D))
        if p == 0:
            return V0
        elif p > 0:
            return V_posHarm[p-1]
        elif p < 0:
            return Vdag[-p-1]
    result = sy.S.Zero if analytics else 0
    ## Load Degenerate Perturbation Theory (DPT) coefficients. Translate them in fraction for analytical expressions.
    if r >= 3:
        ## Extract the dictionary of DPT coefficients
        ## coeffsDPT[m] is the DPT coeff of the tuple m, where m[j] is the energy exponent of j-th step
        coeffsMatrix = np.loadtxt(f"{folder_DPT}/DPT_coefficients_order_{r}.txt")
        coeffsDPT = {}
        for line in coeffsMatrix:
            if analytics:
                numerator, denominator = line[-1].as_integer_ratio()
                coeffsDPT[tuple(line[:-1])] = Fraction(numerator, denominator)
            else:
                coeffsDPT[tuple(line[:-1])] = line[-1]
    if printProcesses:
        ## Count the number of non-zero process
        nbProcesses = 0
    ## Track maximum perturbation ratio for validity check
    max_pert_ratio = 0 if not analytics else None
    max_pert_index = {'a':0, 'b':0, 'ptot':0, 'p':0} if not analytics else None
    for p in itertools.product(*(np.arange(-nHarm, nHarm+1) for _ in range(r-1))):
        ## Iterate through all possible combinations of harmonics such that p[0] + ... + p[r-1] = nb - na
        p = list(p)
        ## Add p[r-1] = nb - na - (p[0] + ... + p[r-2])
        p.append(resonances[b] - resonances[a] - sum(p))
        if abs(p[-1]) > nHarm:
            ## ignore if the required last harmonics is too large
            continue
        for k in itertools.product(*(np.arange(D) for _ in range(r-1))):
            ## Iterate through all possible virtual state k[j]
            # print(k)
            ## Add last state b (to simplify the later loop)
            k = list(k)
            k.append(b)
            ## Find resonant steps
            R = []
            Rcomp = []
            for j in range(r-1):
                # if ( k[j]==i and na+sum(p[:j+1])==0 ) or ( k[j]==f and na+sum(p[:j+1])==n ):
                if k[j] in resonances.keys() and resonances[a] + sum(p[:j+1]) == resonances[k[j]]:
                    ### step j is resonant
                    # print("harmonics", p, "virtual states", k, "resonant step", j)
                    R.append(j)
                else:
                    ## Non resonant step
                    Rcomp.append(j)
            ## Compute process amplitude
            if len(R) == 0:
                ### No resonant step: simple contribution
                contrib = V(p[0])[k[0], a]
                ### Total number of photon absorbed after each step
                ptot = 0
                for j in range(r-1):
                    ptot += p[j]
                    ## Check perturbation validity at this step
                    if check_validity and not analytics:
                        V_element = np.abs(V(p[j+1])[k[j+1], k[j]])
                        energy_gap = np.abs(E[a] + ptot*wd - E[k[j]])
                        if energy_gap > 1e-10:  # Avoid division by near-zero
                            pert_ratio = V_element / energy_gap
                            max_pert_ratio = max(max_pert_ratio, pert_ratio)
                            if max_pert_ratio == pert_ratio:
                                max_pert_index = {'a': k[j+1], 'b': k[j], 'ptot': ptot, 'p':p[j+1]}
                    contrib = contrib*V(p[j+1])[k[j+1], k[j]]/(E[a] + ptot*wd - E[k[j]])
                if printProcesses and ( (analytics and contrib != 0) or (not analytics and not np.isclose(contrib, 0, atol=1e-12))):
                    nbProcesses += 1
                    print("harmonics:", p, "virtual states:", k[:-1], "no resonant step.", "amplitude:", contrib)
                result += contrib
            elif len(Rcomp) != 0:
                ## If there are resonant process: redistribute the exponents on the other processes if it remains some.
                ## List of energy exponents of each process
                m = np.zeros(r-1, dtype=int)
                ## m[j] = 0 for j in R. The exponents are redistributed between the j' in Rcomp
                ## m[j'] = 1 + dm[j'] , where sum_{j'\in Rcomp} dm[j'] = len(R). So dm is length len(Rcomp) with max value len(R)
                # print("harmonics:", p, "virtual states:", k[:-1], "Resonant steps:", R)
                for dm in itertools.product(*(np.arange(0, len(R)+1) for _ in range(len(Rcomp)-1))):
                    ## Extra exponent on all the non-resonant steps except the last one
                    ## iterate through all couples (dm[0], ..., dm[len(Rcomp)-2]) where dm[j]=0, ..., len(R), to fix the last element dm[len(Rcomp)-1] by hand to satisfy the condition sum dm = len(R).
                    ## (If there is only one non-resonant step len(Rcomp)=1, then this loop gives only the empty couple dm=(), such that we accurately fill it by len(R): dm=(len(R)) after filling)
                    dm = list(dm)
                    # print("dm before last filling:", dm)
                    ## Add the extra exponent to the last step to satisfy the total number of extra exponent
                    dm.append(len(R) - sum(dm))
                    # print("dm after last filling:", dm)
                    if dm[-1] < 0:
                        ## Ignore if the process requires a suppression of exponent on last step
                        ## (not entirely sure if it is needed)
                        continue
                    ## dm is now the list of extra exponent to add to each non-resonant transition
                    for l in range(len(Rcomp)):
                        m[Rcomp[l]] = 1 + dm[l]
                    ## Now m[j] is the exponent for the energy gap of jth step
                    ## Check if the contribution as a non-zero DPT coeff
                    try:
                        ## Compute the contribution
                        contrib = coeffsDPT[tuple(m)]*V(p[0])[k[0], a]
                    except KeyError:
                        ## If the DPT coefficient does not exist (i.e. is zero), the skip this exponent list
                        # print(m, "no coeff")
                        continue
                    ## Total number of photon absorbed after each step
                    ptot = 0
                    for j in range(r-1):
                        ptot += p[j]
                        ## Check perturbation validity at this step
                        if check_validity and not analytics:
                            V_element = np.abs(V(p[j+1])[k[j+1], k[j]])
                            energy_gap = np.abs(E[a] + ptot*wd - E[k[j]])
                            if energy_gap > 1e-10:  # Avoid division by near-zero
                                pert_ratio = V_element / energy_gap
                                max_pert_ratio = max(max_pert_ratio, pert_ratio)
                                if max_pert_ratio == pert_ratio:
                                    max_pert_index = {'a': k[j+1], 'b': k[j], 'ptot': ptot, 'p':p[j+1]}
                        contrib = contrib*V(p[j+1])[k[j+1], k[j]]*(E[a] + ptot*wd - E[k[j]])**(-m[j])
                    if printProcesses and ( (analytics and contrib != 0) or (not analytics and not np.isclose(contrib, 0, atol=1e-12))):
                        nbProcesses += 1
                        print("harmonics:", p, "virtual states:", k[:-1], "resonant steps:", R, "powers m:", m, "DPT coeff:", coeffsDPT[tuple(m)], "amplitude:", contrib)
                    result += contrib
    ## Issue warning if perturbation theory validity is questionable
    if check_validity and not analytics and max_pert_ratio > 1.0:
        warnings.warn(
            f"Perturbation theory may not be valid at order {r}:\n"
            f"  Maximum |V|/|ΔE| = {max_pert_ratio:.3f} > 1.0\n"
            f"  Occurs for matrix element <{max_pert_index['a']}|V|{max_pert_index['b']}>={np.abs(V(max_pert_index['p'])[max_pert_index['a'], max_pert_index['b']])}\n"
            f"  with denomitor <{E[a]} + {max_pert_index['p']}*{wd} - {E[max_pert_index['b']]}>\n"
            f"  Perturbation expansion parameter exceeds unity.\n"
            f"  Results may not be reliable. Consider reducing amplitudes.",
            UserWarning
        )            
    if printProcesses:
        print("Total number of process:", nbProcesses)
    if a == b:
        ## Real diagonal term
        result = sy.re(result) if analytics else np.real(result)
    return result

def Heff_Floquet_summed(r, b, a, wd, resonances, E, V_posHarm, V0=None, ref_state=None, printProcesses=False, analytics=False, check_validity=False):
    """
    Effective Hamiltonian for multi-photon resonances of a periodically driven Hamiltonian.

    Sum of a specific matrix element sum_{l<=r} <b|Heff^{(r)}|a> of the effective Hamiltonian at order lower than r between two resonant states a and b.

    The time-dependent Hamiltonian must be decomposed as
        H(t) = H0 + V(t)
             = H0 + V0 + sum_{p>=1} ( V_p e^{-i p wd t} + V_{-p} e^{i p wd t} )
        Where 
            H0 = sum_{k=0}^{D-1} E[k] |k><k| is the un-driven Hamiltonian of eigen-energies E[k] and dimension D.
            V0 is an (optional) static perturbation.
            V_p are the drive harmonics.

    Parameters
    --------
    r : integer
        order of perturbation theory
    b, a : integer
        states of the matrix element <b|Heff|a>
    wd : float or SymPy symbol
        drive frequency
    resonances : dict or list of tuples
        dictionary of resonance orders of the states {a:na, b:nb, ...}. Such that E[a]-na*wd ~ E[b]-nb*wd. Can also give a list of pairs (state, order): [(a,na), (b,nb), ...].
    E : numpy array
        H0 spectrum. list or np.array of shape (D), with D Hilbert space dimension. Can be made of SymPy symbol.
    V_posHarm : list 
        list [V_1, V_2, V_3, ...] of the positive harmonics of the drive operator expressed in the eigenbasis |k> of H0. Each V_posHarm[p] is shape (D,D). Can also give only one harmonics V_1 (then array of shape (D,D)). Can be made of SymPy symbols.
    V0 : numpy array (optional)
        Constant drive operator perturbation. Shape (D,D). Can be made of SymPy symbols.
    ref_state : int (optional)
        reference state from which we define the energy detuning E[a] = E[ref_state] + resonances[a]*wd + Delta[a]. If None, pick the lowest energy resonant state (smallest a).
    printProcesses : bool (optional)
        if True, print the amplitude of each non-zero contribution and its type of Feynmann diagram (harmonics, virtual states, and DPT coefficient).
    analytics : bool (optional)
        if True, forces the use of symolic calculation using the module SymPy.
    
    Returns
    --------
    Hba_summed : float or SymPy symbol
        Sum of matrix element sum_{l<=r} <b| H^{(r)} |a> of the effective Hamiltonian up to order r. Real number if a=b (diagonal element).
    """
    Hba = Heff_Floquet(1, b, a, wd, resonances, E, V_posHarm, V0, ref_state, printProcesses, analytics, check_validity)
    for l in range(2, r+1):
        Hba += Heff_Floquet(l, b, a, wd, resonances, E, V_posHarm, V0, ref_state, printProcesses, analytics, check_validity)
    return Hba

def W_Floquet(r, b, a, wd, resonances, E, V_posHarm, V0=None, ref_state=None, printProcesses=False, analytics=False):
    """
    Matrix element of the W operator computed at order r of degenerate perturbation theory in Floquet Sambe space.

    << b, p | W_r | a, n_a >>

    Return a dictionnary of the harmonics p giving a non-zero matrix element.

    Parameters
    --------
    r : integer
        order of perturbation theory
    b : integer
        System state of bra
    p : integer
        Harmonics of bra
    a : integer
        System state of ket
    wd : float or SymPy symbol
        Drive frequency
    resonances : dict or list
        dictionary of resonance orders of the states {a:na, b:nb, ...}. Such that E[a]-na*wd ~ E[b]-nb*wd. Can also give a list of pairs (state, order): [(a,na), (b,nb), ...].
    E : numpy array
        H0 spectrum. list or np.array of shape (D), with D Hilbert space dimension. Can be made of SymPy symbol.
    V_posHarm : list 
        list [V_1, V_2, V_3, ...] of the positive harmonics of the drive operator expressed in the eigenbasis |k> of H0. Each V_posHarm[p] is shape (D,D). Can also give only one harmonics V_1 (then array of shape (D,D)). Can be made of SymPy symbols.
        Such that
            V(t) = V_0 + sum_{p >= 1} ( e^{-i p wd t}V_posHarm[p-1] + e^{i p wd t}V_posHarm[p-1]^dagger )
            i.e.
            V_posHarm[0] = V_1 ; V_posHarm[1] = V_2 ; etc
    V0 : numpy array (optional)
        Constant drive operator perturbation. Shape (D,D). Can be made of SymPy symbols.
    ref_state : int (optional)
        reference state from which we define the energy detuning E[a] = E[ref_state] + resonances[a]*wd + Delta[a]. If None, pick the lowest energy resonant state (smallest a).
    printProcesses : bool (optional)
        if True, print the amplitude of each non-zero contribution and its type of Feynmann diagram (harmonics, virtual states, and DPT coefficient).
    analytics : bool (optional)
        if True, forces the use of symolic calculation using the module SymPy.

    Returns
    ---------
    W_dict : dict
        dictionnary of non-zero harmonics, such that W_dict[p] = <<b, p| W_r |a, n_a>>
    """
    if r == 0:
        ## order 0, W0 = P such that <<b,p|W0|a,na>> = delta_{b,a} delta_{p,na}
        if b == a:
            return {resonances[a]: 1}
        else:
            return {}
    if type(resonances) == list and type(resonances[0]) in (list, tuple):
        ## If we give a list of state and order (a, n) for resonances, we translate into a dictionnary
        resonances = {k:n for (k,n) in resonances}
    assert a in resonances.keys(), "k must be a state of the degenerate subspace."
    ## Copy arrays to prevent array modification outside the function
    E = np.array(E)
    V_posHarm = np.array(V_posHarm)
    ## Force analytics if one of the data is SymPy symbol
    if E.dtype == 'O' or V_posHarm.dtype == 'O' or np.array(wd).dtype == 'O':
        analytics = True
    ## Hilbert space dimension
    D = len(E)
    ## If analytics, convert energy and V0 to symbol to manipulate the detuning
    dtype_E = sy.Symbol if analytics else float
    dtype_V = sy.Symbol if analytics else complex
    E = np.array(E, dtype=dtype_E)
    V_posHarm = np.array(V_posHarm, dtype=dtype_V)
    V0 = np.zeros((D, D), dtype=dtype_E) if V0 is None else np.array(V0, dtype=dtype_E)
    ## Choose a reference state for the detuning, if it is not given
    if ref_state is None:
        ref_state = min(resonances.keys())
    ## Put detuning in perturbation and set energy of final state to resonance. By def delta[k] = E[k] - E[ref_state] - (resonances[k] - resonances[ref_state])*wd
    for k in resonances.keys():
        if k != ref_state:
            V0[k, k] = V0[k, k] + E[k] - E[ref_state] - (resonances[k] - resonances[ref_state])*wd
            E[k] = E[ref_state] + (resonances[k] - resonances[ref_state])*wd
    ## Transform V_posHarm to list of 1 element if we give only one harmonics
    if type(V_posHarm) == np.ndarray and V_posHarm.shape == (D, D):
        V_posHarm = [V_posHarm]
    ## Number of harmonics (max number of photon absorbed or emit at each step)
    nHarm = len(V_posHarm)
    Vdag = [V_posHarm[p].T.conj() for p in range(nHarm)]
    def V(p):
        """
        Drive harmonics
        V(0) = V0
        V(1) = V_1 = Vpos[0]  ;  V(2) = V_2 = Vpos[1]  ;  
        V(-2) = V_{-2} = V_2^dag = VposDag[-1] ;
        etc"""
        if np.abs(p) > nHarm:
            return np.zeros((D, D))
        if p == 0:
            return V0
        elif p > 0:
            return V_posHarm[p-1]
        elif p < 0:
            return Vdag[-p-1]
    ## Load Degenerate Perturbation Theory (DPT) coefficients. Translate them in fraction for analytical expressions.
    ## Extract the dictionary of DPT coefficients
    ## coeffsDPT[m] is the DPT coeff of the tuple m, where m[j] is the energy exponent of j-th step
    if r == 1:
        coeffsDPT = {(1,): 1}
    else:
        coeffsMatrix = np.loadtxt(f"{folder_DPT}/DPT_unitaryW_coefficients_order_{r}.txt")
        coeffsDPT = {}
        for line in coeffsMatrix:
            if analytics:
                numerator, denominator = line[-1].as_integer_ratio()
                coeffsDPT[tuple(line[:-1])] = Fraction(numerator, denominator)
            else:
                coeffsDPT[tuple(line[:-1])] = line[-1]
    ## Dictionnary of harmonics p giving a non-zero <<b, p|W|a, n_a>> with the corresponding amplitude as a value
    W_terms = {}
    for p_list in itertools.product(np.arange(-nHarm, nHarm+1), repeat=r):
        ## Iterate through all possible combinations of harmonics of the perturbation p[1], ..., p[r]. 
        ## Add [None] as first element such that p is indexed as p[j] for step j=1,...,r
        p_list = [None] + list(p_list)
        # The final harmonics is then p = n_a + p_list[1] + ... + p_list[r]
        p = resonances[a] + sum(p_list[1:])
        ## Initialize the contribution to the amplitude of <<b, p|W|a, n_a>> coming from this combination of harmonics p_list
        Wp_plist_contrib = sy.S.Zero if analytics else 0
        for k in itertools.product(np.arange(D), repeat=r-1):
            ## Iterate through all possible virtual state k[j] j=1,...,r-1
            ## Add first state a and last state b (to simplify the later loop) : k[0] = a , k[1] = k1 , ...,  k[r] = b
            k = [a] + list(k) + [b]
            ## Find resonant steps j in [[1,r]] s.t. E[a] + sum_{j'=1}^j p[j']wd - E[k[j]]] = 0
            R = []
            Rcomp = []
            for j in range(1, r+1):
                if k[j] in resonances.keys() and resonances[a] + sum(p_list[1:j+1]) == resonances[k[j]]:
                    ### step j is resonant
                    # print("harmonics", p_list, "virtual states", k, "resonant step", j)
                    R.append(j)
                else:
                    ## Non resonant step
                    Rcomp.append(j)
            ## Compute process amplitude
            for exponents, c_DPT_W in coeffsDPT.items():
                ## Go through all non-zeros coefficient c_{m_r, ..., m_1}^W and exponent of DPT expansion of W.
                ## Exponent m[j] of step j=1,...,r. Add None to start indexing at j. Must reverse the exponents which gives the powers from last to first step.
                m = [None] + list(np.array(exponents[::-1], dtype=int))
                ## Make sure that all m[j] are 0 for resonant steps and non-zero for non-resonant steps
                if all( m[j] == 0 for j in R ) and all( m[j] != 0 for j in Rcomp ):
                    ## Compute contribution
                    contrib = c_DPT_W
                    ### Total number of photon absorbed after each step
                    ptot = 0
                    for j in range(1, r+1):
                        ptot += p_list[j]
                        contrib = contrib*V(p_list[j])[k[j], k[j-1]]*(E[a] + ptot*wd - E[k[j]])**(-m[j])
                    if printProcesses and ( (analytics and contrib != 0) or (not analytics and not np.isclose(contrib, 0, atol=1e-12))):
                        print("p:", p, "harmonics:", p_list[1:], "virtual states:", k[1:-1], "resonant steps:", R, "powers m:", m[1:], "DPT coeff:", c_DPT_W, "amplitude:", contrib)
                    Wp_plist_contrib += contrib
        if ( (analytics and Wp_plist_contrib != 0) or (not analytics and not np.isclose(Wp_plist_contrib, 0, atol=1e-12))):
            ## Save the matrix element <<b, p|W|a, n_a>> of harmonics p if non-zero
            if p in W_terms:
                ## If it already has a p components, then add the contribution.
                W_terms[p] += Wp_plist_contrib
            else:
                ## Else create a new p component
                W_terms[p] = Wp_plist_contrib
    return W_terms

def Heff_Floquet_matrix_summed(rH, wd, resonances, E, V_posHarm, V0=None, analytics=False):
    """
    Effective Hamiltonian up to order r (sum of orders 1 to r) returned in matrix form.
    
    Intermediate function used for the function `dynamics_eff()`. The output Heff_matrix must be used as an argument of the function `dynamics_eff()`.

    The matrix can be computed symbolically for parameter sweeps. But numerical value must be substituted before giving it as an argument of `dynamics_eff()`.

    The time-dependent Hamiltonian must be decomposed as
        H(t) = H0 + V(t)
             = H0 + V0 + sum_{p>=1} ( V_p e^{-i p wd t} + V_{-p} e^{i p wd t} )
        Where 
            H0 = sum_{k=0}^{D-1} E[k] |k><k| is the un-driven Hamiltonian of eigen-energies E[k] and dimension D.
            V0 is an (optional) static perturbation.
            V_p are the drive harmonics.

    Parameters
    --------
    rH : integer
        order of perturbation theory
    wd : float or SymPy symbol
        drive frequency
    resonances : dict or list of tuples
        dictionary of resonance orders of the states {a:na, b:nb, ...}. Such that E[a]-na*wd ~ E[b]-nb*wd. Can also give a list of pairs (state, order): [(a,na), (b,nb), ...].
    E : numpy array
        H0 spectrum. list or np.array of shape (D), with D Hilbert space dimension. Can be made of SymPy symbol.
    V_posHarm : list 
        list [V_1, V_2, V_3, ...] of the positive harmonics of the drive operator expressed in the eigenbasis |k> of H0. Each V_posHarm[p] is shape (D,D). Can also give only one harmonics V_1 (then array of shape (D,D)). Can be made of SymPy symbols.
    V0 : numpy array (optional)
        Constant drive operator perturbation. Shape (D,D). Can be made of SymPy symbols.
    analytics : bool (optional)
        if True, forces the use of symbolic calculation using the module SymPy.

    Return
    ---------
    Heff_matrix : (d, d) numpy array , where d = len(resonances) is the number of resonant states
        Matrix of the effective Hamiltonian, to be used as an argument of the function `dynamics_eff()`.
    """
    if type(resonances) == list and type(resonances[0]) in (list, tuple):
        ## If we give a list of state and order (a, n) for resonances, we translate into a dictionnary
        resonances = {k:n for (k,n) in resonances}    
    ## Copy arrays to prevent array modification outside the function
    E = np.array(E)
    V_posHarm = np.array(V_posHarm)
    ## Force analytics if one of the data is SymPy symbol
    if E.dtype == 'O' or V_posHarm.dtype == 'O':
        analytics = True
    ## Hilbert space dimension
    D = len(E)
    ## Number of resonant states
    d_res = len(resonances)
    ## Reference state kref for effective Hamiltonian (usually |0>)
    kref = min(resonances.keys())
    ## Indices j in effective Hamiltonian of resonant states ; ind = {i:0, f:1} for 2 resonant states i and f, etc.
    ind = {}
    ## Indices in Hilbert space from indices in effective Hamiltonian ; ind_inv = {0:i, 1:f} for 2 resonant states i and f, etc.
    ind_inv = {}
    s = 0
    for k in resonances.keys():
        ind[k] = s
        ind_inv[s] = k
        s += 1
    ## Effective Hamiltonian computed at order r
    dtype = sy.Symbol if analytics else complex
    Heff_matrix = np.zeros((d_res,d_res), dtype=dtype)
    for j1 in range(d_res):
        for j2 in range(d_res):
            Heff_matrix[j1,j2] = Heff_Floquet_summed(rH, ind_inv[j1], ind_inv[j2], wd, resonances, E, V_posHarm, V0, kref)
    return Heff_matrix

def W_Floquet_elements(rW, wd, resonances, E, V_posHarm, V0=None, ref_state=None, analytics=False):
    """
    All matrix elements of W up to order r needed to compute the effective dynamics.

    Intermediate function used for the function `dynamics_eff()`.

    The output is a dictionnary W_elemt such that W_elemt[r,l,a] is the dictionnary of all {p: <<l, p|W_r|a, n_a>>} for r <= rW, (a, n_a) any resonant state, l any state of the system, and p any harmonics.

    The output can be computed symbolically for parameter sweeps. But numerical values must be substituted before giving it as an argument of `dynamics_eff()`.
    
    The time-dependent Hamiltonian must be decomposed as
        H(t) = H0 + V(t)
             = H0 + V0 + sum_{p>=1} ( V_p e^{-i p wd t} + V_{-p} e^{i p wd t} )
        Where 
            H0 = sum_{k=0}^{D-1} E[k] |k><k| is the un-driven Hamiltonian of eigen-energies E[k] and dimension D.
            V0 is an (optional) static perturbation.
            V_p are the drive harmonics.

    Parameters
    --------
    rW : integer
        order of perturbation theory
    b, a : integer
        states of the matrix element <b|Heff|a>
    wd : float or SymPy symbol
        drive frequency
    resonances : dict or list of tuples
        dictionary of resonance orders of the states {a:na, b:nb, ...}. Such that E[a]-na*wd ~ E[b]-nb*wd. Can also give a list of pairs (state, order): [(a,na), (b,nb), ...].
    E : numpy array
        H0 spectrum. list or np.array of shape (D), with D Hilbert space dimension. Can be made of SymPy symbol.
    V_posHarm : list 
        list [V_1, V_2, V_3, ...] of the positive harmonics of the drive operator expressed in the eigenbasis |k> of H0. Each V_posHarm[p] is shape (D,D). Can also give only one harmonics V_1 (then array of shape (D,D)). Can be made of SymPy symbols.
    V0 : numpy array (optional)
        Constant drive operator perturbation. Shape (D,D). Can be made of SymPy symbols.
    ref_state : int (optional)
        reference state from which we define the energy detuning E[a] = E[ref_state] + resonances[a]*wd + Delta[a]. If None, pick the lowest energy resonant state (smallest a).
    printProcesses : bool (optional)
        if True, print the amplitude of each non-zero contribution and its type of Feynmann diagram (harmonics, virtual states, and DPT coefficient).
    analytics : bool (optional)
        if True, forces the use of symolic calculation using the module SymPy.
    
    Returns
    --------
    W_elemt : dict
        such that W_elemt[r,l,a] is the dictionnary of all {p: <<l, p|W_r|a, n_a>>} for r <= rW, (a, n_a) any resonant state, l any state of the system, and p any harmonics.
        Must be used as an argument of the function `dynamics_eff()`
    """
    if type(resonances) == list and type(resonances[0]) in (list, tuple):
        ## If we give a list of state and order (a, n) for resonances, we translate into a dictionnary
        resonances = {k:n for (k,n) in resonances}
    ## Copy arrays to prevent array modification outside the function
    E = np.array(E)
    V_posHarm = np.array(V_posHarm)
    ## Force analytics if one of the data is SymPy symbol
    if E.dtype == 'O' or V_posHarm.dtype == 'O':
        analytics = True
    ## Hilbert space dimension
    D = len(E)
    ## Reference state kref for effective Hamiltonian (usually |0>)
    if ref_state is None:
        kref = min(resonances.keys())
    else:
        kref = ref_state
    ## Construct element of W needed such that W[r,l,a] is the dictionnary of {p: <<l,p|W_r|a,na>>}
    W_elemt = {(r,l,a): W_Floquet(r, l, a, wd, resonances, E, V_posHarm, V0, kref, analytics=analytics) for r in range(rW+1) for l in range(D) for a in resonances.keys()}
    return W_elemt

def dynamics_eff(initial_state, rH, rW, Heff_matrix, W_elemt, t, wd, resonances, E):
    """
    Effective time-evolution of the wavefunction from the expansion of the effective Hamiltonian at order rH and of W at order rW.

    Return the evolution |psi(t)> of the state. **The state is in general not normalized, it must be normalized at posteriori**.

    The time-dependent Hamiltonian must be decomposed as
        H(t) = H0 + V(t)
             = H0 + V0 + sum_{p>=1} ( V_p e^{-i p wd t} + V_{-p} e^{i p wd t} )
        Where 
            H0 = sum_{k=0}^{D-1} E[k] |k><k| is the un-driven Hamiltonian of eigen-energies E[k] and dimension D.
            V0 is an (optional) static perturbation.
            V_p are the drive harmonics.
    
    Parameters
    --------
    initial_state : numpy array
        initial state in eigenbasis, such that initial_state[k] = <k|psi(t=0)>
    rH : int
        order of expansion of Heff
    rW : int
        order of expansion of W
    Heff_matrix : output of the function `Heff_Floquet_matrix_summed()`
        matrix of the effective Hamiltonian. Must be computed from the function `Heff_Floquet_matrix_summed()` and must be made of complex numbers. (the function `Heff_Floquet_matrix_summed()` allows for symbolic calculation, but the symbolic values must be substituted by their numerical value before giving it to dynamics_eff()).
    W_elemt : output of the function `W_Floquet_elements()`
        matrix elements of the W operator. Must be computed from the function `W_Floquet_elements()` and must be made of complex numbers. (the function `W_Floquet_elements()` allows for symbolic calculation, but the symbolic values must be substituted by their numerical value before giving it to dynamics_eff()).
    t : float or SymPy symbol
        time of evaluation of the wavefunction |psi(t)>. *We recommend using a SymPy symbol and substitute it afterward to evaluate the function only once)*.
    wd : float
        resonance frequency
    E : numpy array
        H0 spectrum. list or np.array of shape (D), with D Hilbert space dimension.

    Return
    --------
    psi_t_not_normalized : numpy array
        state evaluated at time t: psi_t_not_normalized[k] = <k|psi(t)>.
        *The state is not normalized. It must be normalized a posteriori.*
    """
    if type(resonances) == list and type(resonances[0]) in (list, tuple):
        ## If we give a list of state and order (a, n) for resonances, we translate into a dictionnary
        resonances = {k:n for (k,n) in resonances}  
    ## Hilbert space dimension
    D = len(E)
    ## Number of resonant states
    d_res = len(resonances)
    ## Reference state kref for effective Hamiltonian (usually |0>)
    kref = min(resonances.keys())
    ## Indices j in effective Hamiltonian of resonant states ; ind = {i:0, f:1} for 2 resonant states i and f, etc.
    ind = {}
    ## Indices in Hilbert space from indices in effective Hamiltonian ; ind_inv = {0:i, 1:f} for 2 resonant states i and f, etc.
    ind_inv = {}
    s = 0
    for k in resonances.keys():
        ind[k] = s
        ind_inv[s] = k
        s += 1
    ## Diagonalize effective Hamiltonian Heff|s>>=epsilon_s|s>>. u[j,s] is <<k_j, n_kj|s>> 
    epsilon, u = np.linalg.eigh(Heff_matrix)
    ## Time-evolution
    evol_state = np.zeros(D, dtype='O')
    for l in range(D):
        for k in range(D):
            for a in resonances.keys():
                for b in resonances.keys():
                    for r1 in range(rW + 1):
                        for r2 in range(rW - r1 + 1):
                            for s in range(d_res):
                                for p1, W_r1_lp1_ana in W_elemt[r1, l, a].items():
                                    for p2, W_r2_kp2_bnb in W_elemt[r2, k, b].items():
                                        ## cl(t) += e^{-i (epsilon_s + p1 wd) t} <<l,p1|W_r1|a,na>> <<a,na|s>> <<s|b,nb>> (<<k,p2|W_r2|b,nb>>)^* ck(0)
                                        evol_state[l] += sy.exp(-1j*(E[kref] + epsilon[s] + p1*wd)*t) * W_r1_lp1_ana * u[ind[a],s] * np.conj(u[ind[b],s]) * np.conj(W_r2_kp2_bnb) * initial_state[k]
    return evol_state

def Heff_Floquet_multModes(r, b, a, wd, resonances, E, V, V0=None, ref_state_and_n=None, printProcesses=False, analytics=False, detuning_all_replicates=True):
    """
    Effective Hamiltonian for multi-photon resonances of a quasi-periodic Hamiltonian (Nwd driving modes of frequency wd_1, ..., wd_Nwd).

    Return the matrix element <b|Heff^{(r)}|a> of the effective Hamiltonian at order r (no sum of lower orders) between two resonant states a and b.

    The time-dependent Hamiltonian must be decomposed as
        H(t) = H0 + V(t)
             = H0 + V0 + sum_{p_1>=1} ... sum_{p_Nwd>=1} ( V_{p_1,...,p_Nwd} e^{-i (p_1 wd_1 + ... + p_Nwd wd_Nwd) t} + V_{-p_1,...,-p_Nwd} e^{i (p_1 wd_1 + ... + p_Nwd wd_Nwd) t} )
        Where
            Nwd is the number of driving mode
            H0 = sum_{k=0}^{D-1} E[k] |k><k| is the un-driven Hamiltonian of eigen-energies E[k] and dimension D.
            V0 is an (optional) static perturbation.
            V_{p_1,...,p_Nwd} are the drive harmonics
    
    Parameters
    ---------
    r : integer
        order of perturbation theory.
    b, a: integer
        states of the matrix element <b|Heff|a>. Can also give the state and its harmonics (state, n_1, ..., n_Nwd) if several harmonics combinations are possible for the same state (relevent for the multi-mode case at higher orders).
    wd : array like
        list of mode frequencies [wd_1, ..., wd_Nwd] of length Nwd. Can be a scalar for single mode Nwd=1. Can be made of SymPy symbols.
    resonances : list
        resonant states and their orders. List of resonant states [(a, (na_1, ..., na_Nwd)), (b, (nb_1, ..., nb_Nwd)), ...] 
        such that E[a] - (na_1*wd_1 + ... + na_Nwd*wd_Nwd) ~ E[b] - (nb_1*wd_1 + ... + nb_Nwd*wd_Nwd).
        Example: if Nwd = 2 and two states |0> and |1> satisfy E[1] ~ E[0] + n1 wd_1 + n2 wd_2, then resonances = [(0, (0, 0)), (1, (n1, n2))]
    E : array
        H0 spectrum. list or np.array of shape (D), with D Hilbert space dimension. Can be made of SymPy symbol.
    V : list
        list of drive harmonics (V_p, p_1, ..., p_Nwd) such that
        Each V_p is a numpy array of shape (D,D). Can be made of SymPy symbols.
        V(t) = sum_{p} e^{-i p.wd t} V_p. where p is the vector of harmonics.
        
        *Example:* if Nwd = 2 and V(t) = A e^{-i wd_1 t} + A^dag e^{i wd_1 t} + B e^{-i wd_2 t} + B^dag e^{-i wd_2 t} ; then give V = [(A, (1,0)), (B, (0,1))].
        Can give only one of the 2 conjugated harmonics p and -p (the other one is added automatically). In case of a single mode and single harmonics, can give the single harmonics V_{1} evolving at frequency e^{-i wd t}. 
    V0 : array (optional) 
        Static perturbation V_0. Can be given in V directly. Shape (D,D). Can be made of SymPy symbols.
    ref_state_and_n : tuple (optional) 
        tuple (ref_state, ref_n), reference state and harmonics from which we define the energy detuning. I.e. for (a,n) in resonances, we have E[a] = E[ref_state] + (ref_n - n)*wd + epsilon_{a,n} with a small epsilon_{a,n}. If None, pick the lowest energy resonant state (smallest a) with n = (0, ..., 0).
    printProcesses : bool (optional)
        if True, print the amplitude of each non-zero contribution and its type of Feynmann diagram (harmonics, virtual states, and DPT coefficient).
    analytics : bool (optional)
        if True, forces the use of symbolic calculation using the module SymPy.
    detuning_all_replicates : bool (optional)
        if True: method treating the detuning in the perturbation of all Floquet replicas (method 1).
        if False: method treating the detuning in the perturbation of resonant Floquet replicas only (method 2). Must be used if we consider two different resonances for the same state due to quasi-commensurabilities. (Example for Nwd=2 with wd_1/wd_2 ~ p1/p2 with p1,p2 integers ; then (a, (0, 0)) and (a, (-p2, p1)) are quasi-resonant for any state |a>).
    
    Return
    ---------
    Hba : float or SymPy symbol
        matrix element <b| H^{(r)} |a> of the r-th order effective Hamiltonian. Real number if a=b (diagonal element).
    """
    resonances = copy.deepcopy(resonances)
    if r == 0:
        return 0
    ## Copy arrays to prevent array modification outside the function.
    E = np.array(E)
    if E.dtype == 'int':
        ## Convert int to float to avoid integer to negative power
        E = np.array(E, dtype=float)
    ## Hilbert space dimension
    D = len(E)
    ## Convert perturbation to array if a single harmonics is given in the single mode case. V is a DxD matrix
    try:
        ## If a single harmonics is given in the single mode case. V is a DxD matrix
        V = np.array(V)
    except ValueError:
        ## If V is a list of [(Vp, p), ...] such that V[i][0] are the Vp
        pass
    ## Convert frequencies to numpy array
    wd = np.array(wd)
    if wd.shape == ():
        ## Case of a single independent frequency
        wd = np.array([wd])
    Nwd = len(wd)
    # print("wd:", wd)
    ## Convert resonances to a list of pairs (a, (n_1, ..., n_Nwd)) with tuple of harmonics
    for i, value in enumerate(resonances):
        if Nwd == 1:
            ## If we have a single mode (a, n). Do not change.
            assert type(value[1]) is int, "Number of modes in wd and resonances do not match."
            resonances[i] = (value[0], (value[1],))
        elif type(value[1]) in [list, tuple, np.ndarray]:
            ## If we have (a, [n_1, ..., n_Nwd]) --> (a, (n_1, ..., n_Nwd))
            assert len(value[1]) == Nwd, "Number of modes in wd and resonances do not match."
            resonances[i] = (value[0], tuple(value[1]))
        else:
            ## If we have (a, n_1, ..., n_Nwd) --> (a, (n_1, ..., n_Nwd))
            assert len(value[1:]) == Nwd, "Number of modes in wd and resonances do not match."
            resonances[i] = (value[0], tuple(value[1:]))
    # print("resonances:", resonances)
    ## Dictionnary of drive harmonics Vdict[ (p_1, ..., p_Nwd) ] = V_p
    Vdict = {}
    if Nwd == 1 and type(V) == np.ndarray and V.shape == (D,D):
        ## Single mode case with single harmonics V e^{-iwd t} given
        Vdict[(1,)] = V
        Vdict[(-1,)] = V.T.conj()
    else:
        assert type(V) in [list, tuple], "V must be a list of (Vp, p_1, ..., p_Nwd) where Vp is the DxD drive operator of harmonics p=(p_1, ..., p_Nwd), i.e. evolving at e^{-ip.wd t}."
        ## If several harmonics are given
        for value in V:
            if Nwd == 1:
                ## If we have a single mode value = (Vp, p) with p int
                assert type(value[1]) is int, "Number of modes in wd and V do not match."
                Vp = np.array(value[0])
                p = value[1]
                Vdict[(p,)] = Vp
                Vdict[(-p,)] = Vp.conj().T
            elif type(value[1]) in [list, tuple, np.ndarray]:
                ## If we have value = (Vp, [p_1, ..., p_Nwd])
                assert len(value[1]) == Nwd, "Number of modes in wd and resonances do not match."
                Vp = np.array(value[0])
                p = np.array(value[1])
                Vdict[tuple(p)] = Vp
                Vdict[tuple(-p)] = Vp.conj().T
            else:
                ## If we have value = (Vp, p_1, ..., p_Nwd)
                assert len(value[1:]) == Nwd, "Number of modes in wd and resonances do not match."
                Vp = np.array(value[0])
                p = np.array(value[1:])
                Vdict[tuple(p)] = Vp
                Vdict[tuple(-p)] = Vp.conj().T
    if V0 is not None:
        ## Add V0 if given
        Vdict[tuple(np.zeros(Nwd, dtype=int))] = np.array(V0)
    if tuple(np.zeros(Nwd, dtype=int)) not in Vdict.keys():
        ## Add zeroth harmonics if it does not exist. Useful to deal with the perturbative detuning later on.
        Vdict[tuple(np.zeros(Nwd, dtype=int))] = np.zeros((D,D))
    # print("Vdict:", Vdict)
    ## Force analytics if one of the data is SymPy symbol
    if wd.dtype == 'O' or E.dtype == 'O' or np.any([Vp.dtype == 'O' for p, Vp in Vdict.items()]):
        analytics = True
    if analytics:
        ## If analytics, convert energy and V to symbol to manipulate the detuning
        E = np.array(E, dtype=sy.Symbol)
        for p in Vdict.keys():
            Vdict[p] = np.array(Vdict[p], dtype=sy.Symbol)
    # print("analytics:", analytics)
    ## Extract harmonics of a and b if not given
    def extract_harmonics(k, name=""):
        if type(k) in [list, tuple, np.ndarray]:
            ## if give (k, n_1, ..., n_Nwd) or (k, (n_1, ..., n_Nwd))
            assert k[0] in [state for state, _ in resonances], f"{name} state must be in the degenerate subspace."
            if type(k[1]) in [list, tuple, np.ndarray]:
                ## if (k, (n_1, ..., n_Nwd)) is given
                assert len(k[1]) == Nwd, f"Number of modes in wd and {name} state do not match."
                ## Check if it is a resonant state
                assert np.any([k[0] == l and np.all(np.array(k[1]) == n) for l, n in resonances]), f"{name} state must be in the degenerate subspace."
                return k[0], tuple(k[1])
            else:
                ## if (k, n_1, ..., n_Nwd) is given
                assert len(k[1:]) == Nwd, f"Number of modes in wd and {name} state do not match."
                assert np.any([k[0] == l and np.all(np.array(k[1:]) == n) for l, n in resonances]), f"{name} state must be in the degenerate subspace."
                return k[0], tuple(k[1:])
        else:
            ## if only k is given
            assert k in [state for state, _ in resonances], f"{name} state must be in the degenerate subspace."
            ## Find the smallest n among resonant states (k, n) ; with norm |n_1| + ... + |n_Nwd|.
            nk = min([n if state == k else np.infty for state, n in resonances], key=lambda n: np.sum(np.abs(n)))
            return k, nk
    a, na = extract_harmonics(a, name="intial")
    b, nb = extract_harmonics(b, name="final")
    # print("a:", a, "na:", na)
    # print("b:", b, "nb:", nb)
    ## If a reference state for the detuning is not given, choose the lowest energy (in E ordering) and smallest associated n with norm |n_1| + ... + |n_Nwd|
    if ref_state_and_n is None:
        ref_state = min([state for state, n in resonances])
        ref_n = min([n if state == ref_state else np.infty for state, n in resonances], key=lambda n: np.sum(np.abs(n)))
    elif type(ref_state_and_n[1]) in [list, tuple, np.ndarray]:
        ## if (k, (n_1, ..., n_Nwd)) is given
        assert len(ref_state_and_n[1]) == Nwd, f"Number of modes in wd and ref state do not match."
        ## Check if it is a resonant state
        assert np.any([ref_state_and_n[0] == l and np.all(np.array(ref_state_and_n[1]) == n) for l, n in resonances]), f"ref state must be in the degenerate subspace."
        ref_state, ref_n = ref_state_and_n[0], tuple(ref_state_and_n[1])
    else:
        ## if (k, n_1, ..., n_Nwd) is given
        assert len(ref_state_and_n[1:]) == Nwd, f"Number of modes in wd and ref state do not match."
        ## Check if it is a resonant state
        assert np.any([ref_state_and_n[0] == l and np.all(np.array(ref_state_and_n[1:]) == n) for l, n in resonances]), f"ref state must be in the degenerate subspace."
        ref_state, ref_n = ref_state_and_n[0], tuple(ref_state_and_n[1:])
    # print("ref_state:", ref_state, "ref_n:", ref_n)
    ## Definition of ernergy and perturbation for the two different methods
    if detuning_all_replicates:
        ## Method of detuning in numerator only. Requires a change of energies
        ## Check if resonant states are unique
        resonant_states = [k for k, _ in resonances]
        assert len(resonant_states) == len(set(resonant_states)), "The resonant states must have a unique resonant harmonics to use detuning_all_replicates=True. Please use detuning_all_replicates=False if you consider a resonant state (k, n) and (k, n') with different harmonics n and n'."
        ## Dictionnary of unique detuning of each state, for the change of energy and perturbation
        detunings = {}
        for k, n in resonances:
            detunings[k] = E[k] - E[ref_state] - (np.array(n) - np.array(ref_n)).dot(wd)
        ## Change energies of resonant states
        for k in resonant_states:
            E[k] = E[k] - detunings[k]
        def Vtilde(p, k, l, n):
            """ Perturbation taking into account the detuning. Method of detuning in numerator only. """
            value = Vdict[p][k,l]
            if k == l and k in resonant_states and np.all(np.array(p)==0):
                value = value + detunings[k]
            return value
    else:
        ## Method of detuning in the resonant states (k, n) only, keeping E[k] in non-resonant virtual transtions
        ## Dictionnary of detuning for each resonant (k, n) ; not unique for each k.
        detunings = {}
        for k, n in resonances:
            detunings[(k, n)] = E[k] - E[ref_state] - (np.array(n) - np.array(ref_n)).dot(wd)
        def Vtilde(p, k, l, n):
            """ Perturbation taking into account the detuning. Method of detuning in resonances only. """
            value = Vdict[p][k,l]
            if k == l and (k, tuple(n)) in resonances and np.all(np.array(p)==0):
                value = value + detunings[(k, tuple(n))]
            return value
    # print(Vtilde(tuple(np.zeros(Nwd, dtype=int)), b, b, 1))
    result = sy.S.Zero if analytics else 0
    ## Load Degenerate Perturbation Theory (DPT) coefficients. Translate them in fraction for analytical expressions.
    if r >= 3:
        ## Extract the dictionary of DPT coefficients
        ## coeffsDPT[m] is the DPT coeff of the tuple m, where m[j] is the energy exponent of j-th step
        coeffsMatrix = np.loadtxt(f"{folder_DPT}/DPT_coefficients_order_{r}.txt")
        coeffsDPT = {}
        for line in coeffsMatrix:
            if analytics:
                numerator, denominator = line[-1].as_integer_ratio()
                coeffsDPT[tuple(line[:-1])] = Fraction(numerator, denominator)
            else:
                coeffsDPT[tuple(line[:-1])] = line[-1]
    if printProcesses:
        ## Count the number of non-zero processes
        nbProcesses = 0
    for p in itertools.product(*(Vdict.keys() for _ in range(r-1))):
        ## Iterate through all possible combinations of harmonics such that p[0] + ... + p[r-1] = nb - na
        ## p is a list of tuple of harmonics such that p[j] = (p[j]_1, ..., p[j]_Nwd) is the vector of harmonics of jth step.
        p = list(p)
        ## Add p[r-1] = nb - (p[0] + ... + p[r-2]) - na
        p.append(tuple(nb - np.sum(p, axis=0) - na))
        if p[-1] not in Vdict.keys():
            ## ignore if the required last harmonics is not contained in the Hamiltonian
            continue
        for k in itertools.product(*(np.arange(D) for _ in range(r-1))):
            ## Iterate through all possible virtual state k[j]
            ## Add last state b (to simplify the later loop)
            k = list(k)
            k.append(b)
            ## Find resonant steps
            R = []
            Rcomp = []
            for j in range(r-1):
                if (k[j], tuple(na + np.sum(p[:j+1], axis=0))) in resonances:
                    ## step j is resonant
                    # print("harmonics", p, "virtual states", k, "resonant step", j)
                    R.append(j)
                else:
                    ## Non resonant step
                    Rcomp.append(j)
            ## Compute process amplitude
            if len(R) == 0:
                ## No resonant step: simple contribution. (the distinction is useful if we want to ignore resonant processes, but it is not implemented here)
                contrib = Vtilde(p[0], k[0], a, na)
                ## Total number of photon absorbed after each step (in numpy array of sums and dot product)
                ptot = np.zeros(Nwd, dtype=int)
                for j in range(r-1):
                    ptot += p[j]
                    contrib = contrib*Vtilde(p[j+1], k[j+1], k[j], na + ptot)/(E[ref_state] + (na + ptot - ref_n).dot(wd) - E[k[j]])
                    # print("harmonics:", p, "virtual states:", k[:-1], "no resonant step.", "amplitude:", contrib)
                if printProcesses and ( (analytics and contrib != 0) or (not analytics and not np.isclose(contrib, 0, atol=1e-12)) ):
                    nbProcesses += 1
                    print("harmonics:", np.array(p)[:,0] if Nwd == 1 else p, "virtual states:", k[:-1], "no resonant step.", "amplitude:", contrib)
                result += contrib
            elif len(Rcomp) != 0:
                ## If there are resonant process: redistribute the exponents on the other processes if it remains some.
                ## List of energy exponents of each process
                m = np.zeros(r-1, dtype=int)
                ## m[j] = 0 for j in R. The exponents are redistributed between the j' in Rcomp
                ## m[j'] = 1 + dm[j'] , where sum_{j'\in Rcomp} dm[j'] = len(R). So dm is length len(Rcomp) with max value len(R)
                # print("harmonics:", p, "virtual states:", k[:-1], "Resonant steps:", R)
                for dm in itertools.product(*(np.arange(0, len(R)+1) for _ in range(len(Rcomp)-1))):
                    ## Extra exponent on all the non-resonant steps except the last one
                    ## iterate through all couples (dm[0], ..., dm[len(Rcomp)-2]) where dm[j]=0, ..., len(R), to fix the last element dm[len(Rcomp)-1] by hand to satisfy the condition sum dm = len(R).
                    ## (If there is only one non-resonant step len(Rcomp)=1, then this loop gives only the empty couple dm=(), such that we accurately fill it by len(R): dm=(len(R)) after filling)
                    dm = list(dm)
                    # print("dm before last filling:", dm)
                    ## Add the extra exponent to the last step to satisfy the total number of extra exponent
                    dm.append(len(R) - sum(dm))
                    # print("dm after last filling:", dm)
                    if dm[-1] < 0:
                        ## Ignore if the process requires a suppression of exponent on last step
                        ## (not entirely sure if it is needed)
                        continue
                    ## dm is now the list of extra exponent to add to each non-resonant transition
                    for l in range(len(Rcomp)):
                        m[Rcomp[l]] = 1 + dm[l]
                    ## Now m[j] is the exponent for the energy gap of jth step
                    ## Check if the contribution as a non-zero DPT coeff
                    try:
                        ## Compute the contribution
                        contrib = coeffsDPT[tuple(m)]*Vtilde(p[0], k[0], a, na)
                    except KeyError:
                        ## If the DPT coefficient does not exist (i.e. is zero), the skip this exponent list
                        # print(m, "no coeff")
                        continue
                    ## Total number of photon absorbed after each step (in numpy array of sums and dot product)
                    ptot = np.zeros(Nwd, dtype=int)
                    for j in range(r-1):
                        ptot += p[j]
                        contrib = contrib*Vtilde(p[j+1], k[j+1], k[j], na + ptot)*(E[ref_state] + (na + ptot - ref_n).dot(wd) - E[k[j]])**(-m[j])
                    if printProcesses and ( (analytics and contrib != 0) or (not analytics and not np.isclose(contrib, 0, atol=1e-12))):
                        nbProcesses += 1
                        print("harmonics:", np.array(p)[:,0] if Nwd == 1 else p, "virtual states:", k[:-1], "resonant steps:", R, "powers m:", m, "DPT coeff:", coeffsDPT[tuple(m)], "amplitude:", contrib)
                    result += contrib
    if printProcesses:
        print("Total number of processes:", nbProcesses)
    if a == b:
        ## Real diagonal term
        result = sy.re(result) if analytics else np.real(result)
    return result

def Heff_Floquet_multModes_summed(r, b, a, wd, resonances, E, V, V0=None, ref_state_and_n=None, printProcesses=False, analytics=False, detuning_all_replicates=True):
    """
    Effective Hamiltonian for multi-photon resonances of a quasi-periodic Hamiltonian (Nwd driving modes of frequency wd_1, ..., wd_Nwd).

    Sum of a specific matrix element sum_{l<=r} <b|Heff^{(r)}|a> of the effective Hamiltonian at order lower than r between two resonant states a and b.

    The time-dependent Hamiltonian must be decomposed as
        H(t) = H0 + V(t)
             = H0 + V0 + sum_{p_1>=1} ... sum_{p_Nwd>=1} ( V_{p_1,...,p_Nwd} e^{-i (p_1 wd_1 + ... + p_Nwd wd_Nwd) t} + V_{-p_1,...,-p_Nwd} e^{i (p_1 wd_1 + ... + p_Nwd wd_Nwd) t} )
        Where
            Nwd is the number of driving mode
            H0 = sum_{k=0}^{D-1} E[k] |k><k| is the un-driven Hamiltonian of eigen-energies E[k] and dimension D.
            V0 is an (optional) static perturbation.
            V_{p_1,...,p_Nwd} are the drive harmonics
    
    Parameters
    ---------
    r : integer
        order of perturbation theory.
    b, a: integer
        states of the matrix element <b|Heff|a>. Can also give the state and its harmonics (state, n_1, ..., n_Nwd) if several harmonics combinations are possible for the same state (relevent for the multi-mode case at higher orders).
    wd : array like
        list of mode frequencies [wd_1, ..., wd_Nwd] of length Nwd. Can be a scalar for single mode Nwd=1. Can be made of SymPy symbols.
    resonances : list
        resonant states and their orders. List of resonant states [(a, (na_1, ..., na_Nwd)), (b, (nb_1, ..., nb_Nwd)), ...] 
        such that E[a] - (na_1*wd_1 + ... + na_Nwd*wd_Nwd) ~ E[b] - (nb_1*wd_1 + ... + nb_Nwd*wd_Nwd).
        Example: if Nwd = 2 and two states |0> and |1> satisfy E[1] ~ E[0] + n1 wd_1 + n2 wd_2, then resonances = [(0, (0, 0)), (1, (n1, n2))]
    E : array
        H0 spectrum. list or np.array of shape (D), with D Hilbert space dimension. Can be made of SymPy symbol.
    V : list
        list of drive harmonics (V_p, p_1, ..., p_Nwd) such that
        Each V_p is a numpy array of shape (D,D). Can be made of SymPy symbols.
        V(t) = sum_{p} e^{-i p.wd t} V_p. where p is the vector of harmonics.
        
        *Example:* if Nwd = 2 and V(t) = A e^{-i wd_1 t} + A^dag e^{i wd_1 t} + B e^{-i wd_2 t} + B^dag e^{-i wd_2 t} ; then give V = [(A, (1,0)), (B, (0,1))].
        Can give only one of the 2 conjugated harmonics p and -p (the other one is added automatically). In case of a single mode and single harmonics, can give the single harmonics V_{1} evolving at frequency e^{-i wd t}. 
    V0 : array (optional) 
        Static perturbation V_0. Can be given in V directly. Shape (D,D). Can be made of SymPy symbols.
    ref_state_and_n : tuple (optional) 
        tuple (ref_state, ref_n), reference state and harmonics from which we define the energy detuning. I.e. for (a,n) in resonances, we have E[a] = E[ref_state] + (ref_n - n)*wd + epsilon_{a,n} with a small epsilon_{a,n}. If None, pick the lowest energy resonant state (smallest a) with n = (0, ..., 0).
    printProcesses : bool (optional)
        if True, print the amplitude of each non-zero contribution and its type of Feynmann diagram (harmonics, virtual states, and DPT coefficient).
    analytics : bool (optional)
        if True, forces the use of symbolic calculation using the module SymPy.
    detuning_all_replicates : bool (optional)
        if True: method treating the detuning in the perturbation of all Floquet replicas (method 1).
        if False: method treating the detuning in the perturbation of resonant Floquet replicas only (method 2). Must be used if we consider two different resonances for the same state due to quasi-commensurabilities. (Example for Nwd=2 with wd_1/wd_2 ~ p1/p2 with p1,p2 integers ; then (a, (0, 0)) and (a, (-p2, p1)) are quasi-resonant for any state |a>).
    
    Return
    ---------
    Hba_summed : float or SymPy symbol
        Sum of matrix element sum_{l<=r} <b| H^{(r)} |a> of the effective Hamiltonian up to order r. Real number if a=b (diagonal element).
    """
    Hba = Heff_Floquet_multModes(1, b, a, wd, resonances, E, V, V0, ref_state_and_n, printProcesses, analytics)
    for l in range(2, r+1):
        Hba += Heff_Floquet_multModes(l, b, a, wd, resonances, E, V, V0, ref_state_and_n, printProcesses, analytics)
    return Hba


def Heff_Floquet_bosonic(r, wb, wd, cb, cd, alpha, b_op, bd_op, ref_boson=None, analytics=False, printProcesses=False):
    """
    Effective Hamiltonian for resonantly driven bosonic system.

    The Hamiltonian in decomposed as 
        H(t) = sum_{j=0}^{Nb-1} wb[j] b_j^dag b_j + sum_{alpha} alpha[p, nu, mu] e^{-i p.wd t} b^dag^nu b^mu

    We consider a resonance constraint of the form
        cb.wb ~ cd.wd

    Parameters
    ---------
    r : integer
        order
    wb : 
        bosonic modes frequencies
    wd :
        driving frequencies
    cb :
        bosonic modes resonance vector
    cd :
        driving modes resonance vector
    alpha : dict
        coefficients considered of the perturbation. 
        alpha[p, nu, mu]
    b_op :
        bosonic annihilation operators
    bd_op :
        bosonic creation operators
    """
    if r == 0:
        return 0
    ## Convert boson quantities to numpy arrays to deal with single bosonic mode
    if np.array(wb).shape == ():
        ## Case of a single bosonic mode. Convert into 1 element array.
        wb = np.array([wb])
        cb = np.array([cb])
        b_op = np.array([b_op])
        bd_op = np.array([bd_op])
        ## Redefine the nu and mu of alpha as 1 element tuple
        alpha_new = {}
        for p, nu, mu in alpha:
            alpha_new[(p, (nu,), (mu,))] = alpha[p, nu, mu]
        alpha = dict(alpha_new)
    else:
        ## Multi bosonic modes case. Convert to array in case it is not already
        wb = np.array(wb)
        cb = np.array(cb)
        b_op = np.array(b_op)
        bd_op = np.array(bd_op)
    ## number of bosonic modes
    assert len(wb) == len(cb) == len(b_op) == len(bd_op), "Inconsistent number of bosonic modes."
    Nb = len(wb) ## Number of bosonic modes
    ## Same for driving quantities
    if np.array(wd).shape == ():
        wd = np.array([wd])
        cd = np.array([cd])
        ## Redefine the p of alpha as 1 element tuple
        alpha_new = {}
        for p, nu, mu in alpha:
            alpha_new[((p,), nu, mu)] = alpha[p, nu, mu]
        alpha = dict(alpha_new)
    else:
        ## Multi driving mode case. Convert to array in case it is not already
        wd = np.array(wd)
        cd = np.array(cd)
    Nd = len(wd) ## Number of driving modes
    ## TODO: add automatically harmonics (-p, mu, nu) if a (p, nu, mu) is given (optional but convenient)

    ## Check if analytical expression are used (SymPy symbols)
    ## TODO: automatized analytics from alpha values (optionnal but convenient, we can still use analytics argument)
    if not analytics:
        analytics = 'O' in [wd.dtype, wb.dtype]
    # print("analytics:", analytics)

    ## Load Degenerate Perturbation Theory (DPT) coefficients. Translate them in fraction for analytical expressions.
    if r >= 3:
        ## Extract the dictionary of DPT coefficients
        ## coeffsDPT[m] is the DPT coeff of the tuple m, where m[j] is the energy exponent of j-th step
        coeffsMatrix = np.loadtxt(f"{folder_DPT}/DPT_coefficients_order_{r}.txt")
        coeffsDPT = {}
        for line in coeffsMatrix:
            if analytics:
                ## use sympy fraction if we deal with analytical quantities
                numerator, denominator = line[-1].as_integer_ratio()
                coeffsDPT[tuple(line[:-1])] = Fraction(numerator, denominator)
            else:
                coeffsDPT[tuple(line[:-1])] = line[-1]

    ## Deal with the detuning in a ref boson
    epsilon = cb@wb - cd@wd
    # print("epsilon:", epsilon)
    if ref_boson is None:
        ## Find a ref boson such that cb[ref_boson] != 0
        ref_boson = np.where(cb != 0)[0][0]
        # print("ref_boson:", ref_boson)
    ## Remove epsilon from wb[ref_boson]
    wb[ref_boson] -= epsilon/cb[ref_boson]
    ## Add epsilon as epsilon/cb[ref_boson] b^dag_ref b_ref
    ## i.e. alpha_p,nu,mu with p=0 and nu[ref_boson] = 1 = mu[ref_boson]
    p, nu, mu = Nd*[0], Nb*[0], Nb*[0]
    nu[ref_boson] = 1
    mu[ref_boson] = 1
    p, nu, mu = tuple(p), tuple(nu), tuple(mu)
    if (p, nu, mu) in alpha:
        alpha[p, nu, mu] += epsilon/cb[ref_boson]
    else:
        alpha[p, nu, mu] = epsilon/cb[ref_boson]
    # print("alpha:", alpha)
    # print("wb:", wb)

    ## Construct the elementary bosonic operator strings
    b_strings_elmt = {}
    for _, nu, mu in alpha:
        ## Add b^dag^nu b^mu to the operator strings
        b_strings_elmt[nu, mu] = 1
        for i in range(Nb):
            b_strings_elmt[nu, mu] *= bd_op[i]**nu[i]
        for i in range(Nb):
            b_strings_elmt[nu, mu] *= b_op[i]**mu[i]
    # print(b_strings_elmt)

    ## Initialize result
    result = 0

    for list_p_mu_nu in itertools.product(alpha, repeat=r):
        ## Go through all existing (p_1, nu_1, mu_1), ..., (p_r, nu_r, mu_r)
        p = np.array([list_p_mu_nu[l][0] for l in range(r)])
        nu = np.array([list_p_mu_nu[l][1] for l in range(r)])
        mu = np.array([list_p_mu_nu[l][2] for l in range(r)])
        ptot = np.sum(p, axis=0)
        nutot = np.sum(nu, axis=0)
        mutot = np.sum(mu, axis=0)

        ## Check if ptot = N cd for some integer N in two steps:
        ## 1) check that ptot and cd are colinear with Caushy-Schwarz: u // v <=> (u.v)^2 = |u|^2 |v|^2
        ## 2) check proportionality factor is integer
        # print(ptot)
        if (ptot@cd)**2 != (ptot@ptot) * (cd@cd):
            ## Check that ptot and cd are colinear
            # print("no")
            continue
        N = (ptot@cd)/(cd@cd) ## Proportionality factor
        if int(N) != N: ## Check N is integer
            # print("N =", N, "no int")
            continue
        N = int(N)
        ## TODO: We can put a condition here to compute only the terms corresponding to a desired N (only the coupling N!=0 and not the Stark shift N=0 for example).
        # print(" ")
        # print("ptot", ptot)
        # print("yes N =", N)

        ## Check if nutot - mutot = N cb
        # print("nutot", nutot, "mutot", mutot)
        if np.any( nutot - mutot - N*cb ) != 0:
            # print("no boson")
            continue
        ## TODO: could add condition on the total bosonic string

        if r == 1:
            ## First order, one string and no energy penalty
            process = alpha[tuple(p[0]), tuple(nu[0]), tuple(mu[0])]*b_strings_elmt[tuple(nu[0]), tuple(mu[0])]
            if printProcesses:
                print(" ")
                print("(p, nu, mu):", p[0], nu[0], mu[0])
                print("process:", process)
            result += process

        if r > 1:
            ## Higher orders, compute energy denominators
            ## TODO: compute all the Delta_l for l=0,...,r-2
            lDelta = [p[0]@wd - (nu[0] - mu[0])@wb]
            for l in range(1, r-1):
                lDelta.append( lDelta[l-1] + p[l]@wd - (nu[l] - mu[l])@wb )
            lDelta = np.array(lDelta)
            # print("Deltas:", lDelta)

            ## Extract resonant steps
            R = np.where( lDelta == 0 )[0]
            Rcomp = np.where( lDelta != 0 )[0]
            if len(Rcomp) == 0:
                ## Impossible process if all steps are resonant
                continue
            # print(" ")
            # print("Res steps:", R)
            # print("non Res steps:", Rcomp)

            ## Case of no resonant steps, trivial coefficients
            if len(R) == 0:
                # print("No resonant step")
                energy_penalty = np.prod(lDelta)
                ## Compute operator string
                b_string = 1
                for l in range(r):
                    b_string = b_strings_elmt[tuple(nu[l]), tuple(mu[l])] * b_string
                ## Compute alpha product
                alpha_product = 1
                for l in range(r):
                    alpha_product *= alpha[tuple(p[l]), tuple(nu[l]), tuple(mu[l])]
                ## Process
                process = alpha_product/energy_penalty * b_string
                if printProcesses:
                    print(" ")
                    print("(p, nu, mu):", list_p_mu_nu)
                    print("No resonant steps")
                    print("process:", process)
                result += process

            ## Case with resonant steps. Uses energy powers m and DPT coefficients
            if len(R) != 0:
                # print("p:", *[p[l] for l in range(r)])
                # print("nu:", *[nu[l] for l in range(r)])
                # print("mu:", *[mu[l] for l in range(r)])
                for m_Rcomp in itertools.product(np.arange(1,r), repeat=len(Rcomp)-1):
                    ## Go through the powers m for the non-resonant steps except last
                    # print("m_Rcomp:", m_Rcomp)
                    ## Add last power such that sum m[l] = r-1
                    m_Rcomp_last = r - 1 - sum(m_Rcomp)
                    if m_Rcomp_last <= 0:
                        ## The last non-resonant step must have a non-zero power.
                        continue
                    m_Rcomp = list(m_Rcomp) + [ m_Rcomp_last ]
                    ## Define the list of power of all steps (zero for res steps, m_Rcomp for non-res steps)
                    m = np.zeros(r-1, dtype=int)
                    m[Rcomp] = m_Rcomp 
                    # print("m:", m)

                    ## Check if DPT coeff exist for this m
                    try:
                        cDPT = coeffsDPT[*m]
                    except KeyError:
                        # print("no DPT coeff")
                        continue
                    
                    ## Compute energy penalty
                    energy_penalty = np.prod(lDelta[Rcomp]**m[Rcomp])
                    ## Compute operator string
                    b_string = 1
                    for l in range(r):
                        b_string = b_strings_elmt[tuple(nu[l]), tuple(mu[l])] * b_string
                    ## Compute alpha product
                    alpha_product = 1
                    for l in range(r):
                        alpha_product *= alpha[tuple(p[l]), tuple(nu[l]), tuple(mu[l])]
                    ## Process
                    process = cDPT*alpha_product/energy_penalty * b_string
                    if printProcesses:
                        print(" ")
                        print("(p, nu, mu):", list_p_mu_nu)
                        print("Resonant steps:", R, "energy exponents:", m, "DPT coeff:", cDPT)
                        print("process:", process)
                    result += process

    return result