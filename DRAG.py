import sympy as sp
from qutip import Qobj, basis
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar


def sigma_x_ij(i, j, d):
    ei = basis(d, i)
    ej = basis(d, j)
    return ej * ei.dag() + ei * ej.dag()


def sigma_y_ij(i, j, d):
    ei = basis(d, i)
    ej = basis(d, j)
    return -1j * ei * ej.dag() + 1j * ej * ei.dag()


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

def gauss(tsym, A, tg, sigma, symbolic):
    xp = sp if symbolic else np
    B_ratio = xp.exp(-(tg**2) / (8 * sigma**2))
    def pulse_shape(t):
      return A * (xp.exp(-((t - tg / 2) ** 2) / (2 * sigma**2)) - B_ratio)
    def pulse_derivative(t):
        return (
            -A
            * (t - tg / 2)
            * xp.exp(-((t - tg / 2) ** 2) / (2 * sigma**2))
            / sigma**2)
    return pulse_shape(tsym), pulse_derivative(tsym)


def pulse_functions(
    tg, Delta, rrr, sigma_r=0.3, pulse='gauss', symbolic=None, order=5
):
  if symbolic is None:
    symbolic = any(_is_symbolic(v) for v in (tg, Delta, rrr, sigma_r))

  sigma = sigma_r * tg
  xp = sp if symbolic else np
  pi = sp.pi if symbolic else np.pi

  if pulse == 'gauss':
    B_ratio = xp.exp(-(tg**2) / (8 * sigma**2))

    def pulse_shape(t, A):
      return A * (xp.exp(-((t - tg / 2) ** 2) / (2 * sigma**2)) - B_ratio)

    def pulse_derivative(t, A):
      return (
          -A
          * (t - tg / 2)
          * xp.exp(-((t - tg / 2) ** 2) / (2 * sigma**2))
          / sigma**2
      )

  elif pulse == 'tanh':

    def pulse_shape(t, A):
      return A * xp.tanh(t / sigma) * xp.tanh((tg - t) / sigma)

    def pulse_derivative(t, A):
      x, y = t / sigma, (tg - t) / sigma
      return (
          A / sigma * (xp.tanh(y) / xp.cosh(x) ** 2 - xp.tanh(x) / xp.cosh(y) ** 2)
      )

  else:
    raise ValueError(f"Unsupported pulse '{pulse}'. Use 'gauss' or 'tanh'.")

  if symbolic:
    if pulse == 'gauss':
      t_prime = sp.Symbol('t_prime')
      A = pi / sp.Integral(pulse_shape(t_prime, 1), (t_prime, 0, tg)).doit()
    elif pulse == 'tanh':
      # Analytische Stammfunktion nutzen, um unlösbare sp.Integral-Objekte zu vermeiden
      integral_val = (
          2 * sigma * sp.log(sp.cosh(tg / sigma)) / sp.tanh(tg / sigma) - tg
      )
      A = pi / integral_val
  else:
    A = root_scalar(
        lambda A: quad(lambda t: pulse_shape(t, A), 0, tg)[0] - pi,
        bracket=[0, 5],
        method='brentq',
    ).root

  def ep_pi(t, args=None):
    return pulse_shape(t, A)

  def ep_pi_tder(t, args=None):
    return pulse_derivative(t, A)

  def delta1(t, args=None):
    ep = ep_pi(t)
    if order == 4:
      return (rrr**2 - 4) * ep**2 / (4 * Delta)
    elif order == 5:
      return (rrr**2 - 4) * ep**2 / (4 * Delta) - (
          rrr**4 - 7 * rrr**2 + 12
      ) * ep**4 / (16 * Delta**3)
    else:
      raise ValueError('order must be 4 or 5')

  def epsilonx(t, args=None):
    ep = ep_pi(t)
    if order == 4:
      return ep
    elif order == 5:
      return (
          ep
          + (rrr**2 - 4) * ep**3 / (8 * Delta**2)
          - (13 * rrr**4 - 76 * rrr**2 + 112) * ep**5 / (128 * Delta**4)
      )
    else:
      raise ValueError('order must be 4 or 5')

  def epsilony(t, args=None):
    ep = ep_pi(t)
    ep_dot = ep_pi_tder(t)
    if order == 4:
      return -ep_dot / Delta
    elif order == 5:
      return -ep_dot / Delta + 33 * (rrr**2 - 2) * ep**2 * ep_dot / (
          24 * Delta**3
      )
    else:
      raise ValueError('order must be 4 or 5')

  return epsilonx, epsilony, delta1, ep_pi, ep_pi_tder



def h_target(Delta, rrr, tg, sigma_r, pulse='gauss'):
    symbolic = any(_is_symbolic(v) for v in (Delta, rrr, tg, sigma_r))
    if symbolic:
        return h_target_symbolic(Delta, rrr, tg, sigma_r, pulse)
    return h_target_numeric(Delta, rrr, tg, sigma_r, pulse)


def h_target_symbolic(Delta, rrr, tg, sigma_r, pulse='gauss', t=0, order = 5):
    H = sp.diag(0, 0, Delta)
    delta1_mat = sp.diag(0, 1, 0)
    
    # Entpacken der benötigten Pulse-Funktionen
    epsilonx, epsilony, delta1, _, _ = pulse_functions(tg, Delta, rrr, sigma_r, pulse, symbolic=True, order = order)

    H_total = H + delta1_mat * delta1(t)

    for i in range(2):
        lambda_i = 1 if i == 0 else rrr
        H_total += _sigma_op(i, i + 1, 3, symbolic=True) * lambda_i * sp.Rational(1, 2) * epsilonx(t)
        H_total += _sigma_op(i, i + 1, 3, kind='y', symbolic=True) * lambda_i * sp.Rational(1, 2) * epsilony(t)

    return H_total, (epsilonx(t), epsilony(t), delta1(t))


def h_target_numeric(Delta, rrr, tg, sigma_r, pulse='gauss'):
    H = Qobj(np.diag([0, 0, Delta]))
    delta1_mat = Qobj(np.diag([0, 1, 0]))
    
    epsilonx, epsilony, delta1, _, _ = pulse_functions(tg, Delta, rrr, sigma_r, pulse, symbolic=False)

    Vx, Vy = [], []

    for i in range(2):
        lambda_i = 1 if i == 0 else rrr
        Vx.append([Qobj(_sigma_op(i, i + 1, 3, symbolic=False)) * lambda_i / 2, epsilonx])
        Vy.append([Qobj(_sigma_op(i, i + 1, 3, kind='y', symbolic=False)) * lambda_i / 2, epsilony])

    return [H, [delta1_mat, delta1], *Vx, *Vy], (epsilonx, epsilony, delta1)


def h_target_symbolic_abstract(
    Delta, rrr, tg, sigma_r, pulse='gauss', t=None, order=5
):
  if t is None:
    t = sp.Symbol('t', real=True)

  H = sp.diag(0, 0, Delta)
  delta1_mat = sp.diag(0, 1, 0)

  # 1. Abstrakte SymPy-Symbole für ep_pi und ep_pi_tder separat definieren
  ep_pi = sp.Function(r'{\mathcal E}_\pi', real=True)(t)
  ep_pi_tder = sp.Symbol(r'\dot{\mathcal E}_\pi', real=True)

  # 2. Puls-Kompensationsterme abstrakt aufbauen
  if order == 4:
    delta1_expr = (rrr**2 - 4) * ep_pi**2 / (4 * Delta)
    eps_x_expr = ep_pi
    eps_y_expr = -ep_pi_tder / Delta
  elif order == 5:
    delta1_expr = (rrr**2 - 4) * ep_pi**2 / (4 * Delta) - (
        rrr**4 - 7 * rrr**2 + 12
    ) * ep_pi**4 / (16 * Delta**3)
    eps_x_expr = (
        ep_pi
        + (rrr**2 - 4) * ep_pi**3 / (8 * Delta**2)
        - (13 * rrr**4 - 76 * rrr**2 + 112) * ep_pi**5 / (128 * Delta**4)
    )
    eps_y_expr = -ep_pi_tder / Delta + 33 * (rrr**2 - 2) * ep**2 * ep_pi_tder / (
        24 * Delta**3
    )
  else:
    raise ValueError('order must be 4 or 5')

  # 3. Matrix H_total zusammenbauen
  H_total = H + delta1_mat * delta1_expr

  for i in range(2):
    lambda_i = 1 if i == 0 else rrr
    H_total += (
        _sigma_op(i, i + 1, 3, symbolic=True)
        * lambda_i
        * sp.Rational(1, 2)
        * eps_x_expr
    )
    H_total += (
        _sigma_op(i, i + 1, 3, kind='y', symbolic=True)
        * lambda_i
        * sp.Rational(1, 2)
        * eps_y_expr
    )

  # 4. Explizite Ausdrücke für die Ersetzung generieren
  _, _, _, ep_pi_func, ep_pi_tder_func = pulse_functions(
      tg, Delta, rrr, sigma_r, pulse, symbolic=True, order=order
  )

  subs_dict = {ep_pi: ep_pi_func(t), ep_pi_tder: ep_pi_tder_func(t)}

  return H_total, subs_dict