import jax
import jax.numpy as jnp
from utils import find_next_crossing

from mechanix import Hamiltonian_to_state_derivative, State, state_advancer


def HHpotential(x, y):
    """The Henon-Heiles potential energy."""
    return (1 / 2) * (x**2 + y**2) + x**2 * y - (1 / 3) * y**3


def HHham(local):
    """The Henon-Heiles Hamiltonian."""
    _, [x, y], [px, py] = local
    return (1 / 2) * (px**2 + py**2) + HHpotential(x, y)


def HHsysder():
    return Hamiltonian_to_state_derivative(HHham)


def HHmap(E, dt, sec_eps):
    adv = state_advancer(HHsysder)

    def sysmap(qp):
        y, py = qp
        st = section_to_state(E, y, py)

        cross_st = find_next_crossing(st, dt, adv, sec_eps, cross_path=[1, 0])
        y, py = cross_st[1][1], cross_st[2][1]
        return jnp.array([y, py])

    return sysmap


def section_to_state(E, y, py):
    d = E - HHpotential(0, y) - (1 / 2) * py**2
    d = jax.lax.select(d <= 0, jnp.nan, d)

    px = jnp.sqrt(2 * d)
    return State(jnp.array(0.0), jnp.array([0.0, y]), jnp.array([px, py]))
