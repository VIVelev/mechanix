import jax
import jax.numpy as jnp

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

        cross_st = find_next_crossing(st, dt, adv, sec_eps)
        y, py = cross_st[1][1], cross_st[2][1]
        return jnp.array([y, py])

    return sysmap


def section_to_state(E, y, py):
    d = E - HHpotential(0, y) - (1 / 2) * py**2
    d = jax.lax.select(d <= 0, jnp.nan, d)

    px = jnp.sqrt(2 * d)
    return State(jnp.array(0.0), jnp.array([0.0, y]), jnp.array([px, py]))


def find_next_crossing(st, dt, adv, sec_eps):
    def crossed(st, next_st):
        x = st[1][0]
        next_x = next_st[1][0]
        has_crossed = jnp.logical_and(x < 0, next_x > 0)
        is_nan = jnp.logical_or(jnp.isnan(x), jnp.isnan(next_x))
        return jnp.logical_or(has_crossed, is_nan)

    st, next_st = jax.lax.while_loop(
        lambda val: jnp.logical_not(crossed(val[0], val[1])),
        lambda val: (val[1], adv(val[1], dt)),
        (st, adv(st, dt)),
    )

    return refine_crossing(st, adv, sec_eps)


def refine_crossing(st, adv, sec_eps):
    return jax.lax.while_loop(
        lambda st: jnp.abs(st[1][0]) > sec_eps,
        lambda st: adv(st, -st[1][0] / st[2][0]),
        st,
    )
