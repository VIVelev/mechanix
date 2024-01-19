import jax
import jax.numpy as jnp
from utils import find_next_crossing

from mechanix import (
    F2C,
    Lagrangian_to_energy,
    Lagrangian_to_state_derivative,
    State,
    compose,
    state_stepper,
)


def L3rd(m, V):
    """Lagrangian for the third body of mass `m` moving in a
    field derived from a time-varying gravitationa potential `V`.
    """

    def f(local):
        t, q, v = local
        return 0.5 * m * v.T @ v - V(t, q)

    return f


def get_Omega(a, GM0, GM1):
    """Get the rate at which the two bodies orbit their center of mass."""
    # From Kepler's third law:
    # Omega^2 * a^3 = G(M0 + M1)
    # Omega = sqrt(G(M0 + M1) / a^3)
    Omega = jnp.sqrt((GM0 + GM1) / a**3)
    return Omega


def get_a0_a1(a, GM0, GM1):
    """Get the distance b/w each body and the center of mass."""
    a0 = GM1 / (GM0 + GM1) * a
    a1 = GM0 / (GM0 + GM1) * a
    return a0, a1


def V(a, GM0, GM1, m):
    """Time-varying gravitational potential for a third body of mass `m`
    orbiting two bodies of mass `GM0` and `GM1` with distance b/w them `a`.
    """
    Omega = get_Omega(a, GM0, GM1)
    a0, a1 = get_a0_a1(a, GM0, GM1)

    def f(t, xy):
        x, y = xy
        # Get the position of the two bodies at time `t`
        x0 = -a0 * jnp.cos(Omega * t)
        y0 = -a0 * jnp.sin(Omega * t)
        x1 = a1 * jnp.cos(Omega * t)
        y1 = a1 * jnp.sin(Omega * t)

        # Calculate the distance b/w the third body and the two bodies
        r0 = jnp.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        r1 = jnp.sqrt((x - x1) ** 2 + (y - y1) ** 2)

        # Calculate the potential
        return -GM0 * m / r0 - GM1 * m / r1

    return f


def rotation(omega):
    """Rotation by `omega`."""

    def f(local):
        t, [x, y], _ = local
        return jnp.array(
            [
                [jnp.cos(omega * t), -jnp.sin(omega * t)],
                [jnp.sin(omega * t), jnp.cos(omega * t)],
            ]
        ) @ jnp.array([x, y])

    return f


# Simulation Parameters
m = 1
a = 1
GM0 = 1
GM1 = GM0 * 0.005

# Distance to the COM of the Earth-Moon system:
a0, a1 = get_a0_a1(a, GM0, GM1)
# Angular velocity of the Earth-Moon system:
Omega = get_Omega(a, GM0, GM1)

L = compose(L3rd(m, V(a, GM0, GM1, m)), F2C(rotation(Omega)))
Energy = Lagrangian_to_energy(L)
Jacobi = lambda local: -2 * Energy(local)


def R3BPsysder(a, m, GM0, GM1):
    Omega = get_Omega(a, GM0, GM1)
    L = compose(L3rd(m, V(a, GM0, GM1, m)), F2C(rotation(Omega)))
    return Lagrangian_to_state_derivative(L)


def R3BPmap(J, dt, int_eps, sec_eps, *, a, m, GM0, GM1):
    sysder = R3BPsysder(a, m, GM0, GM1)
    step = state_stepper(sysder, tolerance=int_eps)

    def sysmap(qv):
        y, ydot = qv
        st = section_to_state(J, y, ydot)

        cross_st = find_next_crossing(st, dt, step, sec_eps)
        y, ydot = cross_st[1][1], cross_st[2][1]
        return jnp.array([y, ydot])

    return sysmap


def section_to_state(J, y, ydot):
    j = Jacobi(State(jnp.array(0.0), jnp.array([0.0, y]), jnp.array([0.0, ydot])))
    d = J - j
    d = jax.lax.select(d >= 0, jnp.nan, d)

    xdot = jnp.sqrt(-d / m)
    return State(jnp.array(0.0), jnp.array([0.0, y]), jnp.array([xdot, ydot]))
