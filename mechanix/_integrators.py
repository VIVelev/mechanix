import operator
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from ._utils import State

# NOTE: Better abstraction could be created for integrators!

# NOTE: If a method's name starts with `adaptive_`, the method
# returns a tuple of (next_state, suggested_next_step_size)


def eulerstep(f: Callable[[State], State], y0: State, h: float) -> State:
    return y0 + h * f(y0)


def rk4step(f: Callable[[State], State], y0: State, h: float) -> State:
    k1 = f(y0)
    k2 = f(y0 + h * k1 / 2)
    k3 = f(y0 + h * k2 / 2)
    k4 = f(y0 + h * k3)
    return y0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def adaptive_rkf45step(
    f: Callable[[State], State],
    y0: State,
    h: float,
    *,
    tolerance=5e-8,
    safety_factor=0.9,
) -> State:
    """Adaptive step size Runge-Kutta-Fehlberg Method (RKF45).

    !! The actual step may be smaller then `h`

    Returns:
        y1: The state at the next time step
        next_h: The suggested next step size

    References:
        https://maths.cnam.fr/IMG/pdf/RungeKuttaFehlbergProof.pdf
        https://github.com/gwater/RungeKuttaFehlberg.jl/blob/master/src/RungeKuttaFehlberg.jl
    """

    def calc_steps(f, y0, h):
        k1 = f(y0)
        k2 = f(y0 + h * k1 / 4)
        k3 = f(y0 + h * (3 * k1 + 9 * k2) / 32)
        k4 = f(y0 + h * (1932 * k1 - 7200 * k2 + 7296 * k3) / 2197)
        k5 = f(y0 + h * (439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104))
        k6 = f(
            y0
            + h
            * (
                -8 * k1 / 27
                + 2 * k2
                - 3544 * k3 / 2565
                + 1859 * k4 / 4104
                - 11 * k5 / 40
            )
        )
        step_rk4 = h * (25 * k1 / 216 + 1408 * k3 / 2565 + 2197 * k4 / 4104 - k5 / 5)
        step_rk5 = h * (
            16 * k1 / 135
            + 6656 * k3 / 12825
            + 28561 * k4 / 56430
            - 9 * k5 / 50
            + 2 * k6 / 55
        )
        return step_rk4, step_rk5

    def tree_l1_norm(t):
        abs = jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), t)
        return jax.tree_util.tree_reduce(operator.add, abs)

    def suggest_h(err, h):
        return h * safety_factor * (tolerance / err) ** (1 / 5)

    def cond(st):
        err, _, _ = st
        return err > tolerance

    def body(st):
        err, h, step = st
        h = suggest_h(err, h)
        step_rk4, step_rk5 = calc_steps(f, y0, h)
        err = tree_l1_norm(step_rk4 - step_rk5)
        return err, h, step_rk5

    step_rk4, step_rk5 = calc_steps(f, y0, h)
    err = tree_l1_norm(step_rk4 - step_rk5)
    err, h, step_rk5 = jax.lax.while_loop(cond, body, (err, h, step_rk5))
    return y0 + step_rk5, suggest_h(err, h)


def ab2step(f: Callable[[State], State], y0: State, h: float) -> State:
    y1 = eulerstep(f, y0, h)
    return y1 + (h / 2) * (3 * f(y1) - f(y0))


def semi_implicit_eulerstep(f: Callable[[State], State], y0: State, h: float) -> State:
    t, q, p = y0
    _, _, dp = f(y0)
    next_p = p + h * dp
    _, dq, _ = f(State(t, q, next_p))
    next_q = q + h * dq
    return State(t + h, next_q, next_p)


def state_stepper(
    sysder: Callable[[State], State], tolerance=5e-8
) -> Callable[[State, float], State]:
    """Stepper makes a concretely one step in time."""
    return jax.jit(partial(adaptive_rkf45step, sysder, tolerance=tolerance))


def state_advancer(
    sysder: Callable[[State], State],
    tolerance=5e-8,
) -> Callable[[State, float], State]:
    """Advancer advaces the state to the specified final time,
    possibly making many steps.
    """

    step = state_stepper(sysder, tolerance=tolerance)

    @jax.jit
    def adv(y, tf):
        return jax.lax.while_loop(
            # The state is (next_y, suggested_next_h); see `adaptive_rkf45step`
            lambda y_h: y_h[0][0] < tf,
            lambda y_h: step(y_h[0], y_h[1]),
            step(y, tf - y[0]),
        )[0]

    return adv


def odeint(
    sysder: Callable[[State], State],
    tolerance=5e-8,
) -> Callable[[State, jax.Array], State]:
    adv = state_advancer(sysder, tolerance=tolerance)

    def body(y, t):
        next_y = adv(y, t)
        return next_y, next_y

    @jax.jit
    def integrate(y0, t):
        return jax.lax.scan(body, y0, t)[1]

    return integrate
