import operator
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from ._utils import State

# TODO: Implement a generic ode solver


def eulerstep(f, y0, h):
    return y0 + h * f(y0)


def rk4step(f: Callable[[State], State], y0: State, h: float) -> State:
    k1 = f(y0)
    k2 = f(y0 + h * k1 / 2)
    k3 = f(y0 + h * k2 / 2)
    k4 = f(y0 + h * k3)
    return y0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rkf45step(
    f: Callable[[State], State],
    y0: State,
    h: float,
    *,
    tolerance=5e-8,
    safety_factor=0.9,
) -> State:
    """Adopted from
        https://github.com/gwater/RungeKuttaFehlberg.jl/blob/master/src/RungeKuttaFehlberg.jl
    and
        https://www.wikiwand.com/en/Runge-Kutta-Fehlberg_method
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

    def l1_norm(t):
        abs = jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), t)
        return jax.tree_util.tree_reduce(operator.add, abs)

    def body(st):
        err, h, step = st
        h *= safety_factor * (tolerance / err) ** (1 / 5)
        step_rk4, step_rk5 = calc_steps(f, y0, h)
        err = l1_norm(step_rk4 - step_rk5)
        return err, h, step_rk5

    step_rk4, step_rk5 = calc_steps(f, y0, h)
    err = l1_norm(step_rk4 - step_rk5)

    _, _, step_rk5 = jax.lax.while_loop(
        lambda st: st[0] > tolerance,
        body,
        (err, h, step_rk5),
    )
    return y0 + step_rk5


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


def state_advancer(get_sysder, *args, tolerance):
    sysder = jax.jit(get_sysder(*args))
    adv = partial(rkf45step, sysder, tolerance=tolerance)
    return adv
