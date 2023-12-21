from typing import Callable

import jax
import jax.numpy as jnp

from ._utils import State

# TODO: Implement a generic ode solver


def rk4step(f: Callable[[State], State], y0: State, h: float) -> State:
    k1 = f(y0)
    k2 = f(y0 + h * k1 / 2)
    k3 = f(y0 + h * k2 / 2)
    k4 = f(y0 + h * k3)
    return y0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def semi_implicit_eulerstep(f: Callable[[State], State], y0: State, h: float) -> State:
    t, q, p = y0
    _, _, dp = f(y0)
    next_p = p + h * dp
    _, dq, _ = f(State(t, q, next_p))
    next_q = q + h * dq
    return State(t + h, next_q, next_p)


def eulerstep(f, y0, h):
    return y0 + h * f(y0)


def state_advancer(get_dstate, *args):
    # JIT the derivative function
    dstate = jax.jit(get_dstate(*args))

    def advance(y0, dt, *, n=10):
        h = dt / jnp.array(n)
        return jax.lax.fori_loop(0, n, lambda _, y: rk4step(dstate, y, h), y0)

    return advance
