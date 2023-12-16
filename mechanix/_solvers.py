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


def eulerstep(f, y0, h):
    return y0 + h * f(y0)


def state_advancer(get_dstate, *args):
    # JIT the derivative function
    dstate = jax.jit(get_dstate(*args))

    def advance(y0, dt, *, n=2):
        h = dt / jnp.array(n)
        return jax.lax.fori_loop(0, n + 1, lambda _, y: rk4step(dstate, y, h), y0)

    return advance
