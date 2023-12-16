from typing import Callable

import jax
import jax.numpy as jnp

from ._utils import Array


def cartesian_product(xs, ys):
    """Computes the cartesian product of two arrays."""
    xs, ys = jnp.array(xs), jnp.array(ys)
    len_xs, len_ys = len(xs), len(ys)
    maxlen = max(len_xs, len_ys)
    if len_xs < maxlen:
        xs = jnp.tile(xs, maxlen // len_xs)
    else:
        ys = jnp.tile(ys, maxlen // len_ys)

    assert len(xs) == len(ys), "len(xs), len(ys) should not be co-prime"
    return jnp.transpose(jnp.array([jnp.tile(xs, len(ys)), jnp.repeat(ys, len(xs))]))


def evolve_map(xs, ys, sysmap: Callable[[Array], Array], n: int):
    sysmap = jax.vmap(sysmap)
    init_samples = cartesian_product(xs, ys)
    evolution = jnp.empty((0, 0, 2), dtype=float)

    def body(carry, _):
        out = sysmap(carry)
        return out, out

    _, evolution = jax.lax.scan(body, init_samples, jnp.arange(n), length=n)
    return evolution
