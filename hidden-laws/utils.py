from functools import reduce

import jax
import jax.numpy as jnp


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


def get_in(col, path):
    return reduce(lambda c, i: c[i], path, col)


def find_next_crossing(st, dt, step, sec_eps, *, cross_path=[1, 0]):
    def crossed(st, next_st):
        x = get_in(st, cross_path)
        next_x = get_in(next_st, cross_path)
        has_crossed = jnp.logical_and(x < 0, next_x > 0)
        is_nan = jnp.logical_or(jnp.isnan(x), jnp.isnan(next_x))
        return jnp.logical_or(has_crossed, is_nan)

    def cond(prev_next):
        st, (next_st, _) = prev_next
        return jnp.logical_not(crossed(st, next_st))

    def body(prev_next):
        st, (next_st, suggested_h) = prev_next
        st, (next_st, suggested_h) = next_st, step(next_st, suggested_h)
        return st, (next_st, suggested_h)

    st, _ = jax.lax.while_loop(cond, body, (st, step(st, dt)))
    return refine_crossing(st, step, sec_eps, cross_path=cross_path)


def refine_crossing(st, step, sec_eps, *, cross_path=[1, 0]):
    deriv_path = [cross_path[0] + 1] + cross_path[1:]

    def cond(st):
        return jnp.abs(get_in(st, cross_path)) > sec_eps

    def body(st):
        st, _ = step(st, -get_in(st, cross_path) / get_in(st, deriv_path))
        return st

    return jax.lax.while_loop(cond, body, st)
