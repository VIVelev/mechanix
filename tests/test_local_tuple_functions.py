import jax
import jax.numpy as jnp

from mechanix import Dt, Local


def test_total_time_derivative():
    def F(local):
        t, q, v = local
        return Local(3 * t, q, v)

    DtF = Dt(F)
    t = jnp.array(1.0)
    q = jnp.array([1.0, 2.0, 3.0])
    v = jnp.array([1.0, 2.0, 3.0])
    local = Local(t, q, v)

    d = DtF(local)
    d_expected = Local(3 * jnp.ones_like(t), v, jnp.zeros_like(v))
    assert jax.tree_util.tree_reduce(
        lambda x, y: x and y,
        jax.tree_map(jnp.allclose, d, d_expected),
    )

    d_expected_false = Local(jnp.zeros_like(t), q, jnp.ones_like(v))
    assert not jax.tree_util.tree_reduce(
        lambda x, y: x and y,
        jax.tree_map(jnp.allclose, d, d_expected_false),
    )
