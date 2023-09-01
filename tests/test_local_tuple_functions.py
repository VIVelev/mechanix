import jax
import jax.numpy as jnp

from mechanix import Dt, Euler_lagrange_operator, Local


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


def test_euler_lagrange_operator():
    def L_harmonic(local):
        # m = 1, k = 1
        _, q, v, _ = local
        return 0.5 * v @ v - 0.5 * q @ q

    E = Euler_lagrange_operator(L_harmonic)
    t = jnp.array(0.0)
    q = jnp.array([5.0])
    v = jnp.array([3.0])
    a = jnp.array([1.0])
    local = Local(t, q, v, a)

    d = E(local)
    d_expected = a + q  # ma = -kx
    assert jnp.allclose(d, d_expected)
