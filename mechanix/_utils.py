from collections import namedtuple
from functools import cache, reduce

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
import numpy as np


@cache
def _namedtuple(typename: str, *field_names: str) -> PyTree:
    return namedtuple(typename, field_names)


def Local(*args: Float[Array, " n"]) -> PyTree[Float[Array, " n"]]:
    """Represents the state of a system at a given time.
    The position and velocity can be any N-dimensional array,
    but they are flattend to vectors upon computation.
    """

    _field_names = ["t", "pos", "v", "acc", "jerk", "snap", "crackle", "pop"]
    return _namedtuple("Local", *_field_names[: len(args)])(*args)


def partial(i, f):
    """Returns a function the computes the partial derivative
    of f with respect to the ith argument.
    """

    # TODO: Maybe this could be cleaned up?
    def p(local):
        # Get the number of elements in the ith field
        d = np.prod(local[i].shape, dtype=int)
        # Produce a directional vector to the input in
        # the direction of the ith field
        tangent_in = jax.tree_map(
            lambda x: jnp.zeros_like(x)[None, ...].repeat(d, 0), local
        )
        field_name = tangent_in._fields[i]
        field_value = jnp.eye(d) if field_name != "t" else jnp.array([1.0])
        tangent_in = tangent_in._replace(**{field_name: field_value})
        # Perform a Vector-Jacobian product
        vmap_jvp = jax.vmap(jax.jvp, in_axes=(None, None, 0))
        _, tangent_out = vmap_jvp(f, (local,), (tangent_in,))
        return tangent_out.T.squeeze()

    return p


def p2r(local):
    """Converts polar coordinates to rectangular coordinates."""
    r, theta = local.pos
    return r * jnp.array([jnp.cos(theta), jnp.sin(theta)])


def r2p(local):
    """Converts rectangular coordinates to polar coordinates."""
    x, y = local.pos
    return jnp.array([jnp.sqrt(x**2 + y**2), jnp.arctan2(y, x)])


def Rx(theta):
    ct = jnp.cos(theta)
    st = jnp.sin(theta)

    def f(q):
        x, y, z = q
        return jnp.array([x, ct * y - st * z, st * y + ct * z])

    return f


def Ry(theta):
    ct = jnp.cos(theta)
    st = jnp.sin(theta)

    def f(q):
        x, y, z = q
        return jnp.array([ct * x + st * z, y, -st * x + ct * z])

    return f


def Rz(theta):
    ct = jnp.cos(theta)
    st = jnp.sin(theta)

    def f(q):
        x, y, z = q
        return jnp.array([ct * x - st * y, st * x + ct * y, z])

    return f


def factorial(n):
    return jax.lax.while_loop(
        lambda i: i[0] <= n,
        lambda i: (i[0] + 1, i[1] * i[0]),
        (1, 1),
    )[1]


def osculating_path(local):
    t, q, *dqs = local
    dqs = jnp.stack(dqs)

    def o(t_prime):
        dt = t_prime - t

        def body(carry, curr):
            n, coeff = carry
            coeff *= dt / n
            return (n + 1, coeff), coeff * curr

        _, terms = jax.lax.scan(body, (1, 1), dqs)
        return q + terms.sum(0)

    return o


class _operator:
    def __init__(self, name, f):
        self.name = name
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __pow__(self, n):
        return _operator(f"D^{n}", compose(*([self.f] * n)))

    def __repr__(self):
        return f"{self.name} <operator>"


D = _operator("D", lambda f: jax.jacfwd(f))


def Gamma_bar(f_bar):
    return lambda local: f_bar(osculating_path(local))(local.t)


def Gamma(q, n=3):
    assert n >= 2
    ds = [(D**i)(q) for i in range(1, n - 1)]

    def f(t):
        return Local(t, q(t), *[d(t) for d in ds])

    return f


def F2C(F):
    """Given a coordinate transformation (`F`)
    return the corresponding state transformation (`C`).
    """

    def C(local):
        n = len(local)

        def f_bar(q_prime):
            # You enter q' coordinates into the transformation
            q = compose(F, Gamma(q_prime))
            # You differentiate
            return Gamma(q, n)

        # You abstract over position, velocity, etc.
        return Gamma_bar(f_bar)(local)

    return C


def compose(*fs):
    def _compose2(f, g):
        return lambda *args, **kwargs: f(g(*args, **kwargs))

    if len(fs) == 1:
        return fs[0]
    return reduce(_compose2, fs)


def Lagrange_equations(lagrangian):
    def f(q):
        Dp2L = D(compose(partial(2, lagrangian), Gamma(q)))
        p1L = compose(partial(1, lagrangian), Gamma(q))

        def g(t):
            return Dp2L(t) - p1L(t)

        return g

    return f


def make_lagrangian(t, v):
    """Classical Lagrangian: L = T - V."""

    def f(local):
        return t(local) - v(local)

    return f


def Lagrangian_to_acceleration(L):
    p1L = partial(1, L)
    p2L = partial(2, L)
    p02L = partial(0, p2L)
    p12L = partial(1, p2L)
    p22L = partial(2, p2L)

    def f(local):
        a = p22L(local)
        b = p1L(local) - p12L(local) @ local.v - p02L(local)
        # NOTE: Figure out what to do when a is singular.
        jax.debug.print("det(a): {}", jnp.linalg.det(a))
        # return jnp.linalg.solve(a, b)
        return jnp.linalg.pinv(a) @ b

    return f


def Lagrangian_to_state_derivative(L):
    accel = Lagrangian_to_acceleration(L)

    def f(local):
        return Local(
            jnp.ones_like(local.t),
            local.v,
            accel(local),
        )

    return f


def Lagrangian_to_energy(L):
    P = partial(2, L)

    def f(local):
        return P(local) @ local.v - L(local)

    return f


def arity(f):
    return f.__code__.co_argcount


def _jacfwd_parametric(fun, *args, **kwargs):
    def df(*df_args, **df_kwargs):
        return jax.jacfwd(lambda *a: fun(*a)(*df_args, **df_kwargs), **kwargs)(*args)

    return df


def Noether_integral(L, F_tilde):
    """Noether Theorem Support

    F-tilde is a parametric coordinate transformation that given parameters takes
    a state and returns transformed coordinates. F-tilde may take an arbitrary
    number of real-valued parameters. F-tilde applied to zeros is the coordinate
    selector: It takes a state and returns the coordinates. The hypothesis of
    Noether's theorem is that the Lagrangian is invariant under the
    transformation for all values of the parameters
    """
    a = arity(F_tilde)
    zeros = (jnp.zeros(()),) * a
    P = partial(2, L)
    DF_tilde = _jacfwd_parametric(F_tilde, *zeros, argnums=tuple(range(a)))
    return lambda local: P(local) @ jnp.stack(DF_tilde(local)).T


def robust_norm(x, p=2):
    """Taken from:
    https://timvieira.github.io/blog/post/2014/11/10/numerically-stable-p-norms/
    """

    a = jnp.abs(x).max()
    return a * jnp.linalg.norm(x / a, p)
