from collections import namedtuple
from collections.abc import Callable
from functools import cache, reduce
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree, Scalar

LocalTuple = PyTree[Scalar | Float[Array, " n"]]
LocalTupleFunction = Callable[[LocalTuple], Any]

Path = Callable[[Scalar], Float[Array, " n"]]
PathFunction = Callable[[Path], Callable[[Scalar], Any]]


@cache
def _namedtuple(typename: str, *field_names: str) -> type[LocalTuple]:
    return namedtuple(typename, field_names)


def Local(*args: Scalar | Float[Array, " n"]) -> LocalTuple:
    """Represents the state of a system at a given time."""
    _field_names = ["t", "pos", "v", "acc", "jerk", "snap", "crackle", "pop"]
    return _namedtuple("Local", *_field_names[: len(args)])(*args)


def partial(i: int, f: LocalTupleFunction) -> LocalTupleFunction:
    """Returns a function that computes the partial derivative
    of `f` with respect to the `i`th position in the local-tuple.
    """

    # TODO: Maybe this could be cleaned up?
    def p(local: LocalTuple) -> LocalTuple:
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


def p2r(local: LocalTuple) -> Float[Array, " n"]:
    """Converts polar coordinates to rectangular coordinates."""
    r, theta = local.pos
    return r * jnp.array([jnp.cos(theta), jnp.sin(theta)])


def r2p(local: LocalTuple) -> Float[Array, " n"]:
    """Converts rectangular coordinates to polar coordinates."""
    x, y = local.pos
    return jnp.array([jnp.sqrt(x**2 + y**2), jnp.arctan2(y, x)])


def Rx(theta: Scalar) -> Callable[[Float[Array, " n"]], Float[Array, " n"]]:
    """Rotation around the x-axis."""
    ct = jnp.cos(theta)
    st = jnp.sin(theta)

    def f(q: Float[Array, " n"]) -> Float[Array, " n"]:
        x, y, z = q
        return jnp.array([x, ct * y - st * z, st * y + ct * z])

    return f


def Ry(theta: Scalar) -> Callable[[Float[Array, " n"]], Float[Array, " n"]]:
    """Rotation around the y-axis."""
    ct = jnp.cos(theta)
    st = jnp.sin(theta)

    def f(q: Float[Array, " n"]) -> Float[Array, " n"]:
        x, y, z = q
        return jnp.array([ct * x + st * z, y, -st * x + ct * z])

    return f


def Rz(theta: Scalar) -> Callable[[Float[Array, " n"]], Float[Array, " n"]]:
    """Rotation around the z-axis."""
    ct = jnp.cos(theta)
    st = jnp.sin(theta)

    def f(q: Float[Array, " n"]) -> Float[Array, " n"]:
        x, y, z = q
        return jnp.array([ct * x - st * y, st * x + ct * y, z])

    return f


def factorial(n: Scalar) -> Scalar:
    return jax.lax.while_loop(
        lambda i: i[0] <= n,
        lambda i: (i[0] + 1, i[1] * i[0]),
        jnp.array([1.0, 1.0]),
    )[1]


def osculating_path(local: LocalTuple) -> Path:
    """Generates an osculating path with the given local-tuple components.

    Two paths that have the same local description up to the nth derivative
    are said to osculate with order n contact.

    This functions produces the osculating path using the truncated
    power series representation of the path up to order n, where n = len(`local`).

    I.e. O(t, q, v, a, ...)(t') = q + v * (t' - t) + a * (t' - t)^2 / 2 + ...
    """
    t, q, *dqs = local
    dqs = jnp.stack(dqs)

    def o(t_prime: Scalar) -> Float[Array, " n"]:
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


def Gamma_bar(f_bar: PathFunction) -> LocalTupleFunction:
    """Takes a path function and returns the corresponding local-tuple function."""
    return lambda local: f_bar(osculating_path(local))(local.t)


def Gamma(q: Path, n=3) -> Callable[[Scalar], LocalTuple]:
    assert n >= 2
    ds = [(D**i)(q) for i in range(1, n - 1)]

    def f(t):
        return Local(t, q(t), *[d(t) for d in ds])

    return f


def F2C(F: LocalTupleFunction) -> LocalTupleFunction:
    """Given a coordinate transformation (`F`)
    return the corresponding state transformation (`C`).
    """

    def C(local: LocalTuple) -> LocalTuple:
        n = len(local)

        def f_bar(q_prime: Path) -> Callable[[Scalar], LocalTuple]:
            # You enter q' coordinates into the transformation
            q = compose(F, Gamma(q_prime))
            # You differentiate
            return Gamma(q, n)

        # You abstract over position, velocity, etc.
        return Gamma_bar(f_bar)(local)

    return C


def Dt(F: LocalTupleFunction) -> LocalTupleFunction:
    """Return the Total Time Derivatice of F - a local-tuple function.

    DtF o Gamma[q] = D(F o Gamma[q])
    """

    def DtF(local: LocalTuple) -> Array | Scalar:
        n = len(local)

        def DF_on_path(q: Path) -> Callable[[Scalar], Array | Scalar]:
            return D(compose(F, Gamma(q, n)))

        return Gamma_bar(DF_on_path)(local)

    return DtF


def compose(*fs):
    def _compose2(f, g):
        return lambda *args, **kwargs: f(g(*args, **kwargs))

    if len(fs) == 1:
        return fs[0]
    return reduce(_compose2, fs)


def make_lagrangian(t: LocalTupleFunction, v: LocalTupleFunction) -> LocalTupleFunction:
    """Classical Lagrangian: L = T - V."""

    def f(local: LocalTuple) -> Scalar:
        return t(local) - v(local)

    return f


def Lagrangian_to_acceleration(L: LocalTupleFunction) -> LocalTupleFunction:
    p1L = partial(1, L)
    p2L = partial(2, L)
    p02L = partial(0, p2L)
    p12L = partial(1, p2L)
    p22L = partial(2, p2L)

    def f(local: LocalTuple) -> LocalTuple:
        a = p22L(local)
        b = p1L(local) - p12L(local) @ local.v - p02L(local)
        # NOTE: Figure out what to do when a is singular.
        return jnp.linalg.solve(a, b)

    return f


def Lagrangian_to_state_derivative(L: LocalTupleFunction) -> LocalTupleFunction:
    accel = Lagrangian_to_acceleration(L)

    def f(local: LocalTuple) -> LocalTuple:
        return Local(
            jnp.ones_like(local.t),
            local.v,
            accel(local),
        )

    return f


def Lagrangian_to_energy(L: LocalTupleFunction) -> LocalTupleFunction:
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


def Noether_integral(
    L: LocalTupleFunction, F_tilde: Callable[..., LocalTupleFunction]
) -> LocalTupleFunction:
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


def robust_norm(x: Float[Array, " n"], p=2) -> Scalar:
    """Taken from:
    https://timvieira.github.io/blog/post/2014/11/10/numerically-stable-p-norms/
    """

    a = jnp.abs(x).max()
    return a * jnp.linalg.norm(x / a, p)