from collections import namedtuple
from functools import cache, reduce
from typing import Any, Callable

import jax
import jax.numpy as jnp

Scalar = int | float | complex
Array = jax.Array

Tuple = tuple[Scalar | Array, ...]
TupleFunction = Callable[[Tuple], Any]

Path = Callable[[Scalar], Tuple]
PathFunction = Callable[[Path], Callable[[Scalar], Any]]


@cache
def _namedtuple(typename, *field_names):
    return namedtuple(typename, field_names)


def Local(*args):
    """Represents the state of a system at a given time."""
    _field_names = ["t", "pos", "v", "acc", "jerk", "snap", "crackle", "pop"]
    return _namedtuple("Local", *_field_names[: len(args)])(*args)


def p2r(local):
    """Converts polar coordinates to rectangular coordinates."""
    r, theta = local.pos
    return r * jnp.array([jnp.cos(theta), jnp.sin(theta)])


def r2p(local):
    """Converts rectangular coordinates to polar coordinates."""
    x, y = local.pos
    return jnp.array([jnp.sqrt(x**2 + y**2), jnp.arctan2(y, x)])


def Rx(theta) -> Callable[[Array], Array]:
    """Rotation around the x-axis."""
    ct = jnp.cos(theta)
    st = jnp.sin(theta)

    def f(q):
        x, y, z = q
        return jnp.array([x, ct * y - st * z, st * y + ct * z])

    return f


def Ry(theta) -> Callable[[Array], Array]:
    """Rotation around the y-axis."""
    ct = jnp.cos(theta)
    st = jnp.sin(theta)

    def f(q):
        x, y, z = q
        return jnp.array([ct * x + st * z, y, -st * x + ct * z])

    return f


def Rz(theta) -> Callable[[Array], Array]:
    """Rotation around the z-axis."""
    ct = jnp.cos(theta)
    st = jnp.sin(theta)

    def f(q):
        x, y, z = q
        return jnp.array([ct * x - st * y, st * x + ct * y, z])

    return f


def osculating_path(local: Tuple) -> Path:
    """Generates an osculating path with the given local-tuple components.

    Two paths that have the same local description up to the nth derivative
    are said to osculate with order n contact.

    This functions produces the osculating path using the truncated
    power series representation of the path up to order n, where n = len(`local`).

    I.e. O(t, q, v, a, ...)(t') = q + v * (t' - t) + a * (t' - t)^2 / 2 + ...
    """
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


def Gamma_bar(f_bar: PathFunction) -> TupleFunction:
    """Takes a path function and returns the corresponding local-tuple function."""
    return lambda local: f_bar(osculating_path(local))(local.t)


def Gamma(q: Path, n=3) -> Callable[[Scalar], Tuple]:
    assert n >= 2
    ds = [(D**i)(q) for i in range(1, n - 1)]

    def f(t):
        return Local(t, q(t), *[d(t) for d in ds])

    return f


def F2C(F: TupleFunction) -> TupleFunction:
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


def Dt(F: TupleFunction) -> TupleFunction:
    """Return the Total Time Derivatice of F - a local-tuple function.

    DtF o Gamma[q] = D(F o Gamma[q])
    """

    def DtF(local):
        n = len(local)

        def DF_on_path(q):
            return D(compose(F, Gamma(q, n)))

        return Gamma_bar(DF_on_path)(local)

    return DtF


def compose(*fs):
    def _compose2(f, g):
        return lambda *args, **kwargs: f(g(*args, **kwargs))

    if len(fs) == 1:
        return fs[0]
    return reduce(_compose2, fs)


def make_lagrangian(t: TupleFunction, v: TupleFunction) -> TupleFunction:
    """Classical Lagrangian: L = T - V."""

    def f(local: Tuple) -> Scalar:
        return t(local) - v(local)

    return f


def Lagrangian_to_acceleration(L: TupleFunction) -> TupleFunction:
    jacL = jax.jacfwd(L)
    hessL = jax.jacrev(jacL)

    def f(local):
        jacL_ = jacL(local)
        hessL_ = hessL(local)

        p1L = jacL_[1]
        p02L = hessL_[2][0]
        p12L = hessL_[2][1]
        p22L = hessL_[2][2]

        a = p22L
        b = p1L - p12L @ local.v - p02L
        return jnp.linalg.pinv(a) @ b

    return f


def Lagrangian_to_state_derivative(L: TupleFunction) -> TupleFunction:
    accel = Lagrangian_to_acceleration(L)

    def f(local):
        return Local(
            jnp.ones_like(local.t),
            local.v,
            accel(local),
        )

    return f


def Lagrangian_to_energy(L: TupleFunction) -> TupleFunction:
    P = lambda x: jax.jacfwd(L)(x)[2]  # Momentum state function

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
    L: TupleFunction,
    F_tilde: Callable[..., TupleFunction],
) -> TupleFunction:
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
    P = lambda x: jax.jacfwd(L)(x)[2]  # Momentum state function
    DF_tilde = _jacfwd_parametric(F_tilde, *zeros, argnums=tuple(range(a)))
    return lambda local: P(local) @ jnp.stack(DF_tilde(local)).T


def robust_norm(x, p=2):
    """Taken from:
    https://timvieira.github.io/blog/post/2014/11/10/numerically-stable-p-norms/
    """

    a = jnp.abs(x).max()
    return a * jnp.linalg.norm(x / a, p)
