from functools import reduce
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from tree_math import VectorMixin

Scalar = int | float | complex
Array = jax.Array


def principal(q, *, cutoff=np.pi):
    return q - 2 * cutoff * np.round(q / (2 * cutoff))


@jax.tree_util.register_pytree_node_class
class State(VectorMixin):
    """Generic state representation."""

    def __init__(self, *args):
        self.tup = args

    def __repr__(self):
        return f"State{self.tup}"

    def __getitem__(self, i):
        return self.tup[i]

    def __iter__(self):
        return iter(self.tup)

    def __len__(self):
        return len(self.tup)

    def tree_flatten(self):
        return self.tup, None

    @classmethod
    def tree_unflatten(cls, _, tup):
        return cls(*tup)


StateFunction = Callable[[State], Any]
Path = Callable[[Scalar], Array]
PathFunction = Callable[[Path], Callable[[Scalar], Any]]


def p2r(local: State):
    """Converts polar coordinates to rectangular coordinates."""
    _, [r, theta], *_ = local
    return r * jnp.array([jnp.cos(theta), jnp.sin(theta)])


def r2p(local: State):
    """Converts rectangular coordinates to polar coordinates."""
    _, [x, y], *_ = local
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


def osculating_path(local: State) -> Path:
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


def Gamma_bar(f_bar: PathFunction) -> StateFunction:
    """Takes a path function and returns the corresponding local-tuple function."""
    return lambda local: f_bar(osculating_path(local))(local[0])


def Gamma(q: Path, n=3) -> Callable[[Scalar], State]:
    assert n >= 2
    ds = [(D**i)(q) for i in range(1, n - 1)]

    def f(t):
        return State(t, q(t), *[d(t) for d in ds])

    return f


def F2C(F: StateFunction) -> StateFunction:
    """Given a coordinate transformation (`F`)
    return the corresponding state transformation (`C`).
    """

    def C(local: State):
        n = len(local)

        def f_bar(q_prime):
            # You enter q' coordinates into the transformation
            q = compose(F, Gamma(q_prime))
            # You differentiate
            return Gamma(q, n)

        # You abstract over position, velocity, etc.
        return Gamma_bar(f_bar)(local)

    return C


# BUG: Hamiltonian is not same as a Lagrangian?
def F2CH(F: StateFunction) -> StateFunction:
    """Given a coordinate transformation (`F`)
    return the corresponding state transformation (`C`) for a Hamiltonian system.
    """

    # NOTE: Could this be astracted as `F2C`?

    jacF = jax.jacfwd(F)

    def C(local: State):
        t, _, p = local
        p1F = jacF(local)[1]
        return State(t, F(local), p @ jnp.linalg.pinv(p1F))

    return C


def F2K(F: StateFunction) -> StateFunction:
    """If the transformation `F` is time varying
    the Hamiltonian must be adjusted by adding a correction
    to the composition of the Hamiltonian and the transformation.
    """

    jacF = jax.jacfwd(F)

    def K(local: State):
        _, _, p = local
        jacF_ = jacF(local)
        p0F = jacF_[0]
        p1F = jacF_[1]
        return -(p @ jnp.linalg.pinv(p1F)) @ p0F

    return K


def Dt(F: StateFunction) -> StateFunction:
    """Return the Total Time Derivatice of F - a local-tuple function.

    DtF o Gamma[q] = D(F o Gamma[q])
    """

    def DtF(local: State):
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


def make_lagrangian(t: StateFunction, v: StateFunction) -> StateFunction:
    """Classical Lagrangian: L = T - V."""

    def f(local: State) -> Scalar:
        return t(local) - v(local)

    return f


def Lagrangian_to_acceleration(L: StateFunction) -> StateFunction:
    jacL = jax.jacrev(L)
    hessL = jax.jacfwd(jacL)

    def f(local: State):
        _, _, v = local

        jacL_ = jacL(local)
        hessL_ = hessL(local)

        p1L = jacL_[1]
        p02L = hessL_[2][0]
        p12L = hessL_[2][1]
        p22L = hessL_[2][2]

        a = p22L
        b = p1L - p12L @ v - p02L
        return jnp.linalg.pinv(a) @ b

    return f


def Lagrangian_to_state_derivative(L: StateFunction) -> StateFunction:
    accel = Lagrangian_to_acceleration(L)

    def f(local: State):
        t, _, v = local
        return State(jnp.ones_like(t), v, accel(local))

    return f


def Lagrangian_to_energy(L: StateFunction) -> StateFunction:
    jacL = jax.jacrev(L)  # Jacobian of the Lagrangian
    P = lambda x: jacL(x)[2]  # Partial of the Lagrangian w.r.t.
    # the third (index 2) component of the local tuple - i.e. velocity

    def f(local: State):
        _, _, v = local
        return P(local) @ v - L(local)

    return f


def Hamiltonian_to_state_derivative(H: StateFunction) -> StateFunction:
    jacH = jax.jacrev(H)

    def f(local: State):
        t, _, _ = local
        jacH_ = jacH(local)
        return State(jnp.ones_like(t), jacH_[2], -jacH_[1])

    return f


def Legendre_transform(F: Callable[..., Any]) -> Callable[..., Any]:
    """Legendre transform for quadratic functions.

    Given:
        F(v) = 1/2 v^T A v + b^T v + c
        w = DF(v) = A v + b
    The functions G related to F by a Legendre transform is:
        v w = F(v) + G(w)
        G(w) = w v - F(v)
        G(w) = w V(w) - F(V(w))
        V(w) = A^-1 (w - b)
    """

    w_of_v = jax.jacrev(F)
    dw_of_v = jax.jacfwd(w_of_v)

    def G(w):
        zeros = jnp.zeros_like(w)
        A = dw_of_v(zeros)
        b = w_of_v(zeros)
        v = jnp.linalg.solve(A, w - b)
        return jnp.dot(w, v) - F(v)

    return G


def Lagrangian_to_Hamiltonian(L: StateFunction) -> StateFunction:
    def H(local: State):
        t, q, p = local
        G = Legendre_transform(lambda v: L((t, q, v)))
        return G(p)

    return H


def Hamiltonian_to_Lagrangian(H: StateFunction) -> StateFunction:
    def L(local: State):
        t, q, v = local
        F = Legendre_transform(lambda p: H((t, q, p)))
        return F(v)

    return L


def robust_norm(x, p=2):
    """Taken from:
    https://timvieira.github.io/blog/post/2014/11/10/numerically-stable-p-norms/
    """

    a = jnp.abs(x).max()
    return a * jnp.linalg.norm(x / a, p)
