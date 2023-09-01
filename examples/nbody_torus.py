from functools import partial as curry

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.ode import odeint

from mechanix import (
    F2C,
    Lagrangian_to_state_derivative,
    Local,
    compose,
    make_lagrangian,
    robust_norm,
)

mpl.rcParams["axes.formatter.useoffset"] = False
jax.config.update("jax_enable_x64", True)

# RNG keys
seed = 10
key = jax.random.PRNGKey(seed)
kx, kv, km = jax.random.split(key, 3)

# Simulation Parameters
N_BODIES = 5
T0 = 0.0
days = 30
Tf = T0 + days * 24 * 60 * 60
DT = 1
Ts = jnp.arange(T0, Tf, DT)
G = jnp.array(6.67408e-11)
TORUS_R, TORUS_r = 2e10, 1e10


# Initial Values
X0 = jax.random.uniform(kx, shape=(N_BODIES, 2), minval=0, maxval=2 * np.pi)
V0 = jax.random.uniform(kv, shape=(N_BODIES, 2), minval=-1e-5, maxval=1e-5)
M = jax.random.uniform(km, shape=(N_BODIES, 1), minval=5.972e27, maxval=1.898e30)

# System State
init_local = Local(jnp.array(T0), X0.flatten(), V0.flatten())


def T(M):
    @jax.vmap
    def kinetic_energy(m, v):
        return 0.5 * m * jnp.dot(v, v)

    return lambda local: kinetic_energy(M, local.v.reshape(N_BODIES, -1)).sum()


def V(G, M):
    @curry(jax.vmap, in_axes=(None, 0, None, 0))
    @curry(jax.vmap, in_axes=(0, None, 0, None))
    def gravitational_energy(m1, m2, r1, r2):
        r = r2 - r1
        potential = -G * m1 * m2 / robust_norm(r)
        return jnp.where(jnp.allclose(r1, r2), 0, potential)

    mask = jnp.triu(jnp.ones((N_BODIES, N_BODIES), dtype=bool), 1)
    return lambda local: (
        gravitational_energy(
            M,
            M,
            local.pos.reshape(N_BODIES, -1),
            local.pos.reshape(N_BODIES, -1),
        )
        * mask
    ).sum()


L_cartesian = make_lagrangian(T(M), V(G, M))


def toroidal_to_cartesian(R, r):
    @jax.vmap
    def f(pos):
        theta, phi = pos
        x = (R + r * jnp.cos(theta)) * jnp.cos(phi)
        y = (R + r * jnp.cos(theta)) * jnp.sin(phi)
        z = r * jnp.sin(theta)
        return jnp.array([x, y, z])

    return lambda local: f(local.pos.reshape(N_BODIES, 2)).flatten()


L_toroidal = compose(L_cartesian, F2C(toroidal_to_cartesian(TORUS_R, TORUS_r)))
state_derivative = jax.jit(Lagrangian_to_state_derivative(L_toroidal))


# -----------------------------------------------
# ODE Solver
# -----------------------------------------------
# We call `odeint` which solves the system of differential equations
# using the 4th order Runge-Kutta method. The returned X and V values
# have a the shape (time, bodies, dimensions)


def clip_by_norm(x, max_norm):
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return jnp.where(norm > max_norm, (x / norm) * max_norm, x)


def boundary_conditions(y, max_speed=0.1):
    # pos = jnp.mod(y.pos, 2 * np.pi)
    v = clip_by_norm(y.v.reshape(N_BODIES, -1), max_speed).flatten()
    return Local(y.t, y.pos, v)


local = odeint(lambda y, t: state_derivative(y), init_local, Ts)

print("isnan:", jnp.any(jnp.isnan(local.pos)))
print("isinf:", jnp.any(jnp.isinf(local.pos)))


X = jax.vmap(toroidal_to_cartesian(TORUS_R, TORUS_r))(local)
X = np.asarray(X).reshape(-1, N_BODIES, 3)

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=(8, 8))

ax.axis("off")
# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)
# ax.set_zlim(-2, 2)

# Plot surface
n = 50
theta = jnp.linspace(0, 2.0 * np.pi, n)
phi = jnp.linspace(0, 2.0 * np.pi, n)
theta, phi = jnp.meshgrid(theta, phi)


def polar_to_toroid(st):
    theta, phi = st
    x = (TORUS_R + TORUS_r * np.cos(theta)) * np.cos(phi)
    y = (TORUS_R + TORUS_r * np.cos(theta)) * np.sin(phi)
    z = TORUS_R * np.sin(theta)

    return x, y, z


x, y, z = polar_to_toroid((theta, phi))
ax.plot_surface(
    x,
    y,
    z,
    rstride=5,
    cstride=5,
    color="w",
    edgecolors="k",
    alpha=0,
    linewidth=0.2,
    linestyle=":",
)

# Now animate the path
trajs = tuple(
    ax.plot(X[:0, :, 0], X[:0, :, 1], X[:0, :, 2])[0] for _ in range(N_BODIES)
)
ball = ax.scatter(X[0, :, 0], X[0, :, 1], X[0, :, 2])


def animate(i):
    i0 = max(0, i - 42_000)
    for j in range(N_BODIES):
        trajs[j]._verts3d = X[i0:i, j, 0], X[i0:i, j, 1], X[i0:i, j, 2]
    ball._offsets3d = X[i, :, 0], X[i, :, 1], X[i, :, 2]
    return trajs + (ball,)


anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=lambda: animate(0),
    frames=range(0, len(X), 1000),
    interval=20,
    blit=False,
)

plt.show()
# anim.save("nbody_torus.mp4", writer="ffmpeg", fps=60)
