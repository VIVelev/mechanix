from functools import partial

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.ode import odeint

from mechanix import Lagrangian_to_state_derivative, make_lagrangian, robust_norm

mpl.rcParams["axes.formatter.useoffset"] = False
jax.config.update("jax_enable_x64", True)

# RNG keys
seed = 42
seed = jax.random.PRNGKey(seed)
kx, kv, km = jax.random.split(seed, 3)

# Simulation Parameters
N_BODIES = 3
T0 = 0.0
days = 7
Tf = T0 + days * 24 * 60 * 60
DT = 1  # 1 second
Ts = jnp.arange(T0, Tf, DT)
G = jnp.array(6.67408e-11)

# Initial Values
X0 = jax.random.uniform(kx, shape=(N_BODIES, 3), minval=-748e7, maxval=748e7)
V0 = jax.random.uniform(kv, shape=(N_BODIES, 3), minval=-1e5, maxval=1e5)
M = jax.random.uniform(km, shape=(N_BODIES,), minval=5.972e27, maxval=1.898e30)
# Ignore z-axis
X0 = X0.at[:, 2].set(0.0)
V0 = V0.at[:, 2].set(0.0)

# System State
init_local = (jnp.array(T0), X0.flatten(), V0.flatten())


def T(M):
    @jax.vmap
    def kinetic_energy(m, v):
        return 0.5 * m * jnp.dot(v, v)

    return lambda local: kinetic_energy(M, local[2].reshape(N_BODIES, -1)).sum()


def V(G, M):
    @partial(jax.vmap, in_axes=(None, 0, None, 0))
    @partial(jax.vmap, in_axes=(0, None, 0, None))
    def gravitational_energy(m1, m2, r1, r2):
        r = r2 - r1
        norm = robust_norm(jnp.where(jnp.allclose(r1, r2), 0.1 * jnp.ones_like(r), r))
        return -G * m1 * m2 / norm

    mask = jnp.triu(jnp.ones((N_BODIES, N_BODIES), dtype=bool), k=1)
    return lambda local: (
        gravitational_energy(
            M,
            M,
            local[1].reshape(N_BODIES, -1),
            local[1].reshape(N_BODIES, -1),
        )
        * mask
    ).sum()


L = make_lagrangian(T(M), V(G, M))
state_derivative = jax.jit(Lagrangian_to_state_derivative(L))


# -----------------------------------------------
# ODE Solver
# -----------------------------------------------
# We call `odeint` which solves the system of differential equations
# using the 4th order Runge-Kutta method. The returned X and V values
# have a the shape (time, bodies, dimensions)

local = odeint(lambda y, t: state_derivative(y), init_local, Ts)
local = jax.tree_map(np.asarray, local)


# -----------------------------------------------
# Create animation
# -----------------------------------------------
# Here we use the animation module from matplotlib to create an animation
# of the system. The animate function iteratively updates some lines and scatter
# plots.

X = local[1].reshape(len(Ts), N_BODIES, -1)
fig, ax = plt.subplots(figsize=(8, 8))

ax.axis("off")
ax.set_xlim(X[..., 0].min(), X[..., 0].max())
ax.set_ylim(X[..., 1].min(), X[..., 1].max())

trajectories = tuple([ax.plot(X[:0, j, 0], X[:0, j, 1])[0] for j in range(N_BODIES)])
bodies = ax.scatter(X[0, :, 0], X[0, :, 1])


def animate(i):
    i0 = max(0, i - 42_000)
    for j in range(N_BODIES):
        trajectories[j].set_data(X[i0:i, j, 0], X[i0:i, j, 1])

    bodies.set_offsets(X[i, :, :2])

    return trajectories + (bodies,)


anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=lambda: animate(0),
    frames=range(0, len(X), 1000),
    interval=20,
    blit=True,
)

plt.show()

# print("Saving animation...")
# anim.save("nbody.mp4", writer="ffmpeg")
