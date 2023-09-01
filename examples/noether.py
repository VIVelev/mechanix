from inspect import signature

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.ode import odeint

from mechanix import (
    F2C,
    Lagrangian_to_state_derivative,
    Local,
    Noether_integral,
    Rx,
    Ry,
    Rz,
    compose,
)

jax.config.update("jax_enable_x64", True)
mpl.rc("axes.formatter", useoffset=False)


def L_free_cartesian(m):
    return lambda local: 0.5 * m * local.v @ local.v


def ellipsoidal_to_cartesian(rx, ry, rz):
    def F(local):
        theta, phi = local.pos
        x = rx * jnp.cos(theta) * jnp.sin(phi)
        y = ry * jnp.sin(theta) * jnp.sin(phi)
        z = rz * jnp.cos(phi)
        return jnp.array([x, y, z])

    return F


def cartisian_to_ellipsoidal(rx, ry, rz):
    def F(local):
        x, y, z = local.pos
        theta = jnp.arctan2(y * rx, x * ry)
        phi = jnp.arccos(z / rz)
        return jnp.array([theta, phi])

    return F


# --- Parameters ---
# Radii
rx, ry, rz = 1, 2, 2

# Time parameters
dt = 0.01
tmax = 100
ts = jnp.arange(0, tmax, dt)

# Initial state
seed = 0
rng = jax.random.PRNGKey(seed)
km, kx, kv = jax.random.split(rng, 3)
m = jax.random.uniform(km, (), minval=0.5, maxval=1.5)
t0 = jnp.array(0.0)
q0 = jax.random.uniform(kx, (2,), minval=0, maxval=np.pi)
v0 = jax.random.uniform(kv, (2,), minval=0, maxval=np.pi / 4)
init_local = Local(t0, q0, v0)
# ----------------------------


L_free_ellipsoidal = compose(
    L_free_cartesian(m),
    F2C(ellipsoidal_to_cartesian(rx, ry, rz)),
)
dstate = jax.jit(Lagrangian_to_state_derivative(L_free_ellipsoidal))


# --- Integration ---
func = lambda y, t: dstate(y)  # noqa: E731
locals = odeint(func, init_local, ts)

# --- Noether's Integral ---


def F_tilde(angle_x, angle_y, angle_z):
    def F(local):
        q = ellipsoidal_to_cartesian(rx, ry, rz)(local)
        q = Rx(angle_x)(q)
        q = Ry(angle_y)(q)
        q = Rz(angle_z)(q)
        local = Local(local.t, q, local.v)
        q = cartisian_to_ellipsoidal(rx, ry, rz)(local)
        return q

    return F


noether = jax.vmap(Noether_integral(L_free_ellipsoidal, F_tilde))
noether_values = noether(locals)


# --- Plotting ---
locals = jax.tree_map(
    np.asarray, jax.vmap(F2C(ellipsoidal_to_cartesian(rx, ry, rz)))(locals)
)
X = locals.pos

fig = plt.figure(figsize=plt.figaspect(0.5))  # Square figure
ax = fig.add_subplot(121, projection="3d")
ax.set_aspect("equal")

# Plot ellipsoid surface:
# Set of all spherical angles:
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 100)
# Cartesian coordinates that correspond to the spherical angles:
# (this is the equation of an ellipsoid):
x = rx * np.outer(np.cos(theta), np.sin(phi))
y = ry * np.outer(np.sin(theta), np.sin(phi))
z = rz * np.outer(np.ones_like(theta), np.cos(phi))
# Plot:
ax.plot_surface(x, y, z, rstride=4, cstride=4, color="b")
# Adjustment of the axes, so that they all have the same span:
max_radius = max(rx, ry, rz)
for axis in "xyz":
    getattr(ax, "set_{}ticks".format(axis))([])
    getattr(ax, "set_{}label".format(axis))(axis)
    getattr(ax, "set_{}lim".format(axis))((-max_radius, max_radius))

# Plot trajectory:
ax.plot(X[:, 0], X[:, 1], X[:, 2], "r-", zorder=10)

# Noether's Integral:
ax = fig.add_subplot(122)
ax.set_ylim(-1, 1)
for i, name in enumerate(signature(F_tilde).parameters.keys()):
    ax.plot(ts, noether_values[:, i], label=name)
ax.legend(loc="upper left")


plt.show()
