import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint

from mechanix import (
    F2C,
    Lagrangian_to_energy,
    Lagrangian_to_state_derivative,
    Local,
    compose,
    p2r,
    r2p,
)

jax.config.update("jax_enable_x64", True)


def L0(m, V):
    """Langrangiang for the thord particle
    of mass `m` moving in a field derived from a
    time-varying gravitationa potential `V`.
    """

    def f(local):
        return 0.5 * m * jnp.dot(local.v, local.v) - V(local.t, local.pos)

    return f


def get_Omega(a, GM0, GM1):
    # From Kepler's third law:
    # Omega^2 * a^3 = G(M0 + M1)
    # Omega = sqrt(G(M0 + M1) / a^3)
    Omega = jnp.sqrt((GM0 + GM1) / a**3)
    return Omega


def get_a0_a1(a, GM0, GM1):
    a0 = GM1 / (GM0 + GM1) * a
    a1 = GM0 / (GM0 + GM1) * a
    return a0, a1


def V(a, GM0, GM1, m):
    Omega = get_Omega(a, GM0, GM1)
    a0, a1 = get_a0_a1(a, GM0, GM1)

    def f(t, xy):
        x, y = xy
        x0 = -a0 * jnp.cos(Omega * t)
        y0 = -a0 * jnp.sin(Omega * t)
        x1 = a1 * jnp.cos(Omega * t)
        y1 = a1 * jnp.sin(Omega * t)
        r0 = jnp.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        r1 = jnp.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        return -GM0 * m / r0 - GM1 * m / r1

    return f


def rot(omega):
    def f(local):
        t = local.t
        r, theta = local.pos
        return jnp.array([r, theta + omega * t])

    return f


# Simulation Parameters
# A satellite of mass m orbits the Earth and Moon
# Voyager 1 mass:
m = 721.9  # kg
# Distance between the Earth and the Moon:
a = 384400e3  # m
# Gravitational constants:
G = 6.67408e-11  # m^3 kg^-1 s^-2
# Masses of the Earth and Moon (times G):
GM0 = G * 5.972e24  # m^3 s^-2
GM1 = G * 7.34767309e22  # m^3 s^-2
# Time parameters:
T0 = 0.0
Tf = 2 * np.pi * jnp.sqrt(a**3 / (GM0 + GM1))
DT = 1.0  # 1 second
Ts = jnp.arange(T0, Tf, DT)

# Distance to the COM of the Earth-Moon system:
a0, a1 = get_a0_a1(a, GM0, GM1)

# Initial Values
t0 = jnp.array(T0)
# Earth radius (launch site):
d = 6371e3  # m
q0 = jnp.array([-a0 + d, 0.0])
# Escape velocity:
v_esc = jnp.sqrt(2 * GM0 / d)
# Pick a direction:
v0 = jnp.array([0.9, 0.1])
# Scale the velocity to the 98% escape velocity:
v0 = 0.98 * v_esc * v0 / jnp.linalg.norm(v0)
init_local = Local(t0, q0, v0)

L = compose(
    L0(m, V(a, GM0, GM1, m)),
    F2C(p2r),
    F2C(rot(get_Omega(a, GM0, GM1))),
    F2C(r2p),
)
dlocal = jax.jit(Lagrangian_to_state_derivative(L))
energy = jax.jit(jax.vmap(Lagrangian_to_energy(L)))

locals = odeint(lambda y, t: dlocal(y), init_local, Ts)
energies = energy(locals)


# -----------------------------------------------
# Plotting
# -----------------------------------------------

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

mpl.rc("axes.formatter", useoffset=False)

X = np.asarray(locals.pos)
Vs = np.asarray(locals.v)
E = np.asarray(energies)
J = np.mean(E)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
x_lim, y_lim = a * 1.5, a * 1.5
ax[0].axis("off")
ax[0].set_xlim(-x_lim, x_lim)
ax[0].set_ylim(-y_lim, y_lim)

# Animate the rotation of the two bodies and the path of the third body.

# Plot the potential
sample_size = 512
xs, ys = np.meshgrid(
    np.linspace(-x_lim, x_lim, sample_size), np.linspace(-y_lim, y_lim, sample_size)
)
xys = jnp.stack([xs, ys], axis=-1)
vs = jnp.zeros_like(xys)
ts = jnp.zeros((sample_size, sample_size))
states = Local(ts, xys, vs)
zs = np.array(jax.vmap(energy)(states))


zs[zs < J] = np.nan
ax[0].contourf(xs, ys, zs, cmap="Greys")


# Plot the first two bodies
ax[0].scatter([-a0, a1], [0.0, 0.0], c=[GM0, GM1])

# Plot the path of the third body
(traj,) = ax[0].plot([], [], c="k")

# Plot the third body
body = ax[0].scatter([], [], c="k")

# Plot the velocity of the third body (as an arrow)
(velocity,) = ax[0].plot([], [], c="k")

# Energy plot
ax[1].set_xlim(T0, Tf)
ax[1].set_ylim(J * 2, 1)
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Energy")
(energy_line,) = ax[1].plot([], [], c="k")

tail_length = len(Ts) // 10


def animate(i):
    i0 = max(0, i - tail_length)
    traj.set_data(X[i0:i, 0], X[i0:i, 1])
    body.set_offsets(X[i, :])
    velocity.set_data([X[i, 0], X[i, 0] + Vs[i, 0]], [X[i, 1], X[i, 1] + Vs[i, 1]])
    energy_line.set_data(Ts[i0:i], E[i0:i])
    return traj, body, velocity, energy_line


anim = FuncAnimation(
    fig, animate, frames=range(1, len(Ts), 1000), interval=20, blit=False
)
plt.show()
