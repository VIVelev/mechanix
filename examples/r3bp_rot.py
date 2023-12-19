import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint

from mechanix import (
    F2C,
    Lagrangian_to_energy,
    Lagrangian_to_state_derivative,
    compose,
)

jax.config.update("jax_enable_x64", True)


def L0(m, V):
    """Langrangiang for the third particle of mass `m` moving in a
    field derived from a time-varying gravitationa potential `V`.
    """

    def f(local):
        t, q, v = local
        return 0.5 * m * v.T @ v - V(t, q)

    return f


def get_Omega(a, GM0, GM1):
    """Get the rate at which the two bodies orbit their center of mass."""
    # From Kepler's third law:
    # Omega^2 * a^3 = G(M0 + M1)
    # Omega = sqrt(G(M0 + M1) / a^3)
    Omega = jnp.sqrt((GM0 + GM1) / a**3)
    return Omega


def get_a0_a1(a, GM0, GM1):
    """Get the distance b/w each body and the center of mass."""
    a0 = GM1 / (GM0 + GM1) * a
    a1 = GM0 / (GM0 + GM1) * a
    return a0, a1


def V(a, GM0, GM1, m):
    """Time-varying gravitational potential for a third body of mass `m`
    orbiting two bodies of mass `GM0` and `GM1` with distance b/w them `a`.
    """
    Omega = get_Omega(a, GM0, GM1)
    a0, a1 = get_a0_a1(a, GM0, GM1)

    def f(t, xy):
        x, y = xy
        # Get the position of the two bodies at time `t`
        x0 = -a0 * jnp.cos(Omega * t)
        y0 = -a0 * jnp.sin(Omega * t)
        x1 = a1 * jnp.cos(Omega * t)
        y1 = a1 * jnp.sin(Omega * t)

        # Calculate the distance b/w the third body and the two bodies
        r0 = jnp.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        r1 = jnp.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        r0 = jax.lax.select(jnp.isclose(r0, 0.0), 1.0, r0)
        r1 = jax.lax.select(jnp.isclose(r1, 0.0), 1.0, r1)

        # Calculate the potential
        return -GM0 * m / r0 - GM1 * m / r1

    return f


def rot(omega):
    def f(local):
        t, [x, y], _ = local
        return jnp.array([x, y]) @ jnp.array(
            [
                [jnp.cos(omega * t), -jnp.sin(omega * t)],
                [jnp.sin(omega * t), jnp.cos(omega * t)],
            ]
        )

    return f


# Simulation Parameters
m = 1
a = 1
GM0 = 1
GM1 = GM0 * 0.005
# Time parameters:
T0 = 0.0
Tf = 100.0
DT = 0.01
Ts = jnp.arange(T0, Tf, DT)

# Distance to the COM of the Earth-Moon system:
a0, a1 = get_a0_a1(a, GM0, GM1)
print("a0, a1:", a0, a1)
# Angular velocity of the Earth-Moon system:
Omega = get_Omega(a, GM0, GM1)
print("Omega:", Omega)

# Initial Values
t0 = jnp.array(T0)
q0 = jnp.array([-a0, -0.99 * a])
v0 = jnp.array([0.0, 0.0])
init_local = (t0, q0, v0)
print("q0, v0:", init_local[1], init_local[2])

rotation = F2C(rot(-Omega))
L = compose(L0(m, V(a, GM0, GM1, m)), rotation)
dlocal = jax.jit(Lagrangian_to_state_derivative(L))
energy = jax.jit(Lagrangian_to_energy(L))
print("Energy:", energy(init_local))

locals = odeint(lambda y, t: dlocal(y), init_local, Ts)
energies = jax.vmap(energy)(locals)


# -----------------------------------------------
# Plotting
# -----------------------------------------------

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

mpl.rc("axes.formatter", useoffset=False)

_, X, Vs = locals
X = np.asarray(X)
Vs = np.asarray(Vs)
E = np.asarray(energies)
E_mean = np.mean(E)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
x_lim, y_lim = a * 1.5, a * 1.5
ax[0].grid(True)
ax[0].set_xlim(-x_lim, x_lim)
ax[0].set_ylim(-y_lim, y_lim)

# Animate the rotation of the two bodies and the path of the third body.

# Plot the potential
sample_size = 512
xs, ys = np.meshgrid(
    np.linspace(-x_lim, x_lim, sample_size),
    np.linspace(-y_lim, y_lim, sample_size),
)
xys = jnp.stack([xs, ys], axis=-1)
vs = jnp.zeros_like(xys)
ts = jnp.zeros((sample_size, sample_size))
states = (ts, xys, vs)
zs = np.array(jax.vmap(jax.vmap(energy))(states))

# Now get the zero-velocity curves
ax[0].contour(
    xs,
    ys,
    zs,
    levels=np.concatenate((np.linspace(-1.8, -1.6, 3), np.linspace(-1.55, -1.45, 10))),
    cmap="RdBu",
)


# Plot the first two bodies
total = (GM0 + GM1) / 1000
ax[0].scatter([-a0, a1], [0.0, 0.0], s=[GM0 / total, GM1 / total], c="gray")

# Plot the path of the third body
(traj,) = ax[0].plot([], [], c="k")

# Plot the third body
body = ax[0].scatter([], [], c="k")

# Plot the velocity of the third body (as an arrow)
(velocity,) = ax[0].plot([], [], c="k")

# Energy plot
ax[1].set_xlim(T0, Tf)
ax[1].set_ylim(E_mean * 2, np.abs(E_mean))
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
    fig,
    animate,
    frames=range(1, len(Ts), 20),
    interval=20,
    blit=False,
)
plt.show()
