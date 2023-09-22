import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint

from mechanix import Lagrangian_to_energy, Lagrangian_to_state_derivative, Local


def L0(m, V):
    """Langrangiang for the third particle of mass `m` moving in a
    field derived from a time-varying gravitationa potential `V`.
    """

    def f(local):
        return 0.5 * m * jnp.dot(local.v, local.v) - V(local.t, local.pos)

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

        # Calculate the potential
        return -GM0 * m / r0 - GM1 * m / r1

    return f


# Simulation Parameters
m = 1.0
a = 2.0
GM0 = 1.0
GM1 = 1.0
T0 = 0.0
Tf = 10.0
DT = 0.01  # 1 second
Ts = jnp.arange(T0, Tf, DT)

# Initial Values
t0 = jnp.array(T0)
q0 = jnp.array([a, 0.0])
v0 = jnp.array([0.0, 0.0])
init_local = Local(t0, q0, v0)

L = L0(m, V(a, GM0, GM1, m))
dlocal = jax.jit(Lagrangian_to_state_derivative(L))
energy = jax.jit(jax.vmap(Lagrangian_to_energy(L)))

locals = odeint(lambda y, t: dlocal(y), init_local, Ts)
energies = energy(locals)


# -----------------------------------------------
# Plotting
# -----------------------------------------------

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

X = np.asarray(locals.pos)
Vs = np.asarray(locals.v)
E = np.asarray(energies)

a0, a1 = get_a0_a1(a, GM0, GM1)
Omega = get_Omega(a, GM0, GM1)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].axis("off")
ax[0].set_xlim(-a * 1.5, a * 1.5)
ax[0].set_ylim(-a * 1.5, a * 1.5)

# Animate the rotation of the two bodies and the path of the third body.

# Plot the first two bodies
stars = ax[0].scatter([a0, a1], [0.0, 0.0], c=[GM0, GM1])

# Plot the path of the third body
(traj,) = ax[0].plot([], [], c="k")

# Plot the third body
body = ax[0].scatter([], [], c="k")

# Plot the velocity of the third body (as an arrow)
(velocity,) = ax[0].plot([], [], c="k")

# Energy plot
ax[1].set_xlim(T0, Tf)
ax[1].set_ylim(E.min(), E.max())
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Energy")
(energy_line,) = ax[1].plot([], [], c="k")

tail_length = len(Ts) // 10


def animate(i):
    i0 = max(0, i - tail_length)
    t = Ts[i]
    stars.set_offsets(
        [
            [-a0 * np.cos(Omega * t), -a0 * np.sin(Omega * t)],
            [a1 * np.cos(Omega * t), a1 * np.sin(Omega * t)],
        ],
    )
    traj.set_data(X[i0:i, 0], X[i0:i, 1])
    body.set_offsets(X[i, :])
    velocity.set_data([X[i, 0], X[i, 0] + Vs[i, 0]], [X[i, 1], X[i, 1] + Vs[i, 1]])
    energy_line.set_data(Ts[i0:i], E[i0:i])
    return stars, traj, body, velocity, energy_line


anim = FuncAnimation(fig, animate, frames=len(Ts), interval=20, blit=False)
plt.show()
