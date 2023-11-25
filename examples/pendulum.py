import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.ode import odeint

from mechanix import F2C, Lagrangian_to_state_derivative, compose


def T(m):
    def f(local):
        _, _, v = local
        return 0.5 * m * v.T @ v

    return f


def V(m, g):
    def f(local):
        _, [_, y], _ = local
        return m * g * y

    return f


L_rectangular = lambda local: T(m)(local) - V(m, g)(local)


# Convert pendulum coordinates (theta) to rectangular coordinates (x, y)
def pendulum2rect(local):
    _, [theta], _ = local
    return l * jnp.array([jnp.cos(theta), jnp.sin(theta)])


local_tuple_transformation = F2C(pendulum2rect)
L_polar = compose(
    L_rectangular,
    local_tuple_transformation,
)
dstate = jax.jit(Lagrangian_to_state_derivative(L_polar))


# System parameters
l = 1.0  # m  # noqa: E741
m = 1.0  # kg
g = 9.81  # m/s^2

# Time parameters
t0 = 0.0  # s
t1 = 10.0  # s
dt = 0.1  # s
ts = jnp.arange(t0, t1, dt)

# Initial conditions (in polar coordinates)
t0 = jnp.array(t0, dtype=float)  # s
q0 = jnp.array([-np.pi / 4])  # m
v0 = jnp.array([0.0])  # m/s
local0 = (t0, q0, v0)

# Integrate
func = lambda local, t: dstate(local)
locals = odeint(func, local0, ts)
X = np.asarray(jax.vmap(pendulum2rect)(locals))

# Animation
fig, ax = plt.subplots()
ax.set_xlim(-1.2 * l, 1.2 * l)
ax.set_ylim(-1.2 * l, 1.2 * l)
ax.set_aspect("equal")
ax.grid()

(line,) = ax.plot([], [], "-o", lw=2)


def init():
    line.set_data([], [])
    return (line,)


def animate(i):
    i0 = max(0, i - 10)
    line.set_data(X[i0:i, 0], X[i0:i, 1])
    return (line,)


ani = animation.FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=len(ts),
    interval=1000 * dt,
    blit=True,
    repeat=False,
)
plt.show()
