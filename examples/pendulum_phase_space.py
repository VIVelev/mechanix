import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.ode import odeint
from matplotlib.animation import FuncAnimation

from mechanix import Hamiltonian_to_state_derivative

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)


def H(m, g, l):
    def f(local):
        _, [q], [p] = local
        return p**2 / (2 * m) + m * g * l * (1 - jnp.cos(q))

    return f


m = 1.0  # kg
g = 9.8  # m/s^2
l = 1.0  # m

t0 = jnp.array(0.0)
t1 = 2 * np.pi * np.sqrt(l / g)
dt = 0.01
local0 = (t0, jnp.array([np.pi / 6]), jnp.array([0.0]))

hamiltonian = H(m, g, l)
dstate = jax.jit(Hamiltonian_to_state_derivative(hamiltonian))
func = lambda y, t: dstate(y)
locals = odeint(func, local0, jnp.arange(t0, t1 + dt, dt))
Ts, Qs, Ps = tuple(map(np.asarray, locals))

N = 100
ts = np.zeros((N, N))
qs, ps = np.meshgrid(np.linspace(-np.pi, np.pi, N), np.linspace(-20, 20, N))
hs = np.asarray(jax.vmap(jax.vmap(hamiltonian))((ts, qs[:, :, None], ps[:, :, None])))

fig, ax = plt.subplots()
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-20, 20)

cont = ax.contour(qs, ps, hs, np.linspace(0, 50, 10), cmap="coolwarm")

dot = ax.scatter([], [], s=20, c="r")


def init():
    return cont, dot


def principal(q):
    return q - 2 * np.pi * np.round(q / (2 * np.pi))


def update(i):
    dot.set_offsets([[principal(Qs[i, 0]), Ps[i, 0]]])
    return cont, dot


anim = FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=len(Ts),
    interval=20,
    blit=True,
)

plt.show()
