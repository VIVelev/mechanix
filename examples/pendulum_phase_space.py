import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from jax.experimental.ode import odeint
from matplotlib.animation import FuncAnimation
from streamlit.components.v1 import html

from mechanix import Hamiltonian_to_state_derivative

plt.style.use("cyberpunk")


def H(m, g, l):
    def f(local):
        _, q, p = local
        return p**2 / (2 * m) + m * g * l * (1 - jnp.cos(q))

    return f


m = st.slider("Mass", 0.1, 3.0, 1.0)  # kg
g = st.slider("Gravity", 0.0, 15.0, 9.8)  # m/s^2
l = st.slider("Length", 0.1, 3.0, 1.0)  # m

t0 = jnp.array(0.0)
t1 = 2.0
dt = 0.01
local0 = (t0, jnp.array(jnp.pi / 4), jnp.array(2 * jnp.pi))

hamiltonian = H(m, g, l)
dstate = Hamiltonian_to_state_derivative(hamiltonian)
func = lambda y, t: dstate(y)
locals = odeint(func, local0, jnp.arange(t0, t1, dt))
Ts, Qs, Ps = tuple(map(np.asarray, locals))

N = 100
ts = np.zeros((N, N))
qs, ps = np.meshgrid(np.linspace(-np.pi, np.pi, N), np.linspace(-20, 20, N))
hs = np.asarray(jax.vmap(hamiltonian)((ts, qs, ps)))

fig, ax = plt.subplots()
fig.set_dpi(100)
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-20, 20)

cont = ax.contour(qs, ps, hs, np.linspace(0, 50, 10), cmap="coolwarm")

dot = ax.scatter([], [], s=20, c="r")


def init():
    return cont, dot


def principal(q):
    return q - 2 * np.pi * np.round(q / (2 * np.pi))


def update(i):
    dot.set_offsets([principal(Qs[i]), Ps[i]])
    return cont, dot


anim = FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=len(Ts),
    interval=20,
    blit=True,
)

html(anim.to_jshtml(), height=1000)
