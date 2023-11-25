import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from jax.experimental.ode import odeint
from matplotlib.animation import FuncAnimation

from mechanix import (
    F2C,
    Hamiltonian_to_state_derivative,
    Lagrangian_to_Hamiltonian,
    compose,
)

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)


def L_uniform_accl(m, g):
    def f(local):
        _, [_, y], v = local
        return 0.5 * m * v.T @ v - m * g * y

    return f


def dp_coordinates(l, ys):
    def f(local):
        t, [theta], _ = local
        return jnp.array([l * jnp.sin(theta), ys(t) - l * jnp.cos(theta)])

    return f


def L_pend(m, g, l, ys):
    return compose(
        L_uniform_accl(m, g),
        F2C(dp_coordinates(l, ys)),
    )


def periodic_drive(amplitude, frequency, phase):
    def f(t):
        return amplitude * jnp.cos(frequency * t + phase)

    return f


def L_periodically_driven_pendulum(m, g, l, A, omega):
    ys = periodic_drive(A, omega, 0.0)
    return L_pend(m, g, l, ys)


def H(m, g, l, A, omega):
    return Lagrangian_to_Hamiltonian(L_periodically_driven_pendulum(m, g, l, A, omega))


m = st.slider("Mass", 0.1, 3.0, 1.0)  # kg
g = st.slider("Gravity", 0.0, 15.0, 9.8)  # m/s^2
l = st.slider("Length", 0.1, 3.0, 1.0)  # m
A = st.slider("Amplitude", 0.0, 1.0, 0.1)  # m
omega = st.slider("Frequency", 0.0, 10.0, 2 * np.sqrt(g))  # Hz

t0 = jnp.array(0.0)
t1 = 1000.0
dt = 0.01
local0 = (t0, jnp.array([1.0]), jnp.array([0.0]))

hamiltonian = H(m, g, l, A, omega)
dstate = jax.jit(Hamiltonian_to_state_derivative(hamiltonian))
func = lambda y, t: dstate(y)

# NOTE: Try a different integrator?
locals = odeint(func, local0, jnp.arange(t0, t1, dt))
Ts, Qs, Ps = tuple(map(np.asarray, locals))


fig, ax = plt.subplots()
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-10, 10)

(traj,) = ax.plot([], [], lw=1, c="r", ls="--")
dot = ax.scatter([], [], s=20, c="r")


def init():
    return traj, dot


def principal(q, *, cutoff=np.pi):
    return q - 2 * cutoff * np.round(q / (2 * cutoff))


def update(i):
    # Remove the connection b/w jumps
    xs = principal(Qs[:i, 0])
    diffs = np.append(np.diff(xs), 0)
    idxs = np.abs(diffs) > np.pi
    xs[idxs] = np.nan

    traj.set_data(xs, Ps[:i, 0])
    dot.set_offsets([[principal(Qs[i, 0]), Ps[i, 0]]])
    return traj, dot


anim = FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=range(1, len(Ts), 10),
    interval=20,
)

plt.show()
# html(anim.to_jshtml(), height=1000)
