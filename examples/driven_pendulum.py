import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.animation import FuncAnimation

from mechanix import (
    F2C,
    Hamiltonian_to_state_derivative,
    Lagrangian_to_Hamiltonian,
    State,
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

T = 1000.0
dt = 0.01
local0 = State(jnp.array(0.0), jnp.array([1.0]), jnp.array([0.0]))

hamiltonian = H(m, g, l, A, omega)
dstate = jax.jit(Hamiltonian_to_state_derivative(hamiltonian))
func = lambda y, t: dstate(y)


def ab2(f, y0, ts):
    """y_n+2 - y_n+1 = h/2 [3f_n+1 - f_n]"""

    def body(carry, t):
        f_n, y_n1, t_prev = carry
        f_n1 = f(y_n1, t)
        y_n2 = y_n1 + (t - t_prev) / 2 * (3 * f_n1 - f_n)
        return (f_n1, y_n2, t), y_n2

    h0 = ts[1] - ts[0]
    f0 = f(y0, ts[0])
    _, ys = jax.lax.scan(body, (f0, y0 + h0 * f0, ts[0]), ts[1:])
    return ys


def rk4(f, y0, ts):
    def body(carry, t):
        y_prev, t_prev = carry
        h = t - t_prev
        k1 = f(y_prev, t_prev)
        k2 = f(y_prev + h / 2 * k1, t_prev + h / 2)
        k3 = f(y_prev + h / 2 * k2, t_prev + h / 2)
        k4 = f(y_prev + h * k3, t_prev + h)
        y = y_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y, t), y

    _, ys = jax.lax.scan(body, (y0, ts[0]), ts[1:])
    return ys


def euler(f, y0, ts):
    def body(carry, t):
        y_prev, t_prev = carry
        h = t - t_prev
        y = y_prev + h * f(y_prev, t_prev)
        return (y, t), y

    _, ys = jax.lax.scan(body, (y0, ts[0]), ts[1:])
    return ys


locals = ab2(func, local0, jnp.arange(0, T, dt))
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
    range(1, len(Ts), 10),
    init,
    interval=20,
)

plt.show()
# html(anim.to_jshtml(), height=1000)
