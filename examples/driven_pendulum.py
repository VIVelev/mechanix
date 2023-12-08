import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mechanix import (
    F2C,
    Hamiltonian_to_state_derivative,
    Lagrangian_to_Hamiltonian,
    State,
    compose,
    explore_map,
    principal,
    state_advancer,
)

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)

jax.config.update("jax_enable_x64", True)


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


def H_pend_sysder(m, g, l, A, omega):
    return compose(
        Hamiltonian_to_state_derivative,
        Lagrangian_to_Hamiltonian,
        L_periodically_driven_pendulum,
    )(m, g, l, A, omega)


def driven_pendulum_map(m, g, l, A, omega):
    advance = state_advancer(H_pend_sysder, m, g, l, A, omega)
    map_period = (2 * np.pi) / omega

    def sys_map(qp):
        q, p = qp
        ns = advance(State(jnp.array(0.0), jnp.array([q]), jnp.array([p])), map_period)
        return jnp.array([principal(ns[1])[0], ns[2][0]])

    return sys_map


m = 1.0  # kg
g = 9.8  # m/s^2
l = 1.0  # m
A = 0.1  # m
omega0 = np.sqrt(g / l)
omega = 2 * omega0  # Hz
n = 128

fig, ax = plt.subplots()
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-10, 10)
explore_map(fig, driven_pendulum_map(m, g, l, A, omega), n)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$p_{\theta}$")
plt.show()
