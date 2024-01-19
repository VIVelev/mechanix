import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.ode import odeint
from matplotlib.animation import FuncAnimation

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
    advance = state_advancer(H_pend_sysder(m, g, l, A, omega))
    map_period = (2 * np.pi) / omega

    def sys_map(qp):
        q, p = qp
        ns = advance(State(jnp.array(0.0), jnp.array([q]), jnp.array([p])), map_period)
        return jnp.array([principal(ns[1])[0], ns[2][0]])

    return sys_map


m = 1.0  # kg
g = 9.8  # m/s^2
l = 1.0  # m
A = 0.2  # m
omega0 = np.sqrt(g / l)
omega = 10.1 * omega0  # Hz
n = 64

f = plt.figure()
plt.xlim(-jnp.pi, jnp.pi)
plt.ylim(-20, 20)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$p_{\theta}$")
explore_map(f, driven_pendulum_map(m, g, l, A, omega), n)
f.show()

# Make some particular simulation
dstate = jax.jit(H_pend_sysder(m, g, l, A, omega))
func = lambda y, t: dstate(y)
y0 = State(jnp.array(0.0), jnp.array([-3.0]), jnp.array([1.0]))
t = jnp.linspace(0.0, 100 * (2 * jnp.pi / omega), 10_000)
locals = odeint(func, y0, t)
ys = periodic_drive(A, omega, 0.0)
to_cartesian = dp_coordinates(l, ys)
xys = jax.vmap(to_cartesian)(locals)
assert xys.shape == (len(t), 2)


# Plot 2D animation
f2 = plt.figure()
plt.xlim(-1.1 * l, 1.1 * l)
plt.ylim(-1.1 * l, 1.1 * l)

pivot = plt.scatter([], [], marker="x", color="r", s=100)
(chain,) = plt.plot([], [], "r", lw=2)
bob = plt.scatter([], [], marker="o", color="k", s=100)
(traj,) = plt.plot([], [], "k--", lw=2)


def init():
    pivot.set_offsets([0.0, 0.0])
    chain.set_data([], [])
    bob.set_offsets([0.0, 0.0])
    traj.set_data([], [])
    return pivot, chain, bob, traj


def update(i):
    i0 = max(0, i - 100)
    _pivot = np.array([0.0, ys(t[i])])
    pivot.set_offsets(_pivot)
    chain.set_data([xys[i, 0], _pivot[0]], [xys[i, 1], _pivot[1]])

    _bob = xys[i]
    bob.set_offsets(_bob)
    traj.set_data(xys[i0:i, 0], xys[i0:i, 1])

    distance = np.sqrt(np.sum((_bob - _pivot) ** 2))
    assert np.isclose(distance, l)

    return pivot, chain, bob, traj


anim = FuncAnimation(
    f2,
    update,
    frames=range(1, len(t), 5),
    interval=20,
    init_func=init,
    repeat=True,
)
f2.show()

input()
