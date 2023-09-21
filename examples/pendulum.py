import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from jax.experimental.ode import odeint

from mechanix import F2C, Lagrangian_to_state_derivative, Local, compose

"""I will start with a simple pendulum example to
get us familiar with the library.
"""

"""Since its easier to calculate the Potential and Kinetic energy
in rectangular coordinates, I will define the Lagrangian in terms
of rectangular coordinates.
"""

st.latex(r"L = T - V = \frac{1}{2} m v^2 - m g y")

r"""At any given time, the state of the system is given by the local
tuple $(t, q, v)$. Where q is the generalized coordinates and $v = \frac{dq}{dt}$.
Therefore, the lagrangian is a *Local Tuple Function*."""

"""We define the Kinetic energy as"""

st.code(
    """
def T(m):
    def f(local):
        _, _, v = local
        return 0.5 * m * v @ v

    return f
"""
)


def T(m):
    def f(local):
        _, _, v = local
        return 0.5 * m * v @ v

    return f


"""And the Potential"""

st.code(
    """
def V(m, g):
    def f(local):
        _, [_, y], _ = local
        return m * g * y

    return f
"""
)


def V(m, g):
    def f(local):
        _, [_, y], _ = local
        return m * g * y

    return f


"""Both take as input some parameters and return a Local Tuple Function.
Therefore the Lagrangian can easily be defined as"""

st.code("L_rectangular = lambda local: T(m)(local) - V(m, g)(local)")
L_rectangular = lambda local: T(m)(local) - V(m, g)(local)


r"""Now, I actually want to define and monitor the position of the pendulum
in terms of the angle $\theta$, therefore I will define a *Coordinate Transformation*
('F') from the pendulum coordinates ($\theta$) to rectangular coordinates.
"""

st.code(
    """
# Convert pendulum coordinates (theta) to rectangular coordinates (x, y)
def pendulum2rect(local):
    theta = local.pos[0]
    return l * jnp.array([jnp.cos(theta), jnp.sin(theta)])
"""
)


# Convert pendulum coordinates (theta) to rectangular coordinates (x, y)
def pendulum2rect(local):
    theta = local.pos[0]
    return l * jnp.array([jnp.cos(theta), jnp.sin(theta)])


"""But wait, I defined my Lagrangian in terms of rectangular coordinates,
now what? No worries, we just need to figure out the corresponding *Local Tuple
Transformation* ('C'). Fortunately, that can easily be done with the `F2C` function."""

st.code("local_tuple_transformation = F2C(pendulum2rect)")
local_tuple_transformation = F2C(pendulum2rect)

"""Now our final Lagrangian will be the composition of the two."""

st.code(
    """
L_polar = compose(
    L_rectangular,
    local_tuple_transformation,
)
"""
)
L_polar = compose(
    L_rectangular,
    local_tuple_transformation,
)


r"""Now, from this Lagrangian we can take the state (local tuple) derivative:
$dstate = (\frac{dt}{dt}, \frac{dq}{dt}, \frac{dv}{dt})$"""
st.code("dstate = Lagrangian_to_state_derivative(L_polar)")
dstate = Lagrangian_to_state_derivative(L_polar)

"""And finally we integrate the state derivative to simulate the system."""

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
local0 = Local(t0, q0, v0)

# Integrate
func = lambda local, t: dstate(local)


@st.cache_data
def get_X():
    locals = odeint(func, local0, ts)
    return np.asarray(jax.vmap(pendulum2rect)(locals))


X = get_X()

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

components.html(ani.to_jshtml(), height=1000)
