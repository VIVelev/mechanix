import altair as alt
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.ode import odeint

from mechanix import Hamiltonian_to_state_derivative, State, explore_map, state_advancer

plt.style.use("dark_background")


def HHpotential(x, y):
    """The Henon-Heiles potential energy."""
    return (1 / 2) * (x**2 + y**2) + x**2 * y - (1 / 3) * y**3


def HHham(local):
    """The Henon-Heiles Hamiltonian."""
    _, [x, y], [px, py] = local
    return (1 / 2) * (px**2 + py**2) + HHpotential(x, y)


def HHsysder():
    return Hamiltonian_to_state_derivative(HHham)


def HHmap(E, dt, sec_eps):
    adv = state_advancer(HHsysder)

    def sysmap(qp):
        y, py = qp
        st = section_to_state(E, y, py)

        cross_st = find_next_crossing(st, dt, adv, sec_eps)
        y, py = cross_st[1][1], cross_st[2][1]
        return jnp.array([y, py])

    return sysmap


def section_to_state(E, y, py):
    d = E - HHpotential(0, y) - (1 / 2) * py**2
    d = jax.lax.select(d <= 0, jnp.nan, d)

    px = jnp.sqrt(2 * d)
    return State(jnp.array(0.0), jnp.array([0.0, y]), jnp.array([px, py]))


def find_next_crossing(st, dt, adv, sec_eps):
    def crossed(st, next_st):
        x = st[1][0]
        next_x = next_st[1][0]
        has_crossed = jnp.logical_and(x < 0, next_x > 0)
        is_nan = jnp.logical_or(jnp.isnan(x), jnp.isnan(next_x))
        return jnp.logical_or(has_crossed, is_nan)

    st, next_st = jax.lax.while_loop(
        lambda val: jnp.logical_not(crossed(val[0], val[1])),
        lambda val: (val[1], adv(val[1], dt)),
        (st, adv(st, dt)),
    )

    return refine_crossing(st, adv, sec_eps)


def refine_crossing(st, adv, sec_eps):
    def cond(st):
        x = jnp.abs(st[1][0])
        cond = x > sec_eps
        return cond

    def body(st):
        x, xd = st[1][0], st[2][0]
        next_st = adv(st, -x / xd)
        return next_st

    return jax.lax.while_loop(cond, body, st)


E = 1 / 10

plt.xlim(-1, 1)
plt.ylim(-1, 1)
evolution = explore_map(
    plt.gcf(),
    HHmap(E, 0.1, 1e-12),
    128,
    xs=[-0.4, -0.3, -0.25, -0.10, -0.05, 0.05, 0.10, 0.25, 0.3, 0.4],
    ys=[-0.25, -0.10, 0.0, 0.10, 0.25],
)
# plt.show()
# input()


alt.themes.enable("fivethirtyeight")
selector = alt.selection_point(fields=["id"])

section_data = []
for ev, evol in enumerate(np.asarray(evolution)):
    for id, (y, py) in enumerate(evol):
        section_data.append({"ev": ev, "id": id, "y": y, "py": py})

section = (
    alt.Chart(alt.Data(values=section_data))
    .mark_circle(size=10)
    .encode(
        x="y:Q",
        y="py:Q",
        color=alt.condition(
            selector,
            alt.Color("id:N").scale(scheme="dark2").legend(None),
            alt.value("lightgray"),
        ),
    )
    .add_params(selector)
    .properties(width=400, height=400)
    .interactive()
)

dstate = HHsysder()
func = lambda y, t: dstate(y)
t = jnp.linspace(0, 50, 256)

traj_data = []
for id, (y, py) in enumerate(evolution[0]):
    y0 = section_to_state(E, y, py)
    sol = odeint(func, y0, t)
    xys = np.asarray(sol[1])
    for i in range(len(xys)):
        traj_data.append({"t": i, "id": id, "x": xys[i, 0], "y": xys[i, 1]})

traj = (
    alt.Chart(alt.Data(values=traj_data))
    .mark_trail()
    .encode(
        x="x:Q",
        y="y:Q",
        order="t:N",
        color=alt.Color("id:N").scale(scheme="dark2").legend(None),
        size=alt.Size("t:N").legend(None),
    )
    .transform_filter(selector)
    .add_params(selector)
    .properties(width=400, height=400)
    .interactive()
)


chart = section | traj
chart.save("__demo.html")
# chart.save("__demo.json")


# # Animate
#
# xlim = (-2, 2)
# ylim = (-2, 2)
# plt.xlim(xlim)
# plt.ylim(ylim)
#
# xx, yy = np.meshgrid(np.linspace(*xlim, 100), np.linspace(*ylim, 100))
# vs = HHpotential(xx, yy)
# plt.contour(xx, yy, vs, levels=[E, 1.5 * E], cmap="Dark2")
#
# (traj,) = plt.plot([], [], "w--", lw=1)
# dot = plt.scatter([], [], s=10, c="w")
#
#
# def update(i):
#     i0 = max(0, i - 1000)
#     traj.set_data(xys[i0:i, 0], xys[i0:i, 1])
#     dot.set_offsets(xys[i])
#     return traj, dot
#
#
# anim = FuncAnimation(
#     plt.gcf(),
#     update,
#     init_func=lambda: update(0),
#     frames=range(0, len(t), 20),
#     interval=20,
#     blit=True,
# )
# plt.show()
