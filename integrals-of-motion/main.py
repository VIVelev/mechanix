import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mechanix import Hamiltonian_to_state_derivative, State, explore_map, state_advancer

# plt.style.use(
#     "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
# )

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

    # return st
    return refine_crossing(st, adv, sec_eps)


def refine_crossing(st, adv, sec_eps):
    def cond(st):
        x = jnp.abs(st[1][0])
        cond = x > sec_eps
        return cond

    def body(st):
        x, xd = st[1][0], st[2][0]
        next_st = adv(st, -x / xd)
        # jax.debug.print("x = {}", x)
        # jax.debug.print("xd = {}", xd)
        # jax.debug.print("next_st = {}", next_st)
        return next_st

    return jax.lax.while_loop(cond, body, st)


plt.xlim(-0.8, 0.8)
plt.ylim(-0.8, 0.8)
explore_map(plt.gcf(), HHmap(0.125, 0.1, 1e-12), 500)
plt.show()

# adv = state_advancer(HHsysder)
# local = State(jnp.array(0.0), jnp.array([0.8, 0.8]), jnp.array([0.0, 0.0]))
# for i in range(100):
#     local = adv(local, 0.1)
#     print(local)

# dstate = HHsysder()
# func = lambda y, t: dstate(y)
# y0 = (jnp.array(0.0), jnp.array([0.8, 0.8]), jnp.array([0.0, 0.0]))
# E = HHham(y0)
# print("E = ", E)
# t = jnp.linspace(0, 100, 1000)
# sol = odeint(func, y0, t)
# xys = sol[1]
#
#
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
