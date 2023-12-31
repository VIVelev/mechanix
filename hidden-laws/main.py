from functools import partial

import altair as alt
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.ode import odeint
from r3bp import (
    GM0,
    GM1,
    R3BPmap,
    R3BPsysder,
    _a0 as a0,
    _a1 as a1,
    section_to_state as r3bp_section_to_state,
)
from tqdm import tqdm

from mechanix import evolve_map

jax.config.update("jax_enable_x64", True)
alt.themes.enable("fivethirtyeight")
plt.style.use("dark_background")

NUM_STEPS = 1024


def section(
    xys,
    sysmap,
    selector,
    cutoff_step,
    *,
    num_steps=NUM_STEPS,
    xlabel="x",
    xdomain=(-1, 1),
    ylabel="y",
    ydomain=(-1, 1),
):
    print("Evolving section...")
    evolution = evolve_map(xys, sysmap, num_steps)
    print("Done.")
    section_data = []
    for ev, evol in enumerate(np.asarray(evolution)):
        for id, (x, y) in enumerate(evol):
            section_data.append({"t": ev, "id": id, xlabel: x, ylabel: y})

    ch = (
        alt.Chart(alt.Data(values=section_data))
        .mark_circle(size=10)
        .encode(
            x=alt.X(xlabel + ":Q").scale(domain=xdomain),
            y=alt.Y(ylabel + ":Q").scale(domain=ydomain),
            color=alt.condition(
                selector,
                alt.Color("id:N").scale(scheme="paired").legend(None),
                alt.value("#424242"),
            ),
        )
        .add_params(selector)
        .properties(width=600, height=400)
    )

    if cutoff_step is not None:
        ch = ch.transform_filter(alt.datum.t < cutoff_step).add_params(cutoff_step)
    return ch


def trajectories(
    init_states,
    get_sysder,
    selector,
    cutoff_step,
    *,
    num_steps=NUM_STEPS,
    final_time=200,
    xdomain=(-1, 1),
    ydomain=(-1, 1),
):
    dstate = get_sysder()
    func = lambda y, t: dstate(y)
    t = jnp.linspace(0, final_time, num_steps)
    traj_data = []
    print("Evolving trajectories...")
    for id, y0 in enumerate(tqdm(list(init_states))):
        sol = odeint(func, y0, t)
        xys = np.asarray(sol[1])
        for i in range(len(xys)):
            traj_data.append({"t": i, "id": id, "x": xys[i, 0], "y": xys[i, 1]})
    print("Done.")

    return (
        alt.Chart(alt.Data(values=traj_data))
        .mark_trail()
        .encode(
            x=alt.X("x:Q").scale(domain=xdomain),
            y=alt.Y("y:Q").scale(domain=ydomain),
            order="t:N",
            color=alt.Color("id:N").scale(scheme="paired").legend(None),
        )
        .transform_filter(selector)
        .transform_filter(alt.datum.t < cutoff_step)
        .add_params(selector, cutoff_step)
        .properties(width=400, height=400)
    )


selector = alt.selection_point(fields=["id"])
step_slider = alt.binding_range(min=0, max=NUM_STEPS, step=1, name="Step")
cutoff_step = alt.param(bind=step_slider, value=NUM_STEPS // 4)

# E = 1 / 8
# xys = cartesian_product([-0.3, 0.0, 0.1, 0.2], [-0.25, -0.05, 0.0, 0.3])
# s = section(
#     xys,
#     HHmap(E, 0.01, 1e-12),
#     selector,
#     cutoff_step,
#     xlabel="y",
#     ylabel="py",
# ).interactive()
# t = trajectories(xys, HHsysder, partial(hh_section_to_state, E)).interactive()

J = 3.0
# fig = plt.figure(1)
# fig.gca().set_xlim(-1, 1)
# fig.gca().set_ylim(-1, 1)
# xys = explore_map(plt.figure(1), R3BPmap(J, 0.01, 1e-12), NUM_STEPS)
# np.save("samples.npy", xys)

xys = np.load("samples.npy")
print("Loaded samples of shape: ", xys.shape)

s = section(
    xys,
    R3BPmap(J, 0.01, 1e-12),
    selector,
    cutoff_step,
    xlabel="y",
    xdomain=(-1, 0.5),
    ylabel="py",
    ydomain=(-2.3, 2.3),
).interactive()

chart = s.configure(
    background="#151515",
    axis=dict(titleColor="#ddd"),
    view=dict(stroke=None),
)

chart.save("__main_section.html")
chart.save("__main_section.json")

t = trajectories(
    map(partial(r3bp_section_to_state, J), xys[:, 0], xys[:, 1]),
    R3BPsysder,
    selector,
    cutoff_step,
).interactive()

total = (GM0 + GM1) / 1000
bodies = (
    alt.Chart(
        alt.Data(
            values=[
                {"x": -a0, "y": 0, "mass": GM0 / total},
                {"x": a1, "y": 0, "mass": GM1 / total},
            ]
        )
    )
    .mark_circle()
    .encode(x="x:Q", y="y:Q", size=alt.Size("mass:Q").legend(None))
)

chart = (s | (t + bodies)).configure(
    background="#151515",
    axis=dict(titleColor="#ddd"),
    view=dict(stroke=None),
)

chart.save("__main.html")
chart.save("__main.json")
