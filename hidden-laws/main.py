from functools import reduce

import altair as alt
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
from utils import HHmap, HHsysder, section_to_state

from mechanix import evolve_map

alt.themes.enable("fivethirtyeight")


def get_domain(xfield, yfield, data, *, scale=1.1):
    min_predicates = lambda m, c: c if c <= m else m
    max_predicates = lambda m, c: c if c >= m else m
    xs = list(map(lambda x: x[xfield], data))
    ys = list(map(lambda x: x[yfield], data))
    min_x = reduce(min_predicates, xs) * scale
    max_x = reduce(max_predicates, xs) * scale
    min_y = reduce(min_predicates, ys) * scale
    max_y = reduce(max_predicates, ys) * scale
    return [min_x, max_x], [min_y, max_y]


def section(xs, ys, sysmap, *, num_steps=1024):
    selector = alt.selection_point(fields=["id"])

    evolution = evolve_map(xs, ys, sysmap, num_steps)
    section_data = []
    for ev, evol in enumerate(np.asarray(evolution)):
        for id, (y, py) in enumerate(evol):
            section_data.append({"ev": ev, "id": id, "y": y, "py": py})

    xdomain, ydomain = get_domain("y", "py", section_data)
    return (
        alt.Chart(alt.Data(values=section_data))
        .mark_circle(size=10)
        .encode(
            x=alt.X("y:Q").scale(domain=xdomain).axis(titleColor="#ddd"),
            y=alt.Y("py:Q").scale(domain=ydomain).axis(titleColor="#ddd"),
            color=alt.condition(
                selector,
                alt.Color("id:N").scale(scheme="paired").legend(None),
                alt.value("#424242"),
            ),
        )
        .add_params(selector)
        .properties(width=400, height=400)
    )


def explore_map_traj(
    xs,
    ys,
    sysmap,
    get_sysder,
    section_to_state,
    *,
    num_steps=1024,
    final_time=200,
):
    selector = alt.selection_point(fields=["id"])
    step_slider = alt.binding_range(min=0, max=num_steps, step=1, name="Step")
    cutoff_step = alt.param(bind=step_slider, value=num_steps / 4)

    evolution = evolve_map(xs, ys, sysmap, num_steps)
    section_data = []
    for ev, evol in enumerate(np.asarray(evolution)):
        for id, (y, py) in enumerate(evol):
            section_data.append({"ev": ev, "id": id, "y": y, "py": py})

    xdomain, ydomain = get_domain("y", "py", section_data)
    section = (
        alt.Chart(alt.Data(values=section_data))
        .mark_circle(size=10)
        .encode(
            x=alt.X("y:Q").scale(domain=xdomain).axis(titleColor="#ddd"),
            y=alt.Y("py:Q").scale(domain=ydomain).axis(titleColor="#ddd"),
            color=alt.condition(
                selector,
                alt.Color("id:N").scale(scheme="paired").legend(None),
                alt.value("#424242"),
            ),
        )
        .transform_filter(alt.datum.ev < cutoff_step)
        .add_params(selector, cutoff_step)
        .properties(width=400, height=400)
    )

    dstate = get_sysder()
    func = lambda y, t: dstate(y)
    t = jnp.linspace(0, final_time, num_steps)
    traj_data = []
    for id, (y, py) in enumerate(evolution[0]):
        y0 = section_to_state(E, y, py)
        if jnp.any(jnp.isnan(y0[2])):
            print(y, py)
            continue

        sol = odeint(func, y0, t)
        xys = np.asarray(sol[1])
        for i in range(len(xys)):
            traj_data.append({"t": i, "id": id, "x": xys[i, 0], "y": xys[i, 1]})

    xdomain, ydomain = get_domain("x", "y", traj_data)
    traj = (
        alt.Chart(alt.Data(values=traj_data))
        .mark_trail()
        .encode(
            x=alt.X("x:Q").scale(domain=xdomain).axis(titleColor="#ddd"),
            y=alt.Y("y:Q").scale(domain=ydomain).axis(titleColor="#ddd"),
            order="t:N",
            color=alt.Color("id:N").scale(scheme="paired").legend(None),
        )
        .transform_filter(selector)
        .transform_filter(alt.datum.t < cutoff_step)
        .add_params(selector, cutoff_step)
        .properties(width=400, height=400)
    )

    return (section.interactive() | traj.interactive()).configure(
        background="#151515", view=dict(stroke=None)
    )


E = 1 / 8
chart = explore_map_traj(
    [-0.3, 0.0, 0.1, 0.2],
    [-0.25, -0.05, 0.0, 0.3],
    HHmap(E, 0.01, 1e-12),
    HHsysder,
    section_to_state,
)
chart.save("__main.html")
chart.save("__main.json")
