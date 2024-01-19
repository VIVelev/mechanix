import operator
from collections import defaultdict
from functools import reduce
from numbers import Number

import altair as alt
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from r3bp import GM0, GM1, Jacobi, a, a0, a1


def contour_chart(
    energy,
    xlim,
    ylim,
    *,
    label="Energy",
    levels=None,
    n=512,
    state_func=False,
    xdomain=(-1, 1),
    ydomain=(-1, 1),
    **properties,
):
    if isinstance(xlim, Number):
        xlim = (-xlim, xlim)
    if isinstance(ylim, Number):
        ylim = (-ylim, ylim)

    xx, yy = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    if state_func:
        xys = jnp.stack([xx, yy], axis=-1)
        vs = jnp.zeros_like(xys)
        ts = jnp.zeros((n, n))
        states = (ts, xys, vs)
        zs = np.array(jax.vmap(jax.vmap(energy))(states))
    else:
        zs = energy(xx, yy)

    if levels is not None:
        cs = plt.contour(xx, yy, zs, levels=levels)
    else:
        cs = plt.contour(xx, yy, zs)
        levels = cs.levels

    data = defaultdict(list)
    for l, segments in zip(levels, cs.allsegs):
        for i, segment in enumerate(segments):
            data[i].extend(
                {"x": x, "y": y, "t": j, label: l} for j, (x, y) in enumerate(segment)
            )

    properties = {**dict(width=400, height=400), **properties}
    charts = [
        alt.Chart(alt.Data(values=cs))
        .mark_line()
        .encode(
            x=alt.X("x:Q").scale(domain=xdomain),
            y=alt.Y("y:Q").scale(domain=ydomain),
            order="t:Q",
            color=alt.Color(label + ":Q")
            .scale(scheme="spectral", domain=[min(*levels), max(*levels)])
            .legend(None),
            tooltip=alt.Tooltip(label + ":Q", format=".4f"),
        )
        .properties(**properties)
        .interactive()
        for cs in data.values()
    ]
    return reduce(operator.add, charts)


if __name__ == "__main__":
    alt.themes.enable("fivethirtyeight")

    # ch = contour_chart(
    #     HHpotential,
    #     1,
    #     1,
    #     levels=[1 / 100, 1 / 40, 1 / 20, 1 / 12, 1 / 8, 1 / 6],
    #     ydomain=(-0.6, 1.1),
    #     width=300,
    #     height=300,
    # )

    ch = contour_chart(
        Jacobi,
        1.5 * a,
        1.5 * a,
        label="Jacobi",
        levels=np.concatenate(
            (
                np.linspace(2.9, 3.1, 10),
                np.linspace(3.2, 3.6, 3),
            )
        ),
        state_func=True,
        xdomain=(-1.2 * a, 1.2 * a),
        ydomain=(-1.2 * a, 1.2 * a),
    )
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
    ch = ch + bodies

    ch = ch.configure(
        background="#151515",
        axis=dict(titleColor="#ddd"),
        view=dict(stroke=None),
    )
    ch.save("potential.json")
