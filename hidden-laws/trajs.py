import altair as alt
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hh import HHpotential, HHsysder, section_to_state
from jax.experimental.ode import odeint

alt.themes.enable("fivethirtyeight")

E = 1 / 8

dstate = HHsysder()
func = lambda y, t: dstate(y)
t = jnp.linspace(0, 200, 1024)

# Contour line
xx, yy = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
vs = HHpotential(xx, yy)
levels = [E]
cs = plt.contour(xx, yy, vs, levels)
data = []
for l, path in zip(levels, cs.get_paths()):
    data.extend(
        {"x": x, "y": y, "t": i, "Energy": l} for i, (x, y) in enumerate(path.vertices)
    )
contour = (
    alt.Chart(alt.Data(values=data))
    .mark_line()
    .encode(
        x=alt.X("x:Q").axis(titleColor="#ddd"),
        y=alt.Y("y:Q").axis(titleColor="#ddd"),
        order="t:Q",
        color=alt.value("white"),
    )
)


def gentrajchart(E, y, py):
    y0 = section_to_state(E, y, py)
    sol = odeint(func, y0, t)
    xys = np.asarray(sol[1])
    traj_data = [{"t": i, "x": x, "y": y} for i, (x, y) in enumerate(xys)]

    return (
        (
            alt.Chart(alt.Data(values=traj_data))
            .mark_trail()
            .encode(
                x=alt.X("x:Q").axis(titleColor="#ddd"),
                y=alt.Y("y:Q").axis(titleColor="#ddd"),
                order="t:N",
                color=alt.value("orange"),
            )
        )
        + contour
    ).properties(width=300, height=300)


chart = (
    (
        gentrajchart(E, 0.25, 0.08)
        | gentrajchart(E, -0.1, 0.21)
        | gentrajchart(E, 0.4, 0.3)
    )
    .configure(background="#151515", view=dict(stroke=None))
    .interactive()
)

chart.save("__traj.html")
chart.save("__traj.json")
