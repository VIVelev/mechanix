import altair as alt
import numpy as np
from hh import HHpotential
from matplotlib import pyplot as plt

alt.themes.enable("fivethirtyeight")
plt.ion()

xx, yy = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
vs = HHpotential(xx, yy)
levels = [1 / 100, 1 / 40, 1 / 20, 1 / 12, 1 / 8, 1 / 6]
cs = plt.contour(xx, yy, vs, levels)

data = []
for l, path in zip(levels, cs.get_paths()):
    data.extend(
        {"x": x, "y": y, "t": i, "Energy": l} for i, (x, y) in enumerate(path.vertices)
    )

ch = (
    alt.Chart(alt.Data(values=data))
    .mark_line()
    .encode(
        x=alt.X("x:Q").axis(titleColor="#ddd"),
        y=alt.Y("y:Q").axis(titleColor="#ddd").scale(domain=[-0.6, 1.1]),
        order="t:Q",
        color=alt.Color("Energy:Q").scale(scheme="spectral").legend(None),
        tooltip=["Energy:Q"],
    )
    .interactive()
    .properties(width=300, height=300)
    .configure(background="#151515", view=dict(stroke=None))
)
ch.save("__potential.html")
ch.save("__potential.json")
