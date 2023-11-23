import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np

plt.style.use("cyberpunk")


def V(x, y):
    return (1 / 2) * (x**2 + y**2) + x**2 * y - (1 / 3) * y**3


xs, ys = np.meshgrid(np.linspace(-2, 2, 1000), np.linspace(-2, 2, 1000))
vs = V(xs, ys)
vs[vs > 1] = np.nan

plt.gca().set_xlim(-1.1, 1.1)
plt.gca().set_ylim(-0.6, 1.1)
plt.title("Contour plot of Henon-Heiles potential energy")
plt.contour(
    xs,
    ys,
    vs,
    levels=[1 / 100, 1 / 40, 1 / 20, 1 / 12, 1 / 8, 1 / 6, 1],
    cmap="Dark2",
)


mplcyberpunk.add_glow_effects()
plt.show()
