import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import lagrange


def directed(angle):
    return 0.7 * np.array([np.cos(angle), np.sin(angle)])


def arrow(xy, dxy, **kwargs):
    plt.arrow(xy[0], xy[1], dxy[0], dxy[1], **kwargs)


def line(xy, dxy, **kwargs):
    plt.plot([xy[0], xy[0] + dxy[0]], [xy[1], xy[1] + dxy[1]], **kwargs)


def text(txt, xy, **kwargs):
    plt.text(xy[0], xy[1], txt, **kwargs)


angle = -np.pi / 6
scale = 0.7
py_vec = directed(angle)
y_vec = directed(angle - np.pi / 2)

colors = ["#ddd", "gray"]

# draw py
arrow([0, 0], py_vec, color=colors[0], lw=1.5, head_width=0.05)
line(
    scale * y_vec,
    scale * py_vec,
    color=colors[1],
    lw=1.0,
    ls="--",
)
text(r"$p_y$", py_vec + np.array([0, 0.1]), color=colors[0], fontsize=16)

# draw y
arrow([0, 0], y_vec, color=colors[0], linewidth=1.5, head_width=0.05)
line(
    scale * py_vec,
    scale * y_vec,
    color=colors[1],
    lw=1.0,
    ls="--",
)
text(r"$y$", y_vec + np.array([0, 0.1]), color=colors[0], fontsize=16)

# x
arrow([0, 0], [0, 1], color=colors[0], linewidth=1.5, head_width=0.05)
text(r"$x$", [-0.1, 1], color=colors[0], fontsize=16)


# draw spiral crossing y-py
def interp(x, xp, yp):
    return lagrange(xp, yp)(x)


zero = -0.25
scatter = [[], []]


# 1
x = np.linspace(-0.05, 0.02, 100)
y = interp(x, [-0.05, -0.01, 0.02], [zero, zero + 0.1, zero])
plt.plot(x, y, color=colors[0], lw=1.5)
plt.annotate(
    "",
    [x[0] - 0.01, y[0] - 0.04],
    [x[0], y[0]],
    arrowprops=dict(arrowstyle="->", color=colors[0], lw=1.5),
)

scatter[0].append(x[-1])
scatter[1].append(y[-1])

# 2
x = np.linspace(-0.02, 0.02, 100)
y = interp(
    x,
    [-0.02, -0.01, 0.00, 0.01, 0.02],
    [zero, zero - 0.08, zero - 0.1, zero - 0.11, zero],
)
plt.plot(x, y, color=colors[1], lw=1.5, ls="--")

# 3
x = np.linspace(-0.02, 0.1, 100)
y = interp(
    x,
    [-0.02, 0.02, 0.04, 0.7, 0.1],
    [zero, zero + 0.12, zero + 0.13, zero + 0.1, zero],
)
plt.plot(x, y, color=colors[0], lw=1.5)

scatter[0].append(x[-1])
scatter[1].append(y[-1])

# 4
x = np.linspace(0.05, 0.1, 100)
y = interp(x, [0.05, 0.07, 0.09, 0.1], [zero, zero - 0.15, zero - 0.1, zero])
plt.plot(x, y, color=colors[1], lw=1.5, ls="--")

# 5
x = np.linspace(0.05, 0.15, 100)
y = interp(
    x,
    [0.05, 0.08, 0.1, 0.13, 0.15],
    [zero, zero + 0.1, zero + 0.12, zero + 0.1, zero],
)
plt.plot(x, y, color=colors[0], lw=1.5)

plt.scatter(scatter[0], scatter[1], color=colors[0], s=20)

# show
plt.grid(False)
plt.axis("off")
plt.savefig("section.png", transparent=True, dpi=300)
plt.show()
