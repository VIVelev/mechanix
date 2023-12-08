from itertools import cycle
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from ._utils import Array


def explore_map(
    fig: plt.Figure,
    sys_map: Callable[[Array], Array],
    n: int,
    *,
    grid_resolution=16,
):
    ax = fig.gca()
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    xs0, ys0 = (
        jnp.linspace(*x_lim, grid_resolution),
        jnp.linspace(*y_lim, grid_resolution),
    )
    xx0, yy0 = jnp.meshgrid(xs0, ys0)

    vvmap = jax.vmap(jax.vmap(sys_map))

    @jax.jit
    def body(carry, _):
        out = vvmap(carry)
        return out, out

    evolution0 = jnp.stack([xx0, yy0], axis=-1)
    _, evolution = jax.lax.scan(body, evolution0, jnp.arange(n), length=n)
    xx, yy = evolution[:, :, :, 0], evolution[:, :, :, 1]

    # make each different color a different trajectory
    color = cycle(mcolors.XKCD_COLORS.values())

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            ax.scatter(xx[:, i, j], yy[:, i, j], c=next(color), s=1)
