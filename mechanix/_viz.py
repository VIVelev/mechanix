from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from ._utils import Array


def cartesian_product(xs, ys):
    """Computes the cartesian product of two arrays."""
    return jnp.transpose(jnp.array([jnp.tile(xs, len(ys)), jnp.repeat(ys, len(xs))]))


# NOTE: For interactive use cases this may be a bit slow ;(
def explore_map(
    fig: plt.Figure,
    sysmap: Callable[[Array], Array],
    n: int,
    *,
    interactive=False,
    grid_size=16,
):
    ax = fig.gca()
    ax.autoscale(False)
    sysmap = jax.vmap(sysmap)
    init_samples = jnp.empty((0, 2), dtype=float)
    evolution = jnp.empty((0, 0, 2), dtype=float)
    # Make a color for each time step (0, ..., n)
    # And enough to cover each trajectory
    colors = np.tile(list(mcolors.XKCD_COLORS.values()), n).reshape(n, -1)
    if not interactive:
        assert colors.shape[1] >= grid_size**2

    def set_samples():
        nonlocal init_samples
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xs, ys = (
            jnp.linspace(*xlim, grid_size),
            jnp.linspace(*ylim, grid_size),
        )
        init_samples = cartesian_product(xs, ys)

    def compute():
        nonlocal evolution

        def body(carry, _):
            out = sysmap(carry)
            return out, out

        _, evolution = jax.lax.scan(body, init_samples, jnp.arange(n), length=n)

    def plot():
        ax.scatter(
            evolution[:, :, 0].flatten(),
            evolution[:, :, 1].flatten(),
            c=colors[:, : len(init_samples)].flatten(),
            s=0.1,
        )

    # BUG: There is some issue with color ("cs")
    # when you spam the "onclick".
    def plot_interactive():
        k = len(init_samples)
        for j in range(1, n):
            xs = evolution[:j, :, 0].flatten()
            ys = evolution[:j, :, 1].flatten()
            cs = colors[:j, :k].flatten()
            ax.scatter(xs, ys, c=cs, s=0.1)
            plt.pause(0.001)

    def onclick(event):
        """Set a sample point."""

        if not event.dblclick:
            return

        nonlocal init_samples
        x, y = event.xdata, event.ydata
        init_samples = jnp.vstack((init_samples, jnp.array([x, y])))
        # NOTE: Colors may run out!
        ax.scatter(
            init_samples[:, 0],
            init_samples[:, 1],
            c=colors[0, : len(init_samples)],
            s=1,
        )

    def onenter(event):
        """Compute and plot."""

        if event.key != "enter":
            return

        compute()
        plot_interactive()

    if interactive:
        fig.canvas.mpl_connect("button_press_event", onclick)  # set sample
        fig.canvas.mpl_connect("key_release_event", onenter)  # compute and plot
        plt.ion()
    else:
        set_samples()
        compute()
        plot()
