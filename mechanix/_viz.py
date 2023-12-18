import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from tqdm import tqdm


def evolve_map(xys, sysmap, n):
    # NOTE: Non vmap'd version is faster!
    # Maybe due to overhead of vmap?
    sysmap = jax.jit(sysmap)
    evolution = jnp.empty((n, 0, 2), dtype=jnp.float_)

    for xy in tqdm(xys):
        current_ev = jnp.array([xy])
        for j in tqdm(range(1, n), leave=False):
            next = sysmap(current_ev[j - 1, :])
            current_ev = jnp.concatenate([current_ev, next[None, ...]])
        evolution = jnp.concatenate([evolution, current_ev[:, None, :]], axis=1)

    return evolution


def explore_map(fig, sysmap, n):
    ax = fig.gca()
    ax.autoscale(False)
    sysmap = jax.jit(jax.vmap(sysmap))
    start_idx = 0
    init_samples = jnp.empty((0, 2), dtype=jnp.float_)
    evolution = jnp.empty((n, 0, 2), dtype=jnp.float_)
    # Make a color for each time step (0, ..., n)
    # And enough to cover each trajectory
    colors = np.tile(list(mcolors.XKCD_COLORS.values()), n).reshape(n, -1)

    def compute():
        nonlocal evolution
        print("Computing...")
        print("Num. Init Samples:", init_samples.shape[0])
        print("Starting computation from sample:", start_idx)

        new_evolution = init_samples[start_idx:][None, ...]
        for i in range(1, n):
            next = sysmap(new_evolution[i - 1, :, :])
            new_evolution = jnp.concatenate([new_evolution, next[None, ...]], axis=0)

        evolution = jnp.concatenate([evolution, new_evolution], axis=1)
        print("Evolution array shape:", evolution.shape)

    def plot_interactive():
        k = len(init_samples)
        xs = evolution[:, start_idx:, 0].flatten()
        ys = evolution[:, start_idx:, 1].flatten()
        cs = colors[:, start_idx:k].flatten()
        ax.scatter(xs, ys, c=cs, s=1)

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
        nonlocal start_idx

        if event.key != "enter" or len(init_samples) == start_idx:
            return

        compute()
        plot_interactive()
        start_idx = len(init_samples)

    fig.canvas.mpl_connect("button_press_event", onclick)  # set sample
    fig.canvas.mpl_connect("key_release_event", onenter)  # compute and plot
    plt.ion()
    plt.show()
    input()
    return init_samples
