import json
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hh import HHmap, HHsysder, section_to_state as hh_section_to_state
from tqdm import tqdm
from utils import cartesian_product

from mechanix import evolve_map, odeint

jax.config.update("jax_enable_x64", True)
plt.style.use("dark_background")


def section(xys, sysmap, num_steps):
    """Generate the surface-of-section data, given
    a Poincare-map `sysmap` and initial points on the section `xys`
    """

    print("Evolving section...")
    evolution = evolve_map(xys, sysmap, num_steps)
    section_data = []
    for step, evol in enumerate(np.asarray(evolution)):
        for id, (x, y) in enumerate(evol):
            section_data.append({"step": step, "id": id, "x": x, "y": y})

    # Save section data to JSON
    json.dump(section_data, open("section-data.json", "w"))
    print("Done.")


def trajectories(init_states, sysder, num_steps, final_time):
    """Generate the trajectories data, given
    a state-derivative `sysder` and initial states `init_states`
    """

    t = jnp.linspace(0, final_time, num_steps)
    traj_data = []
    print("Evolving trajectories...")
    integrate = jax.jit(partial(odeint(sysder, tolerance=1e-10), t=t))
    for id, y0 in enumerate(tqdm(list(init_states))):
        sol = integrate(y0)
        for step, [x, y] in enumerate(np.asarray(sol[1])):
            traj_data.append({"step": step, "id": id, "x": x, "y": y})

    # Save trajectory data to JSON
    json.dump(traj_data, open("traj-data.json", "w"))
    print("Done.")


E = 1 / 8
xys = cartesian_product([-0.3, 0.0, 0.1, 0.2], [-0.25, -0.05, 0.0, 0.3])
s = section(
    xys,
    HHmap(E, 0.01, 1e-10, 1e-10),
    num_steps=2**12,
)
t = trajectories(
    map(partial(hh_section_to_state, E), xys[:, 0], xys[:, 1]),
    HHsysder(),
    num_steps=2**12,
    final_time=128,
)

# J = 3.0

# fig = plt.figure(1)
# fig.gca().set_xlim(-1, 1)
# fig.gca().set_ylim(-1, 1)
# xys = explore_map(
#     plt.figure(1),
#     R3BPmap(
#         J,
#         0.01,
#         1e-10,
#         1e-10,
#         a=a,
#         m=m,
#         GM0=GM0,
#         GM1=GM1,
#     ),
#     1024,
# )
# np.save("samples_2.npy", xys)

# xys = np.load("samples_2.npy")
# print("Loaded samples of shape: ", xys.shape)
# s = section(xys, R3BPmap(J, 0.01, 1e-10, 1e-10, a=a, m=m, GM0=GM0, GM1=GM1), 1024)
# t = trajectories(
#     map(partial(r3bp_section_to_state, J), xys[:, 0], xys[:, 1]),
#     R3BPsysder(a, m, GM0, GM1),
#     2**12,
#     2**5,
# )
