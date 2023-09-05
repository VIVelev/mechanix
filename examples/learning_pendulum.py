from functools import partial
from itertools import cycle

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import wandb
from jax.example_libraries import optimizers, stax
from jax.experimental.ode import odeint
from tqdm import tqdm

from mechanix import F2C, Lagrangian_to_state_derivative, Local, compose

# jax.config.update("jax_enable_x64", True)

# This is an experiment on learning to simualte physical
# systems using neural networks. We will start with a simple
# physical system - a pendulum!

# First, we have to generate training data!
# Thus, we build the lagrangian of the pendulum and integrate
# it using odeint.


def T(m):
    def f(local):
        _, _, v = local
        return 0.5 * m * v @ v

    return f


def V(m, g):
    def f(local):
        _, [_, y], _ = local
        return m * g * y

    return f


# System parameters
l = 1.0  # m  # noqa: E741
m = 1.0  # kg
g = 9.81  # m/s^2

# Time parameters
t0 = 0.0  # s
t1 = 10.0  # s
dt = 0.1  # s
ts = jnp.arange(t0, t1, dt)

# Initial conditions (in polar coordinates)
t0 = jnp.array(t0, dtype=float)  # s
q0 = jnp.array([-np.pi / 4])  # m
v0 = jnp.array([0.0])  # m/s
local0 = Local(t0, q0, v0)


# Convert pendulum coordinates (theta) to rectangular coordinates (x, y)
def pendulum2rect(local):
    theta = local.pos[0]
    return l * jnp.array([jnp.cos(theta), jnp.sin(theta)])


# Lagrangian
L_rectangular = lambda local: T(m)(local) - V(m, g)(local)
L_polar = compose(
    L_rectangular,
    F2C(pendulum2rect),
)
dstate = jax.jit(Lagrangian_to_state_derivative(L_polar))

# Integrate
func = lambda y, t: dstate(y)
locals = odeint(func, local0, ts)

# --------------------------------------------------------------------
# Now we parametrize the lagrangian using a neural network:
# --------------------------------------------------------------------

seed = 42
key = jax.random.PRNGKey(seed)

# The dataset
X = locals
Y = jax.vmap(dstate)(X).v
testset_size = int(0.3 * len(Y))
X_train, X_test = jax.tree_map(lambda x: x[:-testset_size], X), jax.tree_map(
    lambda x: x[-testset_size:], X
)
Y_train, Y_test = Y[:-testset_size], Y[-testset_size:]

# Print the shapes of the data
print("X_train:", jax.tree_map(lambda x: x.shape, X_train))
print("Y_train:", Y_train.shape)


def loader(x, y, batch_size, *, key, shuffle=False):
    def _shuffle(x, y, *, key):
        n = len(y)
        idx = jax.random.permutation(key, n)
        return jax.tree_map(lambda x: x[idx], x), y[idx]

    n = len(y)
    if shuffle:
        x, y = _shuffle(x, y, key=key)
    for i in range(0, n, batch_size):
        idx = jnp.arange(i, min(i + batch_size, n))
        yield jax.tree_map(lambda x: x[idx], x), y[idx]


# The model
nhidden = 16
nlayers = 2
layers = []


def get_init(i, n):
    """Taken from Lagrangian Neural Networks.
    Miles Cranmer, et al.

    i : int - The index of the layer
    n : int - The number of hidden neurons
    """

    sigma = 1 / np.sqrt(n)
    if i == 0:
        sigma *= 2.2
    elif i == -1:
        sigma *= n
    else:
        sigma *= 0.58 * n

    def init(rng, shape, dtype=jnp.float_):
        return jax.random.normal(rng, shape, dtype) * sigma + 0.0

    return init


# Build the layers
for i in range(nlayers):
    layers.append(stax.Dense(nhidden, W_init=get_init(i, nhidden)))
    layers.append(stax.Softplus)
layers.append(stax.Dense(1, W_init=get_init(-1, nhidden)))
nn_init, nn_apply = stax.serial(*layers)

key, subkey = jax.random.split(key)
in_shape = (1 + 1,)
out_shape, nn_params = nn_init(subkey, in_shape)


def forward(nn_params, local):
    # NOTE: Time dependence seems to break the neural network
    input = jnp.concatenate([local.pos, local.v])
    return nn_apply(nn_params, input)[0]


# Loss function
def mae(y, y_pred):
    return jnp.mean(jnp.sum(jnp.abs(y - y_pred), axis=-1))


def rk4step(f, x, dt):
    def mul(tree, scalr):
        return jax.tree_map(lambda x: x * scalr, tree)

    def add(tree1, tree2):
        return jax.tree_map(lambda x, y: x + y, tree1, tree2)

    # f(x) * dt
    k1 = mul(f(x), dt)
    # f(x + 0.5 * k1) * dt
    k2 = mul(f(add(x, mul(k1, 0.5))), dt)
    # f(x + 0.5 * k2) * dt
    k3 = mul(f(add(x, mul(k2, 0.5))), dt)
    # f(x + k3) * dt
    k4 = mul(f(add(x, k3)), dt)
    # x + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return add(x, mul(add(add(add(k1, mul(k2, 2)), mul(k3, 2)), k4), 1 / 6))


def loss(nn_params, x, y):
    lagrangian = partial(forward, nn_params)
    dstate = Lagrangian_to_state_derivative(lagrangian)
    pred_y = jax.vmap(dstate)(x).v
    return mae(y, pred_y)


def train(
    nn_params,
    X_train,
    Y_train,
    X_test,
    Y_test,
    optim,
    lr,
    batch_size,
    steps,
    log_every,
    *,
    key,
):
    wandb.init(
        project="learning_pendulum",
        config={
            "lr": lr,
            "batch_size": batch_size,
            "steps": steps,
            "t0": t0,
            "t1": t1,
            "nhidden": nhidden,
            "nlayers": nlayers,
        },
    )

    opt_init, opt_update, get_params = optim(lr)
    opt_state = opt_init(nn_params)

    @jax.jit
    def step(i, opt_state, x, y):
        params = get_params(opt_state)
        loss_value, grads = jax.value_and_grad(loss)(params, x, y)
        return opt_update(i, grads, opt_state), loss_value

    key, subkey = jax.random.split(key)
    trainset = cycle(loader(X_train, Y_train, batch_size, key=subkey))
    pbar = tqdm(range(steps))
    for i in pbar:
        x, y = next(trainset)
        opt_state, loss_value = step(i, opt_state, x, y)
        if i % log_every == 0:
            nn_params = get_params(opt_state)
            np.save("nn_params.npy", np.asarray(nn_params))
            key, subkey = jax.random.split(key)
            testset = loader(X_test, Y_test, batch_size, key=subkey)
            test_loss = jnp.mean([loss(nn_params, x, y) for x, y in testset])
            pbar.set_description(f"loss: {loss_value:.4f}, test_loss: {test_loss:.4f}")
            wandb.log({"loss": loss_value, "test_loss": test_loss})

    return get_params(opt_state)


nn_params = train(
    nn_params,
    X_train,
    Y_train,
    X_test,
    Y_test,
    optim=optimizers.adam,
    lr=3e-4,
    batch_size=32,
    steps=10_000_000,
    log_every=1000,
    key=key,
)
print("DONE!")

# Integrate
lagrangian = partial(forward, nn_params)
dstate = jax.jit(Lagrangian_to_state_derivative(lagrangian))
y = local0
ys = [y]
for tprev, tnext in zip(ts[:-1], ts[1:]):
    y = rk4step(dstate, y, tnext - tprev)
    ys.append(y)
    print(f"At time {tnext} obtained value {y}")

locals = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *ys)
X = np.asarray(jax.vmap(pendulum2rect)(locals))

# Visualize

fig, ax = plt.subplots()
ax.set_xlim(-1.2 * l, 1.2 * l)  # type: ignore
ax.set_ylim(-1.2 * l, 1.2 * l)  # type: ignore
ax.set_aspect("equal")  # type: ignore
ax.grid()  # type: ignore

(line,) = ax.plot([], [], "-o", lw=2)  # type: ignore


def animate(i):
    i0 = max(0, i - 10)
    line.set_data(X[i0:i, 0], X[i0:i, 1])
    return (line,)


ani = animation.FuncAnimation(
    fig,
    animate,
    init_func=lambda: animate(0),
    frames=len(ts),
    interval=1000 * dt,
    blit=True,
    repeat=True,
)
plt.show()
