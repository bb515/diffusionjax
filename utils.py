import jax.numpy as jnp
import jax
# enable 64 precision
# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
from jax.experimental.host_callback import id_print
import matplotlib.pyplot as plt
from jax.lax import scan
from jax import grad, jit, vmap
import jax.random as random
from functools import partial
from scipy.stats import norm
from scipy.stats import qmc
from jax.scipy.special import logsumexp
from jax.scipy.linalg import cholesky
from jax.scipy.linalg import solve_triangular
import flax.linen as nn
import optax
import scipy
import seaborn as sns
sns.set_style("darkgrid")
cm = sns.color_palette("mako_r", as_cmap=True)
import numpy as np  # for plotting
# For sampling from MVN
from mlkernels import Linear, EQ
import lab as B
import matplotlib.animation as animation


@partial(jax.jit)
def matrix_solve(L_cov, b):
    x = solve_triangular(
        L_cov, b, lower=True)
    return x

# def S_given_t(mf, t, m_0, C_0):
#     N = mf.shape[0]
#     mean_coeff = mean_factor(t)
#     mean = m_0 * mean_coeff
#     v = var(t)
#     L_cov = matrix_cho(mean_coeff**2 * C_0 + v)
#     return L_cov, mean


# def log_pt_factored_t(x, L_cov, mean):
#     """
#     Analytical distribution score for normal distribution on the hyperplane.
#     Requires a linear solve for each x
#     Requires a cholesky for each t

#     x: One location in R^n, N should be 2 in special case example.
#     t: time    
#     """
#     return -matrix_solve(L_cov, x - mean)




def log_hat_pt(x, t, mf):
    """
    Empirical distribution score for normal distribution on the hyperplane.

    x: One location in R^n, N should be 2 in special case example
    t: time
    """
    N = mf.shape[0]
    means = mf * mean_factor(t)
    v = var(t)
    potentials = jnp.sum(-(x - means)**2 / (2 * v), axis=1)  # doesn't potential also depends on jnp.log(_v_t)? but taking nabla_x
    return logsumexp(potentials, axis=0, b=1/N)


def nabla_log_pt_scalar(x, t, m_0, C_0):
    """
    Analytical distribution score for normal distribution on the hyperplane.
    Requires a linear solve for each x
    Requires a cholesky for each t

    x: One location in R^n, N should be 2 in special case example.
    t: time    
    """
    N = m_0.shape[0]
    mean_coeff = mean_factor(t)
    mean = m_0 * mean_coeff
    v = var(t)
    mat = mean_coeff**2 * C_0 + v
    if N == 1:
        return (x - m_0) / mat
    else:
        L_cov = matrix_cho(mean_coeff**2 * C_0 + v)
        return -matrix_solve(L_cov, x - mean)


def nabla_log_pt_scalar_hyperplane(x, t):
    N = x.shape[0]
    mean_coeff = mean_factor(t)**2
    v = var(t)
    mat = mean_coeff * jnp.array([[1, 0], [0, 0]]) + v * jnp.identity(N)
    if N == 1:
        return (x) / mat
    else:
        L_cov = matrix_cho(mat)
        return -matrix_solve(L_cov, x)


def sample_mvn(J, N, kernel=Linear(), m_0=0.0):
    """
    J: How many samples to generate
    N: Dimension of the embedding space
    Returns a (J, N) array of samples
    """
    rng = random.PRNGKey(123)
    rng, step_rng = jax.random.split(rng)  # the missing line
    # sample from an input space
    z = np.array(random.normal(step_rng, (N,)))
    print(N)
    C_0 = kernel(z)
    C_0 = B.dense(C_0)
    print(m_0)
    print(C_0)
    manifold = scipy.stats.multivariate_normal.rvs(mean=m_0, cov=C_0, size=J)
    # Normalization is necessary because of the unit prior distribution
    manifold = manifold - jnp.mean(manifold, axis=0)
    manifold = manifold / jnp.max(jnp.var(manifold, axis=0))
    return manifold, m_0, C_0, z


def sample_sphere(J, M, N):
    """
    J: How many samples to generate
    M: Dimension of the submanifold
    N: Dimension of the embedding space
    Returns a (J, N) array of samples
    """
    assert M <= N-1
    #Implement for other dimensions than two
    dist = scipy.stats.qmc.MultivariateNormalQMC(mean=jnp.zeros(M+1))
    manifold = jnp.array(dist.random(J))
    norms = jnp.sqrt(jnp.sum(manifold**2, axis=1).reshape((J, 1)))
    manifold = manifold/norms
    norms = jnp.sum(manifold**2, axis=1).reshape((J, 1))
    if M+1 < N:
        manifold = jnp.concatenate([manifold, jnp.zeros((J, N-(M+1)))], axis=1)
    #normalization
    manifold = manifold - jnp.mean(manifold, axis=0)
    manifold = manifold / jnp.max(jnp.var(manifold, axis=0))
    return manifold


def sample_hyperplane(J, M, N):
    """
    J: How many samples to generate
    M: Dimension of the submanifold
    N: Dimension of the embedding space
    Returns a (J, N) array of samples
    """
    assert M <= N
    dist = scipy.stats.qmc.MultivariateNormalQMC(mean=jnp.zeros(M))  # is this better than other samplers
    manifold = jnp.array(dist.random(J))
    if M < N:
        manifold = jnp.concatenate([manifold, jnp.zeros((J, N-(M)))], axis=1)
    manifold = manifold - jnp.mean(manifold, axis=0)
    manifold = manifold / jnp.max(jnp.var(manifold, axis=0))
    return manifold


def beta_t(t):
    """
    t: time (number)
    returns beta_t as explained above
    """
    return beta_min + t * (beta_max - beta_min)


def alpha_t(t):
    """
    t: time (number)
    returns alpha_t as explained above
    """
    return t * beta_min + 0.5 * t**2 * (beta_max - beta_min)


def drift(x, t):
    """
    x: location of J particles in N dimensions, shape (J, N)
    t: time (number)
    returns the drift of a time-changed OU-process for each batch member, shape (J, N)
    """
    _beta = beta_t(t)
    return - 0.5 * _beta * x


def dispersion(t):
    """
    t: time (number)
    returns the dispersion
    """
    _beta = beta_t(t)
    return jnp.sqrt(_beta)


def mean_factor(t):
    """
    t: time (number)
    returns m_t as above
    """
    _alpha_t = alpha_t(t)
    return jnp.exp(-0.5 * _alpha_t)


def var(t):
    """
    t: time (number)
    returns v_t as above
    """
    _alpha_t = alpha_t(t)
    return 1.0 - jnp.exp(-_alpha_t)


def forward_potential(x_0, x, t):
    # evaluate density of marginal for a single datapoint on x_0 = [0]^T
    mean_coeff = mean_factor(t)
    v = var(t)
    x = x.reshape(-1, 1)
    density = (x - mean_coeff * x_0) / v
    return density


def forward_density(x_0, x, t):
    mean_coeff = mean_factor(t)
    v = var(t)
    x = x.reshape(-1, 1)
    return norm.pdf(x, loc = mean_coeff * x_0, scale=jnp.sqrt(v))


def loss_fn(params, model, rng, batch):
    """
    params: the current weights of the model
    model: the score function
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    
    returns an random (MC) approximation to the loss \bar{L} explained above
    """
    rng, step_rng = random.split(rng)
    n_batch = batch.shape[0]
    time_samples = random.randint(step_rng, (n_batch, 1), 1, R) / (R - 1)  # why these not independent? I guess that they can be? (n_samps,)
    mean_coeff = mean_factor(time_samples)  # (n_batch, N)
    v = var(time_samples)  # (n_batch, N)
    stds = jnp.sqrt(v)
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, batch.shape)
    x_t = mean_coeff * batch + stds * noise # (n_batch, N)
    s = model.apply(params, x_t, time_samples)
    loss = jnp.mean((noise + s * v)**2)
    return loss


def loss_fn_t(t, params, model, rng, batch):
    """
    params: the current weights of the model
    model: the score function
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    
    returns an random (MC) approximation to the loss \bar{L} explained above
    """
    rng, step_rng = random.split(rng)
    n_batch = batch.shape[0]
    times = jnp.ones((n_batch, 1)) * t
    mean_coeff = mean_factor(times)  # (n_batch, N)
    v = var(times)  # (n_batch, N)
    stds = jnp.sqrt(v)
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, batch.shape)
    x_t = mean_coeff * batch + stds * noise # (n_batch, N)
    s = model.apply(params, x_t, times)
    loss = jnp.mean((noise + s * v)**2)
    return loss


class ApproximateScore(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""

    @nn.compact
    def __call__(self, x, t):
        in_size = x.shape[1]
        n_hidden = 256
        act = nn.relu
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)],axis=1)
        x = jnp.concatenate([x, t],axis=1)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x


class ApproximateScoreLinear(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""

    @nn.compact
    def __call__(self, x, t):
        batch_size = x.shape[0]
        N = jnp.shape(x)[1]
        in_size = (N + 1) * N
        n_hidden = 256
        s = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=1)
        s = nn.Dense(n_hidden)(s)
        s = nn.relu(s)
        s = nn.Dense(n_hidden)(s)
        s = nn.relu(s)
        s = nn.Dense(n_hidden)(s)
        s = nn.relu(s)
        s = nn.Dense(in_size)(s)
        s = jnp.reshape(s, (batch_size, N + 1, N))  # (batch_size, N + 1, N)
        #print(s[0, :, 1])
        #print(s[0, :, 1:])
        x = jnp.einsum('ijk, ij -> ik', s, jnp.hstack((jnp.ones((batch_size, 1)), x)))
        return x  # (n_batch,)

    @nn.compact
    def evaluate_eig(self, x, t):
        batch_size = x.shape[0]
        N = jnp.shape(x)[1]
        in_size = (N + 1) * N
        n_hidden = 256
        s = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=1)
        s = nn.Dense(n_hidden)(s)
        s = nn.relu(s)
        s = nn.Dense(n_hidden)(s)
        s = nn.relu(s)
        s = nn.Dense(n_hidden)(s)
        s = nn.relu(s)
        s = nn.Dense(in_size)(s)
        s = jnp.reshape(s, (batch_size, N + 1, N))  # (batch_size, N + 1, N)
        #print(s[0, :, 1])
        #print(s[0, :, 1:])
        x = jnp.einsum('ijk, ij -> ik', s, jnp.hstack((jnp.ones((batch_size, 1)), x)))
        print(s[0, 0, :])
        print(jnp.linalg.eig(s[0, 1:, :]))
        print(jnp.hstack((jnp.ones((batch_size, 1)), x)))


@partial(jit, static_argnums=[4])
def update_step(params, rng, batch, opt_state, model):
    """
    params: the current weights of the model
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    opt_state: the internal state of the optimizer
    model: the score function

    takes the gradient of the loss function and updates the model weights (params) using it. Returns
    the value of the loss function (for metrics), the new params and the new optimizer state
    """
    val, grads = jax.value_and_grad(loss_fn)(params, model, rng, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, opt_state


@jit
def average_distance_to_hyperplane(samples):
    J = samples.shape[0]
    return jnp.sqrt(1/J * jnp.sum(samples[:, 1:]**2))


def average_distance_to_training_data(mf, samples):
    def one_sample_distance(s, mf):
        dists = jnp.sqrt(jnp.sum((s - mf)**2, axis=1))
        return jnp.min(dists)
    
    all_dists = vmap(one_sample_distance, in_axes=(0, None), out_axes=(0))(samples, mf)
    return jnp.mean(all_dists)


def w1_dd(samples_1, samples_2):
    return scipy.stats.wasserstein_distance(samples_1, samples_2)


def w1_stdnormal(samples):
    J = samples.shape[0]
    true_inv_cdf = jax.scipy.stats.norm.ppf(jnp.linspace(0, 1, J)[1:-1])
    approx_inv_cdf = jnp.sort(samples)[1:-1]
    return jnp.mean(jnp.abs(true_inv_cdf - approx_inv_cdf))


def train_nn(mf, N):
    """Train nn"""
    rng = random.PRNGKey(123)
    ## Neural network training
    batch_size = 16
    #some dummy input data. Flax is able to infer all the dimensions of the weights
    #if we supply if with the kind of input data it has to expect
    x = jnp.zeros(N*batch_size).reshape((batch_size, N))
    time = jnp.ones((batch_size, 1))
    #initialize the model weights
    score_model = ApproximateScore()
    params = score_model.init(rng, x, time)

    #Initialize the optimizer
    opt_state = optimizer.init(params)

    N_epochs = 1000
    train_size = mf.shape[0]
    batch_size = 50
    steps_per_epoch = train_size // batch_size
    for k in range(N_epochs):
        rng, step_rng = random.split(rng)
        perms = jax.random.permutation(step_rng, train_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        losses = []
        for perm in perms:
            batch = mf[perm, :]
            rng, step_rng = random.split(rng)
            loss, params, opt_state = update_step(params, step_rng, batch, opt_state, score_model)
            losses.append(loss)
        mean_loss = jnp.mean(jnp.array(losses))
        if k % 10 == 0:
            print("Epoch %d \t, Loss %f " % (k, mean_loss))
    trained_score = lambda x, t: score_model.apply(params, x, t)
    rng, step_rng = random.split(rng)
    samples = reverse_sde(step_rng, N, 1000, drift, dispersion, trained_score)
    plot_heatmap(samples)
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.savefig("generative_samples.png")


def train_new_nn_on(rng, mf):
        rng, step_rng = random.split(rng)
        score_model = ApproximateScoreLinear()
        train_size = mf.shape[0]
        N = mf.shape[1]
        batch_size = 5
        batch_size = min(train_size, batch_size)
        x = jnp.zeros(N*batch_size).reshape((batch_size, N))
        time = jnp.ones((batch_size, 1))
        params = score_model.init(step_rng, x, time)

        opt_state = optimizer.init(params)
        N_epochs = 100  # may be too little
        steps_per_epoch = train_size // batch_size
        for k in range(N_epochs):
            rng, step_rng = random.split(rng)
            perms = jax.random.permutation(step_rng, train_size)
            perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            losses = []
            for perm in perms:
                batch = mf[perm, :]
                rng, step_rng = random.split(rng)
                loss, params, opt_state = update_step(params, step_rng, batch, opt_state, score_model)
                losses.append(loss)
            mean_loss = jnp.mean(jnp.array(losses))
            if k % 100 == 0:
                print("Epoch %d \t, Loss %f " % (k, mean_loss))
        return score_model, params


def train_new_nn_on_retrain(N_epochs, rng, mf, score_model, params, opt_state):
    train_size = mf.shape[0]
    batch_size = 5
    batch_size = min(train_size, batch_size)
    steps_per_epoch = train_size // batch_size
    # steps_per_epoch = 2
    for k in range(N_epochs):
        rng, step_rng = random.split(rng)
        perms = jax.random.permutation(step_rng, train_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        losses = []
        for perm in perms:
            batch = mf[perm, :]
            rng, step_rng = random.split(rng)
            loss, params, opt_state = update_step(params, step_rng, batch, opt_state, score_model)
            losses.append(loss)
        mean_loss = jnp.mean(jnp.array(losses))
        if k % 10 == 0:
            print("Epoch %d \t, Loss %f " % (k, mean_loss))
    return score_model, params, opt_state

