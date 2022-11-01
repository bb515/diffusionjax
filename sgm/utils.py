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
# from scipy.stats import qmc
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
from mlkernels import Linear
import lab as B
# For configs
import ml_collections

rng = random.PRNGKey(123)

#Initialize the optimizer
optimizer = optax.adam(1e-3)


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 128
    training.n_iters = 2000
    training.continuous = True
    training.snapshot_freq = 50000
    training.log_freq = 50
    training.eval_freq = 100
    # TODO: need to review these config options

    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = True
    training.n_jitted_steps = 5
    training.reduce_mean = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 9
    evaluate.end_ckpt = 26
    evaluate.batch_size = 1024
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = 'test'

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'CIFAR10'
    data.image_size = 32
    data.random_flip = True
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 3

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.1
    model.embedding_type = 'fourier'

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42

    return config


@partial(jax.jit, static_argnames=['N'])  # TODO: keep this here?
def matrix_inverse(matrix, N):
    L_cov = cholesky(matrix, lower=True)
    L_covT_inv = solve_triangular(L_cov, jnp.eye(N), lower=True)
    cov = solve_triangular(L_cov.T, L_covT_inv, lower=False)
    return cov, L_cov


@partial(jax.jit)
def matrix_cho(matrix):
    L_cov = cholesky(matrix, lower=True)
    return L_cov


@partial(jax.jit)
def matrix_solve(L_cov, b):
    x = solve_triangular(
        L_cov, b, lower=True)
    return x


def inverse_scaler(x):
    """TODO make this better"""
    return x


def orthogonal_projection_matrix(tangent):
    return 1./ jnp.linalg.norm(tangent) * jnp.array([[tangent[0]**2, tangent[0] * tangent[1]], [tangent[0] * tangent[1], tangent[1]**2]])


def sample_mvn(J, N, kernel=Linear(), m_0=0.0):
    """
    J: How many samples to generate
    N: Dimension of the embedding space
    Returns a (J, N) array of samples
    """
    # rng = random.PRNGKey(123)
    # rng, step_rng = jax.random.split(rng)  # the missing line
    # sample from an input space
    # z = np.array(random.normal(step_rng, (N,)))
    z = np.linspace(-3, 3, N)
    C_0 = kernel(z)
    C_0 = B.dense(C_0)
    manifold = scipy.stats.multivariate_normal.rvs(mean=m_0, cov=C_0, size=J)
    # Normalization may be necessary because of the unit prior distribution, but must unnormalize later
    #manifold = manifold - jnp.mean(manifold, axis=0)
    #manifold = manifold / jnp.max(jnp.var(manifold, axis=0))
    marginal_mean = jnp.mean(jnp.mean(manifold, axis=0))
    manifold = manifold - marginal_mean
    marginal_std = jnp.max(jnp.std(manifold, axis=0))
    manifold = manifold / marginal_std
    return manifold, m_0, C_0, z, marginal_mean, marginal_std


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


def sample_hyperplane_mvn(J, N, C_0, m_0, projection_matrix):
    """
    J: How many samples to generate
    N: Dimension of the embedding space
    Returns a (J, N) array of samples
    """
    rng = random.PRNGKey(123)
    rng, step_rng = jax.random.split(rng)  # the missing line
    # sample from an input space
    manifold = scipy.stats.multivariate_normal.rvs(mean=m_0, cov=C_0, size=J)
    # Normalization is done in practice because of the unit prior distribution
    manifold = manifold - jnp.mean(manifold, axis=0)
    manifold = manifold / jnp.max(jnp.var(manifold, axis=0))
    # Project onto the manifold
    manifold = manifold @ projection_matrix.T
    return manifold


def sample_multimodal_mvn(J, N, C_0, m_0, weights):
    """
    J is approx number of samples
    """
    rng = random.PRNGKey(123)
    rng, step_rng = jax.random.split(rng)
    nmodes = jnp.shape(m_0)[0]
    manifolds = []
    for i in range(nmodes):
        print(m_0[i])
        print(C_0[i])
        print(int(J * weights[i]))
        manifolds.append(scipy.stats.multivariate_normal.rvs(mean=m_0[i], cov=C_0[i], size=int(J * weights[i])))
    manifold = jnp.concatenate(manifolds, axis=0)
    manifold = manifold - jnp.mean(manifold, axis=0)
    manifold = manifold / jnp.max(jnp.var(manifold, axis=0))
    return manifold


def sample_multimodal_hyperplane_mvn(J, N, C_0, m_0, weights, projection_matrix):
    """
    J: How many samples to generate
    N: Dimension of the embedding space
    Returns a (J, N) array of samples
    """
    rng = random.PRNGKey(123)
    rng, step_rng = jax.random.split(rng)  # the missing line
    nmodes = jnp.shape(m_0)[0]
    manifolds = []
    for i in range(nmodes):
        print(m_0[i])
        print(C_0[i])
        print(int(J * weights[i]))
        manifolds.append(scipy.stats.multivariate_normal.rvs(mean=m_0[i], cov=C_0[i], size=int(J * weights[i])))
    manifold = jnp.concatenate(manifolds, axis=0)
    print(jnp.shape(manifold))
    manifold = manifold - jnp.mean(manifold, axis=0)
    manifold = manifold / jnp.max(jnp.var(manifold, axis=0))
    # print(projection_matrix)
    manifold = manifold @ projection_matrix.T  # transpose since it is right multiplied
    return manifold


def forward_sde_hyperplane_t(t, rng, N_samps, m_0, C_0):
    # Exact transition kernel for sampling from a OU with Gaussian inital law
    rng, step_rng = jax.random.split(rng)
    z = random.normal(step_rng, (m_0.shape[0], N_samps))
    m_t = mean_factor(t)
    m_0 = m_0.reshape(-1, 1)
    v_t = var(t)
    C = m_t**2 * C_0 + v_t * jnp.eye(jnp.shape(m_0)[0])
    L_cov = matrix_cho(C)
    return m_t * m_0 + L_cov @ z


def retrain_nn_alt(update_step, N_epochs, rng, mf, score_model, params, opt_state, loss_fn, score, batch_size=5, decomposition=False):
    if decomposition:
        L = 2
    else:
        L = 1
    train_size = mf.shape[0]
    batch_size = min(train_size, batch_size)
    steps_per_epoch = train_size // batch_size
    mean_losses = jnp.zeros((N_epochs, L))
    for i in range(N_epochs):
        rng, step_rng = random.split(rng)
        perms = jax.random.permutation(step_rng, train_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        losses = jnp.zeros((jnp.shape(perms)[0], L))
        for j, perm in enumerate(perms):
            batch = mf[perm, :]
            rng, step_rng = random.split(rng)
            loss, params, opt_state = update_step(params, step_rng, batch, opt_state, score_model, loss_fn, score, has_aux=decomposition)
            if decomposition:
                loss = loss[1]
            losses = losses.at[j].set(loss)
        # TODO: is mean_loss really what I need to plot? Loses the covariance between different directions
        # For now, don't batch by setting train_size == batch_size == mf.shape[0]
        mean_loss = jnp.mean(losses, axis=0)
        mean_losses = mean_losses.at[i].set(mean_loss)
        if i % 10 == 0:
            if L==1: print("Epoch {:d}, Loss {:.2f} ".format(i, mean_loss[0]))
            if L==2: print("Tangent loss {:.2f}, perpendicular loss {:.2f}".format(mean_loss[0], mean_loss[1]))
    return score_model, params, opt_state, mean_losses


def retrain_nn(
        update_step, N_epochs, rng, mf, score_model, params,
        opt_state, loss_fn, batch_size=5, decomposition=False):
    if decomposition:
        L = 2
    else:
        L = 1
    train_size = mf.shape[0]
    batch_size = min(train_size, batch_size)
    steps_per_epoch = train_size // batch_size
    mean_losses = jnp.zeros((N_epochs, L))
    for i in range(N_epochs):
        rng, step_rng = random.split(rng)
        perms = jax.random.permutation(step_rng, train_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        losses = jnp.zeros((jnp.shape(perms)[0], L))
        for j, perm in enumerate(perms):
            batch = mf[perm, :]
            rng, step_rng = random.split(rng)
            loss, params, opt_state = update_step(params, step_rng, batch, opt_state, score_model, loss_fn, has_aux=decomposition)
            if decomposition:
                loss = loss[1]
            losses = losses.at[j].set(loss)
        # TODO: is mean_loss really what I need to plot? Loses the covariance between different directions
        # For now, don't batch by setting train_size == batch_size == mf.shape[0]
        mean_loss = jnp.mean(losses, axis=0)
        mean_losses = mean_losses.at[i].set(mean_loss)
        if i % 10 == 0:
            if L==1: print("Epoch {:d}, Loss {:.2f} ".format(i, mean_loss[0]))
            if L==2: print("Tangent loss {:.2f}, perpendicular loss {:.2f}".format(mean_loss[0], mean_loss[1]))
    return score_model, params, opt_state, mean_losses


def get_score_fn(sde, model, params, score_scaling):
    if score_scaling is True:
        # Scale score by a marginal stddev
        # model.evaluate is h = -\sigma_t s(x_t)
        return lambda x, t: -model.evaluate(params, x, t) / sde.marginal_prob(x, t)[1]
    else:
        # Not likelihood rescaling
        return lambda x, t: -model.evaluate(params, x, t)


def moving_average(a, n=100) :
    a = np.asarray(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


#@partial(jit, static_argnums=[4, 5, 6, 7])
# TODO work out workaround for it not being possible to jit dynamic indexing
# of the sample time
def reverse_update_step(params, rng, batch, opt_state, model, loss_fn, score, has_aux=False):
    # TODO: There isn't a more efficient place to factorize jax value_and_grad?
    val, grads = jax.value_and_grad(loss_fn, has_aux=has_aux)(params, model, score, rng, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, opt_state


@partial(jit, static_argnums=[4, 5, 6])
def update_step(params, rng, batch, opt_state, model, loss_fn, has_aux=False):
    """
    params: the current weights of the model
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    opt_state: the internal state of the optimizer
    model: the score function

    takes the gradient of the loss function and updates the model weights (params) using it. Returns
    the value of the loss function (for metrics), the new params and the new optimizer state
    """
    # TODO: There isn't a more efficient place to factorize jax value_and_grad?
    val, grads = jax.value_and_grad(loss_fn, has_aux=has_aux)(params, model, rng, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, opt_state


def get_mf(tangent_basis, m_0, C_0, data_string, Js, J_true, M, N, weights=None):
    """Get the manifold data."""
    # Tangent vector needs to have unit norm
    tangent_basis /= jnp.linalg.norm(tangent_basis)
    print(tangent_basis)
    projection_matrix = orthogonal_projection_matrix(tangent_basis)
    # Note so far that tangent_basis only implemented for 1D basis
    # tangent_basis is dotted with (N, n_batch) errors, so must be (N, 1)
    tangent_basis = tangent_basis.reshape(-1, 1)
    if data_string=="hyperplane_mvn":
        # For ND hyperplane example,
        check_dims(m_0, C_0)
        mf_true = sample_hyperplane_mvn(J_true, N, C_0, m_0, projection_matrix)
    elif data_string=="multimodal_hyperplane_mvn":
        # For 1D multimodal hyperplane example,
        # e.g.,
        # m_0 = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        # C_0 = jnp.array(
            # [
                # [[0.05, 0.0], [0.0, 0.1]],
                # [[0.05, 0.0], [0.0, 0.1]]
            # ]
        # )
        # weights = jnp.array([0.5, 0.5])
        N = 100
        mf_true = sample_multimodal_hyperplane_mvn(J_true, N, C_0, m_0, weights, projection_matrix)
    elif data_string=="multimodal_mvn":
        mf_true = sample_multimodal_mvn(J, N, C_0, m_0, weights)
    elif data_string=="sample_sphere":
        mf_true = sample_sphere(J_true, M, N)
    else:
        raise NotImplementedError()
    mfs = {}
    for J in Js:
        mfs["{:d}".format(J)] = mf_true[:J, :]
    return mfs, mf_true, projection_matrix


def check_dims(m_0, C_0):
    J = jnp.size(m_0)
    if (jnp.shape(m_0) != (J,) or jnp.shape(C_0) != (J, J)):
        raise ValueError("Unexpected shape for (m_0, C_0), expected ({}, {}), got ({} , {})".format((J,), (J, J), jnp.shape(m_0), jnp.shape(C_0)))
