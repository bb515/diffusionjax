"""Compare samples from the forward and reverse diffusion processes."""
import os
path = os.path.join(os.path.expanduser('~'), 'sync', 'exp/')

num_threads = "6"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
os.environ["NUMBA_NUM_THREADS"] = num_threads
os.environ["--xla_cpu_multi_thread_eigen"] = "false"
os.environ["inta_op_parallelism_threads"] = num_threads
# XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" python my_file.py

# assert 0
# import tensorflow as tf
# tf.config.threading.set_intra_op_parallelism_threads(int(num_threads))
# tf.config.threading.set_inter_op_parallelism_threads(int(num_threads))


import jax
from jax import jit, vmap, grad
import jax.numpy as jnp
# enable 64 precision
# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')
import matplotlib.pyplot as plt
import jax.random as random
import seaborn as sns
sns.set_style("darkgrid")
cm = sns.color_palette("mako_r", as_cmap=True)
from sgm.plot import plot_score_ax, plot_score_diff
from sgm.utils import (
    optimizer, sample_hyperplane,
    train_ts, retrain_nn, drift, dispersion,
    reverse_sde_t, forward_sde_hyperplane_t,
    forward_sde_t)
from sgm.non_linear import (
    update_step, ApproximateScore,
    loss_fn, loss_fn_t, orthogonal_loss_fn, orthogonal_loss_fn_t)


def main():
    reduce_mean = True
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    # # Plot drift and diffusion as functions of time
    # plt.plot(train_ts, drift(train_ts))
    # plt.savefig(path + "drift")
    # plt.close()

    # Plot drift and diffusion as functions of time
    # plt.plot(train_ts, dispersion(train_ts))
    # plt.savefig(path + "dispersion")
    # plt.close()

    # tangent_basis = jnp.zeros((N, N - M))
    # tangent_basis = tangent_basis.at[jnp.array([[0, 0]])].set(jnp.sqrt(2)/2)
    # tangent_basis = tangent_basis.at[jnp.array([[1, 0]])].set(jnp.sqrt(2)/2)

    C_0 = jnp.array([[1, 0], [0, 0]])
    m_0 = jnp.zeros(2)

    rng = random.PRNGKey(123)
    rng, step_rng, step_rng2 = random.split(rng, 3)
    # The data can be for example, pixels in an image, described by a single point in Euclidean space
    Jstart = 2  # 1
    Jstop = 4  # 4
    Jnum = 6  # 10
    J_train = jnp.logspace(Jstart, Jstop, Jnum).astype(int)
    # J_test = jnp.logspace(Jstart, Jstop, Jnum).astype(int)
    J_test = [10000]
    M = 1
    N = 2

    # p_0 samples from the forward diffusion
    mf_data = {}

    for i, J in enumerate(J_train):
        mf_data["{:d}".format(J)] = sample_hyperplane(J, M, N)
        # mf_data["{:d}".format(J)] sample_hyperplane_mvn(J, N, C_0, m_0, tangent_basis)
        # mf_data["{:d}".format(J)] sample_multimodal_mvn(J, N, C_0, m_0, weights)
        # mf_data["{:d}".format(J)] sample_multimodal_hyperplane_mvn(J, N, C_0, m_0, weights, tangent_basis)
        # mf_data["{:d}".format(J)] sample_sphere(J, M, N)
        # mf_data["{:d}".format(J)] sample_hyperplane(J, M, N)
        # mf_true = sample_hyperplane(J_true, M, N)

    plt.scatter(mf_data["{:d}".format(J_train[0])][:, 0], mf_data["{:d}".format(J_train[0])][:, 1])
    plt.savefig(path + "scatter.png")
    plt.close()

    tang_basis = jnp.zeros((N, N - M))
    tang_basis = tang_basis.at[jnp.array([[0, 0]])].set(1.0)

    perp_basis = jnp.zeros((N, N - M))
    perp_basis = perp_basis.at[jnp.array([[1, 0]])].set(1.0)

    colors = plt.cm.jet(jnp.linspace(0,1,jnp.size(J_train)))

    # Choose whether to make plots of loss function split into orthgonal components
    decomposition = False
    if decomposition:
        loss_function = orthogonal_loss_fn(tang_basis)
        loss_function_t = orthogonal_loss_fn_t(tang_basis)
    else:
        loss_function = loss_fn
        loss_function_t = loss_fn_t

    for i, train_size in enumerate(J_train):
        mf = mf_data["{:d}".format(train_size)]
        train_size = mf.shape[0]
        N = mf.shape[1]
        batch_size = train_size
        x = jnp.zeros(N*batch_size).reshape((batch_size, N))
        time = jnp.ones((batch_size, 1))
        rng = random.PRNGKey(123)
        rng, step_rng = random.split(rng)
        score_model = ApproximateScore()
        params = score_model.init(step_rng, x, time)
        opt_state = optimizer.init(params)

        N_epochs = 2000

        score_model, params, opt_state, mean_losses = retrain_nn(
            update_step,
            N_epochs, step_rng, mf, score_model, params, opt_state,
            loss_function, batch_size, decomposition=decomposition)

        trained_score = lambda x, t: score_model.apply(params, x, t)
        rng, step_rng = random.split(rng)
        for j, test_size in enumerate(J_test):
            q_samples = reverse_sde_t(step_rng, N, test_size, drift, dispersion, trained_score, train_ts)
            q_samples = q_samples.transpose(0, 2, 1)[1:]
            #forward_sde_hyperplane = jit(vmap(lambda t: forward_sde_hyperplane_t(t, rng, test_size, m_0, C_0), in_axes=(0), out_axes=(0)))
            rng, step_rng = random.split(rng)
            initial = sample_hyperplane(test_size, M, N)
            p_samples, i = forward_sde_t(initial, rng, N, test_size, drift, dispersion, train_ts)
            p_samples = p_samples.transpose(0, 2, 1)[:-1]
            # Compute average mean squared distance between the samples
            distance_p_samples = jnp.einsum('ijk, jk', p_samples)
            distance_q_samples = jnp.einsum('ijk, jk', q_samples)
            assert 0
            distance_p_samples = p_samples[:, 1, :]
            distance_q_samples = q_samples[:, 1, :]
            plt.plot(train_ts[:-1], distance_p_samples[:, :100])
            plt.savefig(path + "testpperp.png")
            plt.close()
            plt.plot(train_ts[::-1][:-1], distance_q_samples[:, :100])
            plt.savefig(path + "testqperp.png")
            plt.close()
            distance_p_samples = distance_p_samples**2
            distance_q_samples = distance_q_samples**2
            p_losses = reduce_op(distance_p_samples.reshape((distance_p_samples.shape[0], -1)), axis=-1)
            q_losses = reduce_op(distance_q_samples.reshape((distance_q_samples.shape[0], -1)), axis=-1)

            plt.plot(train_ts[:-1], p_losses, label='p')  # , color=colors[i])
            plt.plot(train_ts[::-1][:-1], q_losses, '--', label='q')  # , color=colors[i])
            plt.ylim(0.0, 1.1)
            plt.savefig(path + "test_perp.png")
            plt.close()

            distance_p_samples = p_samples[:, 0, :]
            distance_q_samples = q_samples[:, 0, :]
            plt.plot(train_ts[:-1], distance_p_samples[:, :100])
            plt.savefig(path + "testpparallel.png")
            plt.close()
            plt.plot(train_ts[::-1][:-1], distance_q_samples[:, :100])
            plt.savefig(path + "testqparallel.png")
            plt.close()
            distance_p_samples = distance_p_samples**2
            distance_q_samples = distance_q_samples**2
            p_losses = reduce_op(distance_p_samples.reshape((distance_p_samples.shape[0], -1)), axis=-1)
            q_losses = reduce_op(distance_q_samples.reshape((distance_q_samples.shape[0], -1)), axis=-1)
            plt.plot(train_ts[:-1], p_losses, label='p')  # , color=colors[i])
            plt.plot(train_ts[::-1][:-1], q_losses, '--', label='q')  # , color=colors[i])
            plt.ylim(0.0, 1.1)
            plt.savefig(path + "test_parallel.png")
            plt.close()


if __name__ == "__main__":
    main()
