"""
Compare samples from the forward and reverse diffusion processes.

"""
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
    get_mf,
    update_step,
    optimizer,
    train_ts, retrain_nn, beta_t, drift, dispersion,
    reverse_sde_t, forward_sde_hyperplane_t,
    forward_sde_t)
from sgm.non_linear import NonLinear
from sgm.linear import Matrix


def main():
    rng = random.PRNGKey(123)
    rng, step_rng, step_rng2 = random.split(rng, 3)
    reduce_mean = True
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    # Plot drift and diffusion as functions of time

    plt.plot(train_ts, beta_t(train_ts))
    plt.savefig(path + "drift")
    plt.close()
    plt.plot(train_ts, dispersion(train_ts))
    plt.savefig(path + "dispersion")
    plt.close()

    # The data can be for example, pixels in an image, described by a single point in Euclidean space
    # Jstart = 2
    # Jstop = 4
    # Jnum = 6
    # Js = jnp.logspace(Jstart, Jstop, Jnum).astype(int)
    Js = [10, 100, 1000]
    Js = [1000]  # TODO tmp
    Jnum = len(Js)
    # J_test = jnp.logspace(Jstart, Jstop, Jnum).astype(int)
    J_test = [100]
    M = 1
    N = 2
    data_strings = ["hyperplane", "multimodal_hyperplane_mvn"]
    data_string = data_strings[1]
    _, test_data, *_ = get_mf(
        data_string, Js=J_test, J_true=J_test[0], M=M, N=N)
    mfs, mf_true, m_0, C_0, tangent_basis, projection_matrix = get_mf(data_string, Js=Js, J_true=Js[-1], M=M, N=N)


    plt.scatter(mfs["{:d}".format(Js[0])][:, 0], mfs["{:d}".format(Js[0])][:, 1])
    plt.savefig(path + "scatter.png")
    plt.close()

    architectures = ["non_linear", "matrix", "cholesky"]
    architecture = architectures[1]
    if architecture == "non_linear":
        score_model = NonLinear()
    elif architecture == "matrix":
        score_model = Matrix()
    elif architecture == "cholesky":
        score_model = Cholesky()
    else:
        raise ValueError()

    colors = plt.cm.jet(jnp.linspace(0,1,Jnum))

    # Get functions that return loss
    decomposition = False
    if decomposition:
        # Plot projected and orthogonal components of loss
        from sgm.utils import orthogonal_loss_fn, orthogonal_loss_fn_t
        loss_function = orthogonal_loss_fn(projection_matrix)
        loss_function_t = orthogonal_loss_fn_t(projection_matrix)
        if architecture in ["matrix", "cholesky"]:
            from sgm.linear import orthogonal_oracle_loss_fn_t
            oracle_loss_function_t = orthogonal_oracle_loss_fn_t(projection_matrix)
    else:
        from sgm.utils import loss_fn
        from sgm.utils import loss_fn_t
        from sgm.linear import oracle_loss_fn_t
        loss_function = loss_fn
        loss_function_t = loss_fn_t
        if architecture in ["matrix", "cholesky"]:
            oracle_loss_function_t = oracle_loss_fn_t

    N_epochs = 10000
    for i, train_size in enumerate(Js):
        mf = mfs["{:d}".format(train_size)]
        train_size = mf.shape[0]
        N = mf.shape[1]
        batch_size = train_size
        time = jnp.ones((batch_size, 1))
        # reset params and opt_state
        rng, step_rng = random.split(rng)
        if architecture == "non_linear":
            params = score_model.init(step_rng, mf, time)
        else:
            params  = score_model.init(step_rng, time, N)
        opt_state = optimizer.init(params)

        score_model, params, opt_state, mean_losses = retrain_nn(
            update_step,
            N_epochs, step_rng, mf, score_model, params, opt_state,
            loss_function, batch_size, decomposition=decomposition)

        trained_score = lambda x, t: score_model.evaluate(params, x, t)

        rng, step_rng = random.split(rng)
        for j, test_size in enumerate(J_test):
            q_samples = reverse_sde_t(step_rng, N, test_size, drift, dispersion, trained_score, train_ts)
            q_samples = q_samples.transpose(0, 2, 1)[1:]
            #forward_sde_hyperplane = jit(vmap(lambda t: forward_sde_hyperplane_t(t, rng, test_size, m_0, C_0), in_axes=(0), out_axes=(0)))
            rng, step_rng = random.split(rng)
            p_samples, i = forward_sde_t(test_data, rng, N, test_size, drift, dispersion, train_ts)
            p_samples = p_samples.transpose(0, 2, 1)[:-1]
            # Compute average mean squared distance between the samples
            # TODO with the projection matrix at hand, finding the distances should be easier than below
            print(jnp.shape(p_samples))
            projected_p_samples = projection_matrix @ p_samples  # projection matrix
            projected_q_samples = projection_matrix @ q_samples
            perpendicular_p_samples = p_samples - projected_p_samples
            perpendicular_q_samples = q_samples - projected_q_samples
            # Could plot histogram of one of the axis
            # Distance to the hyperplane of the samples
            distance_p_ps = jnp.linalg.norm(projected_p_samples, axis=1)
            distance_p_qs = jnp.linalg.norm(projected_q_samples, axis=1)
            distance_p_p = jnp.mean(distance_p_ps, axis=1)
            distance_p_q = jnp.mean(distance_p_qs, axis=1)
            distance_ps = jnp.linalg.norm(perpendicular_p_samples, axis=1)
            distance_qs = jnp.linalg.norm(perpendicular_q_samples, axis=1)
            distance_p = jnp.mean(distance_ps, axis=1)
            distance_q = jnp.mean(distance_qs, axis=1)
            plt.plot(train_ts[::-1][1:], distance_qs[:], color='k', alpha=0.2)
            plt.savefig(path + "testqparallel.png")
            plt.close()

            plt.plot(train_ts[::-1][1:], distance_p_q[:], label='q')
            plt.plot(train_ts[:-1], distance_p_p[:], label='p')
            plt.legend()
            plt.savefig(path + "testparallel.png")
            plt.close()
            plt.plot(train_ts[::-1][1:], distance_q[:], label='q')
            plt.plot(train_ts[:-1], distance_p[:], label='p')
            plt.legend()
            plt.savefig(path + "testperpendicular.png")
            plt.close()
            assert 0
            # TODO What is actually being asked for here
            # plt.savefig(path + "testqperp.png")
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
