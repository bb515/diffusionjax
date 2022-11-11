"""
Compare samples from the forward and reverse diffusion processes.

"""
import os
path = os.path.join(os.path.expanduser('~'), 'sync', 'exp/')

num_threads = "8"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
os.environ["NUMBA_NUM_THREADS"] = num_threads
os.environ["--xla_cpu_multi_thread_eigen"] = "false"
os.environ["inta_op_parallelism_threads"] = num_threads
# XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" python my_file.py

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
    get_score_fn, 
    get_solver,
    get_mf,
    update_step,
    optimizer,
    retrain_nn,
    forward_sde_hyperplane_t)
from sgm.non_linear import NonLinear
from sgm.linear import Matrix
from sgm.sde import get_sde


def get_config():
  config = get_default_configs()
  # # training
  # training = config.training
  # training.sde = 'vpsde'
  # training.continuous = True
  # training.reduce_mean = True

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'none'
  ##BB
  sampling.noise_removal = False
  sampling.snr = 0.16

  ## data
  ## model
  return config


def main():
    rng = random.PRNGKey(123)
    rng, step_rng, step_rng2 = random.split(rng, 3)
    reduce_mean = True
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

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
    data_strings = ["hyperplane_mvn", "multimodal_hyperplane_mvn"]
    data_string = data_strings[0]
    # tangent_basis = 1.0 * jnp.array([0., 1.])
    tangent_basis = 3.0 * jnp.array([1./jnp.sqrt(2) , 1./jnp.sqrt(2)])
    m_0 = jnp.zeros(N)
    C_0 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    _, test_data, *_ = get_mf(tangent_basis, m_0, C_0, data_string, Js=J_test, J_true=J_test[0], M=M, N=N)
    mfs, mf_true, projection_matrix = get_mf(tangent_basis, m_0, C_0, data_string, Js=Js, J_true=Js[-1], M=M, N=N)
    mf = mfs['{:d}'.format(Js[0])]

    plt.scatter(mf_true[:, 0], mf_true[:, 1], label="mf", alpha=0.01)
    plt.scatter(mf[:, 0], mf[:, 1], label="data")
    plt.legend()
    plt.savefig(path + "mf_data.png")
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

    # Get sde model
    sde = get_sde("OU")

    # Get functions that return loss
    decomposition = False
    if decomposition:
        # Plot projected and orthogonal components of loss
        from sgm.losses import sde_projected_loss_fn
        loss_fn = get_projected_loss_fn(projection_matrix, sde, score_model, score_scaling=True, likelihood_weighting=False)
        loss_fn_t = get_projected_loss_fn(projection_matrix, sde, score_model, score_scaling=True, likelihood_weighting=False, pointwise_t=True)
        if architecture in ["matrix", "cholesky"]:
            from sgm.losses import get_oracle_loss_fn
            oracle_loss_fn = get_oracle_loss_fn(sde, score_model, m_0, C_0, score_scaling=True, likelihood_weighting=False, projection_matrix=projection_matrix)
            oracle_loss_fn_t = get_oracle_loss_fn(sde, score_model, m_0, C_0, score_scaling=True, likelihood_weighting=False, pointwise_t=True, projection_matrix=projection_matrix)
    else:
        from sgm.losses import get_loss_fn
        loss_fn = get_loss_fn(sde, score_model, score_scaling=True, likelihood_weighting=False)
        loss_fn_t = get_loss_fn(sde, score_model, score_scaling=True, likelihood_weighting=False, pointwise_t=True)
        if architecture in ["matrix", "cholesky"]:
            from sgm.losses import get_oracle_loss_fn
            oracle_loss_fn = get_oracle_loss_fn(sde, score_model, m_0, C_0, score_scaling=True, likelihood_weighting=False)
            oracle_loss_fn_t = get_oracle_loss_fn(sde, score_model, m_0, C_0, score_scaling=True, likelihood_weighting=False, pointwise_t=True)

    N_epochs = 100  # 10000
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
            loss_fn, batch_size, decomposition=decomposition)

        score_fn = get_score_fn(sde, score_model, params, score_scaling=True)
        rsde = sde.reverse(score_fn)

        f, g = rsde.discretize(jnp.array([[0.0, 0.0], [0.1, -0.1]]), jnp.array([0.5, 0.5]))
        print(f)
        print(g)

        solve = get_solver(rsde, pointwise_t=True)
        rng, step_rng = random.split(rng)
        for j, test_size in enumerate(J_test):
            q_samples = solve(step_rng, N, test_size, sde.train_ts)
            print(q_samples)
            assert 0

            q_samples = sde.reverse_sde_t(step_rng, N, test_size, trained_score, sde.train_ts)
            q_samples = q_samples.transpose(0, 2, 1)[1:]
            #forward_sde_hyperplane = jit(vmap(lambda t: forward_sde_hyperplane_t(t, rng, test_size, m_0, C_0), in_axes=(0), out_axes=(0)))
            rng, step_rng = random.split(rng)
            p_samples, i = sde.forward_sde_t(test_data, rng, N, test_size, sde.train_ts)
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
            plt.plot(sde.train_ts[::-1][1:], distance_qs[:], color='k', alpha=0.2)
            plt.savefig(path + "testqparallel.png")
            plt.close()

            plt.plot(sde.train_ts[::-1][1:], distance_p_q[:], label='q')
            plt.plot(sde.train_ts[:-1], distance_p_p[:], label='p')
            plt.legend()
            plt.savefig(path + "testparallel.png")
            plt.close()
            plt.plot(sde.train_ts[::-1][1:], distance_q[:], label='q')
            plt.plot(sde.train_ts[:-1], distance_p[:], label='p')
            plt.legend()
            plt.savefig(path + "testperpendicular.png")
            plt.close()
            assert 0
            # TODO What is actually being asked for here
            # plt.savefig(path + "testqperp.png")
            p_losses = reduce_op(distance_p_samples.reshape((distance_p_samples.shape[0], -1)), axis=-1)
            q_losses = reduce_op(distance_q_samples.reshape((distance_q_samples.shape[0], -1)), axis=-1)

            plt.plot(sde.train_ts[:-1], p_losses, label='p')  # , color=colors[i])
            plt.plot(sde.train_ts[::-1][:-1], q_losses, '--', label='q')  # , color=colors[i])
            plt.ylim(0.0, 1.1)
            plt.savefig(path + "test_perp.png")
            plt.close()

            distance_p_samples = p_samples[:, 0, :]
            distance_q_samples = q_samples[:, 0, :]
            plt.plot(sde.train_ts[:-1], distance_p_samples[:, :100])
            plt.savefig(path + "testpparallel.png")
            plt.close()
            plt.plot(sde.train_ts[::-1][:-1], distance_q_samples[:, :100])
            plt.savefig(path + "testqparallel.png")
            plt.close()
            distance_p_samples = distance_p_samples**2
            distance_q_samples = distance_q_samples**2
            p_losses = reduce_op(distance_p_samples.reshape((distance_p_samples.shape[0], -1)), axis=-1)
            q_losses = reduce_op(distance_q_samples.reshape((distance_q_samples.shape[0], -1)), axis=-1)
            plt.plot(sde.train_ts[:-1], p_losses, label='p')  # , color=colors[i])
            plt.plot(sde.train_ts[::-1][:-1], q_losses, '--', label='q')  # , color=colors[i])
            plt.ylim(0.0, 1.1)
            plt.savefig(path + "test_parallel.png")
            plt.close()


if __name__ == "__main__":
    main()
