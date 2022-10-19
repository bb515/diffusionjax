"""
Attempt to train via an oracle loss, to see if the loss calculation works.

Possible that some values do not need the expectation.
Plotting score matching loss as function of time for a learned hessian S_\theta(t).
"""
import os
path = os.path.join(os.path.expanduser('~'), 'sync', 'exp/')

import jax
from jax import jit, vmap, grad
import jax.numpy as jnp
# enable 64 precision
# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import matplotlib.pyplot as plt
import jax.random as random
import seaborn as sns
import numpy as np
sns.set_style("darkgrid")
cm = sns.color_palette("mako_r", as_cmap=True)
from sgm.plot import plot_score_ax, plot_score_diff
from sgm.utils import (
    optimizer, sample_hyperplane,
    sample_multimodal_hyperplane_mvn,
    sample_multimodal_mvn,
    sample_hyperplane_mvn,
    sample_sphere,
    orthogonal_projection_matrix,
    train_ts, retrain_nn, update_step)
from sgm.non_linear import NonLinear
from sgm.linear import Matrix


def moving_average(a, n=100) :
    a = np.asarray(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_mf(data_string, J, J_true, M, N):
    """Get the manifold data."""
    # TODO: try a 2-D or M-D basis 
    tangent_basis = 3.0 * jnp.array([1./jnp.sqrt(2), 1./jnp.sqrt(2)])
    # Tangent vector needs to have unit norm
    tangent_basis /= jnp.linalg.norm(tangent_basis)
    # tangent_basis = jnp.array([1.0, 0.1])
    projection_matrix = orthogonal_projection_matrix(tangent_basis)
    # Note so far that tangent_basis only implemented for 1D basis
    # tangent_basis is dotted with (N, n_batch) errors, so must be (N, 1)
    tangent_basis = tangent_basis.reshape(-1, 1)
    print(tangent_basis)
    print(projection_matrix)
    if data_string=="hyperplane":
        # For 1D hyperplane example,
        C_0 = jnp.array([[1, 0], [0, 0]])
        m_0 = jnp.zeros(N)
        mf_true = sample_hyperplane(J_true, M, N)
    elif data_string=="hyperplane_mvn":
        mf_true = sample_hyperplane_mvn(J_true, N, C_0, m_0, projection_matrix)
        C_0 = jnp.array([[1, 0], [0, 0]])
        m_0 = jnp.zeros(N)
    elif data_string=="multimodal_hyperplane_mvn":
        # For 1D multimodal hyperplane example,
        m_0 = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        C_0 = jnp.array(
            [
                [[0.05, 0.0], [0.0, 0.1]],
                [[0.05, 0.0], [0.0, 0.1]]
            ]
        )
        weights = jnp.array([0.5, 0.5])
        N = 100
        mf_true = sample_multimodal_hyperplane_mvn(J_true, N, C_0, m_0, weights, projection_matrix)
    elif data_string=="multimodal_mvn":
        mf_true = sample_multimodal_mvn(J, N, C_0, m_0, weights)
    elif data_string=="sample_sphere":
        m_0 = None
        C_0 = None
        mf_true = sample_sphere(J_true, M, N)
    else:
        raise NotImplementedError()
    mf = mf_true[:J, ]
    return mf, mf_true, m_0, C_0, tangent_basis, projection_matrix


def main():
    rng = random.PRNGKey(123)
    rng, step_rng, step_rng2 = random.split(rng, 3)
    J = 10
    J_true = 3000
    M = 1
    N = 2
    data_string = "multimodal_hyperplane_mvn"
    # data_string = "hyperplane"
    mf, mf_true, m_0, C_0, tangent_basis, projection_matrix = get_mf(data_string, J=J, J_true=J_true, M=M, N=N)

    plt.scatter(mf[:, 0], mf[:, 1])
    plt.savefig(path + "scatter.png")
    plt.close()

    plt.scatter(mf_true[:, 0], mf_true[:, 1], alpha=0.01)
    plt.savefig(path + "mf_true.png")
    plt.close()

    train_size = mf.shape[0]
    N = mf.shape[1]
    batch_size = train_size
    x = jnp.zeros(N*batch_size).reshape((batch_size, N))
    time = jnp.ones((batch_size, 1))
    # time
    rng = random.PRNGKey(123)
    rng, step_rng = random.split(rng)
    # model = "non_linear"
    model = "matrix"
    if model == "non_linear":
        score_model = NonLinear()
        params = score_model.init(step_rng, mf, time)
    elif model == "matrix":
        score_model = Matrix()
        params = score_model.init(step_rng, time, N)
    elif model == "cholesky":
        score_model = Cholesky()
        params = score_model.init(step_rng, time, N)
    else:
        raise ValueError()
    opt_state = optimizer.init(params)

    # Get functions that return loss
    decomposition = True
    if decomposition:
        # Plot projected and orthogonal components of loss
        from sgm.utils import orthogonal_loss_fn, orthogonal_loss_fn_t
        loss_function = orthogonal_loss_fn(projection_matrix)
        loss_function_t = orthogonal_loss_fn_t(projection_matrix)
        if model in ["matrix", "cholesky"]:
            from sgm.linear import orthogonal_oracle_loss_fn_t
            oracle_loss_function_t = orthogonal_oracle_loss_fn_t(projection_matrix)
    else:
        from sgm.utils import loss_fn
        from sgm.utils import loss_fn_t
        from sgm.linear import oracle_loss_fn_t
        loss_function = loss_fn
        loss_function_t = loss_fn_t
        if model in ["matrix", "cholesky"]:
            oracle_loss_function_t = oracle_loss_fn_t

    N_epochs = 2000

    score_model, params, opt_state, mean_losses = retrain_nn(
        update_step,
        N_epochs, step_rng, mf, score_model, params, opt_state,
        loss_function, batch_size, decomposition=decomposition)

    if decomposition:
        fig, ax = plt.subplots(1)
        ax.set_title("Orthogonal decomposition of losses")
        ax.plot(mean_losses[:, 0], label="tangent")
        ax.plot(mean_losses[:, 1], label="perpendicular")
        ax.set_ylabel("Loss component")
        ax.set_xlabel("Number of epochs")
        plt.legend()
        plt.savefig(path + "losses.png")
        plt.close()
 
        eval = lambda t: loss_function_t(t, params, score_model, rng, mf)
        eval_steps = vmap(eval, in_axes=(0), out_axes=(0))
        fx1 = eval_steps(train_ts[:])
        eval_true = lambda t: loss_function_t(t, params, score_model, rng, mf_true)
        eval_steps_true = vmap(eval_true, in_axes=(0), out_axes=(0))
        fx2 = eval_steps_true(train_ts[:])

        fig, ax = plt.subplots(1)
        #ax.set_title("Orthogonal decomposition of losses")
        ax.plot(train_ts, fx2[1][:, 0], 'r', label="tangent")
        ax.plot(train_ts, fx2[1][:, 1], 'b', label="perpendicular")
        #ax.set_ylabel("Loss component")
        #ax.set_xlabel(r"$t$")
        ylim = ax.get_ylim()
        #plt.legend()
        #plt.savefig(path + "losses_2.png")
        #plt.close()

        ax.set_ylim(ylim)

        #fig, ax = plt.subplots(1)
        ax.set_title("Orthogonal decomposition of losses")
        ax.plot(train_ts, fx1[1][:, 0], 'r', alpha=0.3) # , label="tangent")
        ax.plot(train_ts, fx1[1][:, 1], 'b', alpha=0.3) # , label="perpendicular")
        ax.set_ylabel("Loss component")
        ax.set_xlabel(r"$t$")
        plt.legend()
        plt.savefig(path + "losses_1.png")
        plt.close()

        if model in ["matrix", "cholesky"]:
            eval = lambda t: oracle_loss_function_t(t, params, score_model, m_0, C_0)
            eval_steps = vmap(eval, in_axes=(0), out_axes=(0))
            fx0 = eval_steps(train_ts[:])

            fig, ax = plt.subplots(1)
            ax.set_title("Orthogonal decomposition of losses")
            ax.plot(train_ts, fx0[1][:, 0], label="tangent")
            ax.plot(train_ts, fx0[1][:, 1], label="perpendicular")
            ax.set_ylabel("Loss component")
            ax.set_xlabel(r"$t$")
            ax.set_ylim(ylim)
            plt.legend()
            plt.savefig(path + "losses_0.png")
            plt.close()

        d =  jnp.abs(-fx1[1][0, 0] + fx2[1][0, 0])
        fig, ax = plt.subplots(1)
        ax.set_title("Orthogonal decomposition of difference in loss, {:d} vs {:d} samples".format(J_true, J))
        ax.plot(train_ts, jnp.abs(-fx1[1][:, 0] + fx2[1][:, 0]), label=r"tangent $|L(\theta) - \hat{L}(\theta)|$")
        ax.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
        ax.set_ylabel("Loss component")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend()
        plt.savefig(path + "losses_d12parallel.png")
        plt.close()

        d =  jnp.abs(-fx1[1][0, 1] + fx2[1][0, 1])
        fig, ax = plt.subplots(1)
        ax.set_title("Orthogonal decomposition of difference in loss, {:d} vs {:d} samples".format(J_true, J))
        ax.plot(train_ts, jnp.abs(-fx1[1][:, 1] + fx2[1][:, 1]), label=r"perpendicular $|L(\theta) - \hat{L}(\theta)|$")
        ax.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
        ax.set_ylabel("Loss component")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend()
        plt.savefig(path + "losses_d12perpendicular.png")


        if model in ["matrix", "cholesky"]:
            d =  jnp.abs(-fx0[1][0, 1] + fx1[1][0, 1])
            fig, ax = plt.subplots(1)
            ax.plot(train_ts, jnp.abs(-fx0[1][:, 1] + fx1[1][:, 1]), label=r"perpendicular $|L(\theta) - \hat{L}(\theta)|$")
            ax.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
            ax.set_ylabel("Loss component")
            ax.set_xlabel(r"$t$")
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            plt.legend()
            plt.savefig(path + "losses_d01perpendicular.png")
            plt.close()

            d =  jnp.abs(-fx0[1][0, 0] + fx1[1][0, 0])
            fig, ax = plt.subplots(1)
            ax.plot(train_ts, jnp.abs(-fx0[1][:, 0] + fx1[1][:, 0]), label=r"tangent $|L(\theta) - \hat{L}(\theta)|$")
            ax.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
            ax.set_ylabel("Loss component")
            ax.set_xlabel(r"$t$")
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            plt.legend()
            plt.savefig(path + "losses_d01parallel.png")
            plt.close()

            d =  jnp.abs(-fx0[1][0, 1] + fx2[1][0, 1])
            fig, ax = plt.subplots(1)
            ax.plot(train_ts, jnp.abs(-fx0[1][:, 1] + fx2[1][:, 1]), label=r"perpendicular $|L(\theta) - \hat{L}(\theta)|$")
            ax.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
            ax.set_ylabel("Loss component")
            ax.set_xlabel(r"$t$")
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            plt.legend()
            plt.savefig(path + "losses_d02perpendicular.png")
            plt.close()

            d =  jnp.abs(-fx0[1][0, 0] + fx2[1][0, 0])
            fig, ax = plt.subplots(1)
            ax.plot(train_ts, jnp.abs(-fx0[1][:, 0] + fx2[1][:, 0]), label=r"tangent $|L(\theta) - \hat{L}(\theta)|$")
            ax.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
            ax.set_ylabel("Loss component")
            ax.set_xlabel(r"$t$")
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            plt.legend()
            plt.savefig(path + "losses_d02parallel.png")
            plt.close()

    else:
        fig, ax = plt.subplots(1)
        ax.set_title("Loss")
        ax.plot(mean_losses[:])
        ax.plot(moving_average(mean_losses))
        ax.set_ylabel("Loss")
        ax.set_xlabel("Number of epochs")
        plt.savefig(path + "losses.png")
        plt.close()

        eval = lambda t: loss_function_t(t, params, score_model, rng, mf)
        eval_steps = vmap(eval, in_axes=(0), out_axes=(0))
        fx1 = eval_steps(train_ts[:])
        fig, ax = plt.subplots(1)
        ax.set_title("Loss")
        ax.plot(train_ts[:], fx1[:], label=r"$\hat{L}(\theta)$")
        ylim = ax.get_ylim()
        ax.set_ylabel("Loss")
        ax.set_xlabel(r"$t$")
        plt.savefig(path + "losses_1.png")
        plt.close()

        eval_approx_true = lambda t: loss_function_t(t, params, score_model, rng, mf_true)
        eval_steps_approx_true = vmap(eval_approx_true, in_axes=(0), out_axes=(0))
        fx2 = eval_steps_approx_true(train_ts[:])

        fig, ax = plt.subplots(1)
        ax.plot(train_ts[:], fx2[:], label=r"$\tilde L(\theta)$")
        ax.set_ylabel("Loss")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend()
        plt.savefig(path + "losses_2.png")
        plt.close()

        if model in ["matrix", "cholesky"]:
            eval_oracle = lambda t: oracle_loss_function_t(t, params, score_model, m_0, C_0)
            eval_steps_oracle = vmap(eval_oracle, in_axes=(0), out_axes=(0))
            fx0 = eval_steps_oracle(train_ts[:])

            fig, ax = plt.subplots(1)
            ax.plot(train_ts[:], fx0[:], label=r"$L(\theta)$")
            ax.set_ylabel("Loss")
            ax.set_xlabel(r"$t$")
            ax.set_ylim((0.0, ylim[1]))
            plt.legend()
            plt.savefig(path + "losses_0.png")
            plt.close()


            fig, ax = plt.subplots(1)
            ax.set_title("Difference in loss, {:d} samples vs {:d} samples".format(J_true, J))
            ax.plot(train_ts[:], (-fx0[:] + fx1[:]), label=r"$|L(\theta) - \hat{L}(\theta)|$")
            ax.set_ylabel("Loss component")
            ax.set_xlabel(r"$t$")
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            plt.legend()
            plt.savefig(path + "losses_d01.png")
            plt.close()

            # ax.plot(train_ts[:], d * jnp.exp(-2 * train_ts[:]), label=r"${:.2f}\exp (-2t)$".format(d))

            fig, ax = plt.subplots(1)
            ax.plot(train_ts[:],  (-fx0 + fx2), label=r"$|L(\theta) - \hat{L}(\theta)|$")
            ax.set_ylabel("Loss component")
            ax.set_xlabel(r"$t$")
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            plt.legend()
            plt.savefig(path + "losses_d02.png")
            plt.close()


if __name__ == "__main__":
    main()
