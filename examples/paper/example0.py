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
    get_mf,
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


def main():
    rng = random.PRNGKey(123)
    rng, step_rng, step_rng2 = random.split(rng, 3)
    # Jstart = 2
    # Jstop = 4
    # Jnum = 6
    # Js = jnp.logspace(Jstart, Jstop, Jnum).astype(int)
    Js = [10, 100, 1000]
    Jnum = len(Js)
    colors = plt.cm.jet(jnp.linspace(0,1,Jnum))
    J_true = 3000
    M = 1
    N = 2
    data_strings = ["hyperplane", "multimodal_hyperplane_mvn"]
    data_string = data_strings[1]
    mfs, mf_true, m_0, C_0, tangent_basis, projection_matrix = get_mf(data_string, Js=Js, J_true=J_true, M=M, N=N)

    # TODO: include a for loop over data mf with different number of samples J
    mf = mfs['{:d}'.format(Js[0])]

    plt.scatter(mf[:, 0], mf[:, 1])
    plt.savefig(path + "scatter.png")
    plt.close()

    plt.scatter(mf_true[:, 0], mf_true[:, 1], alpha=0.01)
    plt.savefig(path + "mf_true.png")
    plt.close()

    train_size = mf.shape[0]
    N = mf.shape[1]
    batch_size = train_size
    time = jnp.ones((batch_size, 1))
    rng = random.PRNGKey(123)
    rng, step_rng = random.split(rng)
    architectures = ["non_linear", "matrix", "cholesky"]
    architecture = architectures[1]
    if architecture == "non_linear":
        score_model = NonLinear()
        params = score_model.init(step_rng, mf, time)
    elif architecture == "matrix":
        score_model = Matrix()
        params = score_model.init(step_rng, time, N)
    elif architecture == "cholesky":
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

    N_epochs = 2000

    score_model, params, opt_state, mean_losses = retrain_nn(
        update_step,
        N_epochs, step_rng, mf, score_model, params, opt_state,
        loss_function, batch_size, decomposition=decomposition)

    if decomposition:
        fig0, ax0 = plt.subplots(1)
        ax0.set_title("Orthogonal decomposition of losses")
        ax0.plot(mean_losses[:, 0], label="tangent")
        ax0.plot(mean_losses[:, 1], label="perpendicular")
        ax0.set_ylabel("Loss component")
        ax0.set_xlabel("Number of epochs")
        plt.legend()
        plt.savefig(path + "losses.png")
        plt.close()
 
        eval = lambda t: loss_function_t(t, params, score_model, rng, mf)
        eval_steps = vmap(eval, in_axes=(0), out_axes=(0))
        fx1 = eval_steps(train_ts[:])
        eval_true = lambda t: loss_function_t(t, params, score_model, rng, mf_true)
        eval_steps_true = vmap(eval_true, in_axes=(0), out_axes=(0))
        fx2 = eval_steps_true(train_ts[:])

        fig1, ax1 = plt.subplots(1)
        #ax1.set_title("Orthogonal decomposition of losses")
        ax1.plot(train_ts, fx2[1][:, 0], 'r', label="tangent")
        ax1.plot(train_ts, fx2[1][:, 1], 'b', label="perpendicular")
        #ax1.set_ylabel("Loss component")
        #ax1.set_xlabel(r"$t$")
        ylim = ax1.get_ylim()
        ax1.set_ylim(ylim)
        ax1.set_title("Orthogonal decomposition of losses")
        ax1.plot(train_ts, fx1[1][:, 0], 'r', alpha=0.3) # , label="tangent")
        ax1.plot(train_ts, fx1[1][:, 1], 'b', alpha=0.3) # , label="perpendicular")
        ax1.set_ylabel("Loss component")
        ax1.set_xlabel(r"$t$")
        plt.legend()
        plt.savefig(path + "losses_12.png")
        plt.close()

        if architecture in ["matrix", "cholesky"]:
            eval = lambda t: oracle_loss_function_t(t, params, score_model, m_0, C_0)
            eval_steps = vmap(eval, in_axes=(0), out_axes=(0))
            fx0 = eval_steps(train_ts[:])

            fig2, ax2 = plt.subplots(1)
            ax2.set_title("Orthogonal decomposition of losses")
            ax2.plot(train_ts, fx0[1][:, 0], label="tangent")
            ax2.plot(train_ts, fx0[1][:, 1], label="perpendicular")
            ax2.set_ylabel("Loss component")
            ax2.set_xlabel(r"$t$")
            ax2.set_ylim(ylim)
            plt.legend()
            plt.savefig(path + "losses_0.png")
            plt.close()

        d =  jnp.abs(-fx1[1][0, 0] + fx2[1][0, 0])
        fig3, ax3 = plt.subplots(1)
        ax3.set_title("Orthogonal decomposition of difference in loss, {:d} vs {:d} samples".format(J_true, J))
        ax3.plot(train_ts, jnp.abs(-fx1[1][:, 0] + fx2[1][:, 0]), label=r"tangent $|L(\theta) - \hat{L}(\theta)|$")
        ax3.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
        ax3.set_ylabel("Loss component")
        ax3.set_xlabel(r"$t$")
        # ax3.set_xscale("log")
        # ax3.set_yscale("log")
        plt.legend()
        plt.savefig(path + "losses_d12parallel.png")
        plt.close()

        d =  jnp.abs(-fx1[1][0, 1] + fx2[1][0, 1])
        fig4, ax4 = plt.subplots(1)
        ax4.set_title("Orthogonal decomposition of difference in loss, {:d} vs {:d} samples".format(J_true, J))
        ax4.plot(train_ts, jnp.abs(-fx1[1][:, 1] + fx2[1][:, 1]), label=r"perpendicular $|L(\theta) - \hat{L}(\theta)|$")
        ax4.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
        ax4.set_ylabel("Loss component")
        ax4.set_xlabel(r"$t$")
        # ax4.set_xscale("log")
        # ax4.set_yscale("log")
        plt.legend()
        plt.savefig(path + "losses_d12perpendicular.png")


        if architecture in ["matrix", "cholesky"]:
            d =  jnp.abs(-fx0[1][0, 1] + fx1[1][0, 1])
            fig5, a5x = plt.subplots(1)
            ax5.plot(train_ts, jnp.abs(-fx0[1][:, 1] + fx1[1][:, 1]), label=r"perpendicular $|L(\theta) - \hat{L}(\theta)|$")
            ax5.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
            ax5.set_ylabel("Loss component")
            ax5.set_xlabel(r"$t$")
            # ax5.set_xscale("log")
            # ax5.set_yscale("log")
            plt.legend()
            plt.savefig(path + "losses_d01perpendicular.png")
            plt.close()

            d =  jnp.abs(-fx0[1][0, 0] + fx1[1][0, 0])
            fig6, ax6 = plt.subplots(1)
            ax6.plot(train_ts, jnp.abs(-fx0[1][:, 0] + fx1[1][:, 0]), label=r"tangent $|L(\theta) - \hat{L}(\theta)|$")
            ax6.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
            ax6.set_ylabel("Loss component")
            ax6.set_xlabel(r"$t$")
            # ax6.set_xscale("log")
            # ax6.set_yscale("log")
            plt.legend()
            plt.savefig(path + "losses_d01parallel.png")
            plt.close()

            d =  jnp.abs(-fx0[1][0, 1] + fx2[1][0, 1])
            fig7, ax7 = plt.subplots(1)
            ax7.plot(train_ts, jnp.abs(-fx0[1][:, 1] + fx2[1][:, 1]), label=r"perpendicular $|L(\theta) - \hat{L}(\theta)|$")
            ax7.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
            ax7.set_ylabel("Loss component")
            ax7.set_xlabel(r"$t$")
            # ax7.set_xscale("log")
            # ax7.set_yscale("log")
            plt.legend()
            plt.savefig(path + "losses_d02perpendicular.png")
            plt.close()

            d =  jnp.abs(-fx0[1][0, 0] + fx2[1][0, 0])
            fig8, ax8 = plt.subplots(1)
            ax8.plot(train_ts, jnp.abs(-fx0[1][:, 0] + fx2[1][:, 0]), label=r"tangent $|L(\theta) - \hat{L}(\theta)|$")
            ax8.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
            ax8.set_ylabel("Loss component")
            ax8.set_xlabel(r"$t$")
            # ax8.set_xscale("log")
            # ax8.set_yscale("log")
            plt.legend()
            plt.savefig(path + "losses_d02parallel.png")
            plt.close()

    else:
        fig9, ax9 = plt.subplots(1)
        ax9.set_title("Loss")
        ax9.plot(mean_losses[:])
        ax9.plot(moving_average(mean_losses))
        ax9.set_ylabel("Loss")
        ax9.set_xlabel("Number of epochs")
        plt.savefig(path + "losses.png")
        plt.close()

        eval = lambda t: loss_function_t(t, params, score_model, rng, mf)
        eval_steps = vmap(eval, in_axes=(0), out_axes=(0))

        eval_approx_true = lambda t: loss_function_t(t, params, score_model, rng, mf_true)
        eval_steps_approx_true = vmap(eval_approx_true, in_axes=(0), out_axes=(0))
        fx2 = eval_steps_approx_true(train_ts[:])

        fx1 = eval_steps(train_ts[:])
        fig10, ax10 = plt.subplots(1)
        ax10.set_title("Loss")
        ax10.plot(train_ts[:], fx1[:], 'k', label=r"$\hat{L}(\theta)$", alpha=0.3)
        ax10.plot(train_ts[:], fx2[:], 'k', label=r"$\tilde L(\theta)$")
        ylim = ax10.get_ylim()
        ax10.set_ylabel("Loss")
        ax10.set_xlabel(r"$t$")
        plt.legend()
        plt.savefig(path + "losses_12.png")
        plt.close()

        if architecture in ["matrix", "cholesky"]:
            eval_oracle = lambda t: oracle_loss_function_t(t, params, score_model, m_0, C_0)
            eval_steps_oracle = vmap(eval_oracle, in_axes=(0), out_axes=(0))
            fx0 = eval_steps_oracle(train_ts[:])

            fig12, ax12 = plt.subplots(1)
            ax12.plot(train_ts[:], fx0[:], label=r"$L(\theta)$")
            ax12.set_ylabel("Loss")
            ax12.set_xlabel(r"$t$")
            ax12.set_ylim((0.0, ylim[1]))
            plt.legend()
            plt.savefig(path + "losses_0.png")
            plt.close()


            fig13, ax13 = plt.subplots(1)
            ax13.set_title("Difference in loss, {:d} samples vs {:d} samples".format(J_true, J))
            ax13.plot(train_ts[:], (-fx0[:] + fx1[:]), label=r"$|L(\theta) - \hat{L}(\theta)|$")
            ax13.set_ylabel("Loss component")
            ax13.set_xlabel(r"$t$")
            # ax13.set_xscale("log")
            # ax13.set_yscale("log")
            plt.legend()
            plt.savefig(path + "losses_d01.png")
            plt.close()

            fig14, ax14 = plt.subplots(1)
            ax14.plot(train_ts[:],  (-fx0 + fx2), label=r"$|L(\theta) - \hat{L}(\theta)|$")
            ax14.set_ylabel("Loss component")
            ax14.set_xlabel(r"$t$")
            # ax14.set_xscale("log")
            # ax14.set_yscale("log")
            plt.legend()
            plt.savefig(path + "losses_d02.png")
            plt.close()


if __name__ == "__main__":
    main()
