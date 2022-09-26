"""See the estimate of the Loss, as a function of t, converging with increased samples."""
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
sns.set_style("darkgrid")
cm = sns.color_palette("mako_r", as_cmap=True)
from sgm.plot import plot_score_ax, plot_score_diff
from sgm.utils import (
    optimizer, sample_hyperplane,
    train_ts, retrain_nn)
from sgm.non_linear import (
    update_step, ApproximateScore,
    loss_fn, loss_fn_t, orthogonal_loss_fn, orthogonal_loss_fn_t)


def main():
    rng = random.PRNGKey(123)
    rng, step_rng, step_rng2 = random.split(rng, 3)
    # Js = jnp.logspace(2, 9, num=25, base=2).astype(int)

    # The data can be for example, pixels in an image, described by a single point in Euclidean space
    J = 64
    Jstart = 2  # 1
    Jstop = 4  # 4
    Jnum = 6  # 10
    J_trues = jnp.logspace(Jstart, Jstop, Jnum).astype(int)
    M = 1
    N = 2

    mf = sample_hyperplane(J, M, N)
    mf_true = {}
    plt.scatter(mf[:, 0], mf[:, 1])
    plt.savefig("scatter.png")
    plt.close()

    for i, J_true in enumerate(J_trues):
        mf_true["{:d}".format(J_true)] = sample_hyperplane(J_true, M, N)

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
    basis = jnp.zeros((N, N - M))
    basis = basis.at[jnp.array([[0, 0]])].set(1.0)

    # Choose whether to make plots of loss function split into orthgonal components
    decomposition = False
    if decomposition:
        loss_function = orthogonal_loss_fn(basis)
        loss_function_t = orthogonal_loss_fn_t(basis)
    else:
        loss_function = loss_fn
        loss_function_t = loss_fn_t

    N_epochs = 2000

    score_model, params, opt_state, mean_losses = retrain_nn(
        update_step,
        N_epochs, step_rng, mf, score_model, params, opt_state,
        loss_function, batch_size, decomposition=decomposition)

    if decomposition:
        fig, ax = plt.subplots(1)
        ax.set_title("Loss")
        ax.plot(mean_losses[:])
        ax.set_ylabel("Loss")
        ax.set_xlabel("Number of epochs")
        plt.savefig("losses.png")
        plt.close()

        fig, ax = plt.subplots(1)
        ax.set_title("Orthogonal decomposition of losses")
        ax.plot(mean_losses[:, 0], label="\parallel")
        ax.plot(mean_losses[:, 1], label="\perpendicular")
        ax.set_ylabel("Loss component")
        ax.set_xlabel("Number of epochs")
        plt.legend()
        plt.savefig("lossescomponents.png")
        plt.close()

        eval = lambda t: loss_function_t(t, params, score_model, rng, mf)

        eval_steps = vmap(eval, in_axes=(0), out_axes=(0))
        fx = eval_steps(train_ts[:])

        fig, ax = plt.subplots(1)
        ax.set_title("Orthogonal decomposition of losses")
        ax.plot(train_ts, fx[1][:, 0], label="$\parallel \hat{L}(\theta)$")
        ax.plot(train_ts, fx[1][:, 1], label="$\perp \hat{L}(\theta)$")
        ax.set_ylabel("Loss component")
        ax.set_xlabel(r"$t$")
        ax.legend()
        plt.savefig("losses_t1hat.png")
        plt.close()

        fxapproxtrues = {}

        colors = plt.cm.jet(jnp.linspace(0,1,Jnum))
        fig2, ax2 = plt.subplots(1)
        # eval_true = lambda t: evaluate_step(t, params, rng, mf_true, score_model, loss_function_t, has_aux=True)
        for i, J_true in enumerate(J_trues):
            eval_approx_true = lambda t: loss_function_t(t, params, score_model, rng, mf_true["{:d}".format(J_trues[i])])
            eval_steps_approx_true = vmap(eval_approx_true, in_axes=(0), out_axes=(0))
            fxapproxtrues["{:d}".format(J_trues[i])] = eval_steps_approx_true(train_ts[:])

        for i, J_true in enumerate(J_trues):
            ax2.plot(train_ts[:], fxapproxtrues["{:d}".format(J_trues[i])][1][:, 0], label="N_samp={}".format(J_trues[i]), color=colors[i])
        ax2.set_title(r"$\parallel L(\theta)$")
        ax2.set_ylabel("Loss component")
        ax2.set_xlabel(r"$t$")
        # ax2.set_xscale("log")
        # ax2.set_yscale("log")
        plt.legend(fontsize='xx-small')
        plt.savefig("losses_t1approxparallelpng")
        plt.close()

        fig3, ax3 = plt.subplots(1)
        for i, J_true in enumerate(J_trues):
            ax3.plot(train_ts[:], fxapproxtrues["{:d}".format(J_trues[i])][1][:, 1], label="N_samp={}".format(J_trues[i]), color=colors[i])
        ax3.set_title(r"$\perp L(\theta)$")
        ax3.set_ylabel("Loss component")
        ax3.set_xlabel(r"$t$")
        # ax3.set_xscale("log")
        # ax3.set_yscale("log")
        plt.legend(fontsize='xx-small')
        plt.savefig("losses_t1approxperp.png")
        plt.close()

        fig0, ax0 = plt.subplots(1)
        for i, J_true in enumerate(J_trues):
            ax0.plot(train_ts[:], -fx[1][:, 0] + fxapproxtrues["{:d}".format(J_trues[i])][1][:, 0], label=r"\parallel $(L(\theta) - \hat{L}(\theta))$" + " N_samp={}".format(J_trues[i]), color=colors[i])
        ax0.set_title(r"difference $\parallel (L(\theta) - \hat{L}(\theta))$")
        ax0.set_ylabel("difference")
        ax0.set_xlabel(r"$t$")
        # ax0.set_xscale("log")
        # ax0.set_yscale("log")
        plt.legend(fontsize='xx-small')
        plt.savefig("losses_t1dapproxparallelpng")
        plt.close()

        fig1, ax1 = plt.subplots(1)
        for i, J_true in enumerate(J_trues):
            ax1.plot(train_ts[:], -fx[1][:, 1] + fxapproxtrues["{:d}".format(J_trues[i])][1][:, 1], label=r"\perp $(L(\theta) - \hat{L}(\theta))$" + " N_samp={}".format(J_trues[i]), color=colors[i])
        ax1.set_title(r"difference $\perp (L(\theta) - \hat{L}(\theta))$")
        ax1.set_ylabel("difference")
        ax1.set_xlabel(r"$t$")
        # ax1.set_xscale("log")
        # ax1.set_yscale("log")
        plt.legend(fontsize='xx-small')
        plt.savefig("losses_t1dapproxperp.png")
        plt.close()

        # ax.plot(train_ts[100:], d * jnp.exp(-2 * train_ts[100:]), label=r"${:.2f}\exp (-2t)$".format(d))
        d = 0.06
        fig, ax = plt.subplots(1)
        ax.plot(train_ts[:], -fx[1][:, 0] + fxapproxtrues["{:d}".format(J_trues[-1])][1][:, 0], label=r"$\parallel (L(\theta) - \hat{L}(\theta))$" + " N_samp={}".format(J_trues[-1]))
        ax.plot(train_ts[:], -fx[1][:, 1] + fxapproxtrues["{:d}".format(J_trues[-1])][1][:, 1], label=r"$\perp (L(\theta) - \hat{L}(\theta))$" + " N_samp={}".format(J_trues[-1]))
        ax.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
        ax.set_ylabel("difference")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend()
        plt.savefig("losses_t1dapprox.png")
        plt.close()

    else:
        fig, ax = plt.subplots(1)
        ax.set_title("Loss")
        ax.plot(mean_losses[:])
        ax.set_ylabel("Loss")
        ax.set_xlabel("Number of epochs")
        plt.savefig("losses0.png")
        plt.close()

        # eval = lambda t: loss_function_t(params, score_model, t, m_0, C_0)
        eval = lambda t: loss_function_t(t, params, score_model, rng, mf)

        eval_steps = vmap(eval, in_axes=(0), out_axes=(0))
        fx = eval_steps(train_ts[:])

        fig, ax = plt.subplots(1)
        ax.set_title("Loss")
        ax.plot(train_ts[:], fx[:], label=r"$\hat{L}(\theta)$")
        ax.set_ylabel("Loss")
        ax.set_xlabel(r"$t$")
        plt.savefig("losses_t0hat.png")
        plt.close()

        colors = plt.cm.jet(jnp.linspace(0,1,Jnum))
        fig, ax = plt.subplots(1)
        for i, J_true in enumerate(J_trues):
            eval_approx_true = lambda t: loss_function_t(t, params, score_model, rng, mf_true["{:d}".format(J_trues[i])])
            eval_steps_approx_true = vmap(eval_approx_true, in_axes=(0), out_axes=(0))
            fxapproxtrue = eval_steps_approx_true(train_ts[:])
            ax.plot(train_ts[:], fxapproxtrue, label=r"$L(\theta)$" + " N_samp={}".format(J_trues[i]), color=colors[i])
        ax.set_ylabel("Loss")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend(fontsize='xx-small')
        plt.savefig("losses_t0approx.png")
        plt.close()

        print(fxapproxtrue)
        print(fx)

        # ax.plot(train_ts[100:], d * jnp.exp(-2 * train_ts[100:]), label=r"${:.2f}\exp (-2t)$".format(d))
  
        fig, ax = plt.subplots(1)
        ax.plot(train_ts[:],  (-fx + fxapproxtrue), label=r"$(L(\theta) - \hat{L}(\theta))$")
        ax.set_ylabel("difference")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend()
        plt.savefig("losses_t0dapprox.png")
        plt.close()


if __name__ == "__main__":
    main()
