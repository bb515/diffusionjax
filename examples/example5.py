from sre_parse import fix_flags
import jax
from jax import jit, vmap, grad, value_and_grad
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
from sgm.plot import plot_score_ax, plot_score_diff, plot_video
from sgm.utils import (
    optimizer, sample_hyperplane, retrain_nn, train_ts)
from sgm.non_linear import (
    nabla_log_pt, loss_fn, loss_fn_t,
    orthogonal_loss_fn, orthogonal_loss_fn_t,
    nabla_log_hat_pt)
from sgm.linear import ApproximateScoreLinear


def main():
    rng = random.PRNGKey(123)
    rng, step_rng, step_rng2 = random.split(rng, 3)
    # Js = jnp.logspace(2, 9, num=25, base=2).astype(int)

    # The data can be for example, pixels in an image, described by a single point in Euclidean space
    J = 50  # 100 may not be enough, why cant use less than 50? something is still wrong
    J_true = 1000
    M = 1
    N = 2

    # mf = sample_sphere(J, M, N)  # This data lies on sphere
    mf = sample_hyperplane(J, M, N)
    mf_true = sample_hyperplane(J_true, M, N)
    plt.scatter(mf[:, 0], mf[:, 1])
    plt.savefig("scatter.png")
    plt.close()

    # lim_tuple = (xlim_max, ylim_min_1, ylim_max_3, ylim_min_3, ylim_max_4, ylim_min_4)
    _J = 2
    _K = 2
    fig, ax = plt.subplots(_J, _K)

    def plot_frame(
            ax, t, nabla_log_pt,
            nabla_log_hat_pt, trained_score):
        # xlim_max = 
        # ylim_min_1 = 
        # ylim_max_3 = 
        # ylim_min_3 = 
        # ylim_max_4 = 
        # ylim_min_4 =

        ((ax1, ax2), (ax3, ax4)) = ax

        plot_score_ax(
            ax1, nabla_log_hat_pt, t, N, -3, 3, fname="plot_scorediff")
        # cumulative total cost
        #ax1.set_xlim(0, xlim_max)
        #ax1.set_ylim(ylim_min_1, 0)
        ax1.set_xlabel("nabla_log_hat_pt")
        # ax1.set_xlabel(r"$x_0$")
        # ax1.set_ylabel(r"$x_1$")
        ax1.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)

        # ax2.set_xlim(0, xlim_max)
        # ax2.set_ylim(0.4, 3.1)
        plot_score_ax(
            ax2, trained_score, t, N, -3, 3, fname="plot_scorediff")
        ax2.set_xlabel("trained_score")
        # ax2.set_xlabel(r"$x_0$")
        # ax2.set_ylabel(r"$x_1$")
        ax2.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)

        # ax3.set_xlim(0, xlim_max)
        # ax3.set_ylim(ylim_min_3, ylim_max_3)
        plot_score_diff(
            ax3, trained_score, nabla_log_hat_pt, t, N, -3, 3, fname="plot_scorediff")
        ax3.set_xlabel("trained_score - nabla_log_hat_pt")
        # ax3.set_xlabel(r"$x_0$")
        # ax3.set_ylabel(r"$x_1$")
        ax3.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)

        # ax4.set_ylim(ylim_min_4, ylim_max_4)
        # ax4.set_xlim(0, xlim_max)
        plot_score_ax(
            ax4, nabla_log_pt, t, N, -3, 3, fname="plot_scorediff")
        ax4.set_xlabel("nabla_log_pt")
        # ax4.set_xlabel(r"$x_0$")
        # ax4.set_ylabel(r"$x_1$")
        ax4.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        # plt.savefig("plot_scorediff{}".format(i))
        return ((ax1, ax2), (ax3, ax4))

    fig, ax = plt.subplots(_J, _K, figsize=(10,5))

    frames = 10

    def animate(i, ax):
 
        print('frame {} rendered'.format(i))
        # Clear the axis
        for j in range(_J):
            for k in range(_K):
                ax[j, k].clear()
        # True score
        score_diff = lambda x, t: (score_model.apply(params, x, t) - nabla_log_hat_pt(x, t))
        ax = plot_frame(ax, 1 - (i / frames), nabla_log_pt, nabla_log_hat_pt, trained_score) 
 
        # plt.tight_layout()
        # plt.show()
        return None

    decomposition = False
    train_size = mf.shape[0]
    N = mf.shape[1]
    batch_size = train_size
    x = jnp.zeros(N*batch_size).reshape((batch_size, N))
    time = jnp.ones((batch_size, 1))
    rng = random.PRNGKey(123)
    rng, step_rng = random.split(rng)
    score_model = ApproximateScoreLinear()
    params = score_model.init(step_rng, x, time)
    opt_state = optimizer.init(params)
    tangent_basis = jnp.zeros((N, N - M))
    tangent_basis = tangent_basis.at[jnp.array([[0, 0]])].set(1.0)


    if decomposition:
        loss_function = orthogonal_loss_fn(tangent_basis)
        loss_function_t = orthogonal_loss_fn_t(tangent_basis)
    else:
        loss_function = loss_fn
        loss_function_t = loss_fn_t
    score_model, params, opt_state, mean_losses = retrain_nn(
        2000, step_rng, mf, score_model, params, opt_state,
        loss_function, batch_size, decomposition=decomposition)

    if decomposition:
        # eval = lambda t: evaluate_step(t, params, rng, mf, score_model, loss_function_t, has_aux=True)
        eval = lambda t: loss_function_t(t, params, score_model, rng, mf)
 
        eval_steps = vmap(eval, in_axes=(0), out_axes=(0))

        #fx0, fx1 = eval_steps(train_ts)
        fx0 = eval_steps(train_ts)

        fig, ax = plt.subplots(1)
        ax.set_title("Orthogonal decomposition of losses")
        ax.plot(train_ts, fx0[1][:, 0], label="tangent")
        ax.plot(train_ts, fx0[1][:, 1], label="perpendicular")
        ax.set_ylabel("Loss component")
        ax.set_xlabel(r"$t$")
        plt.legend()
        plt.savefig("losses_t1.png")
        plt.close()

        # eval_true = lambda t: evaluate_step(t, params, rng, mf_true, score_model, loss_function_t, has_aux=True)
        eval_true = lambda t: loss_function_t(t, params, score_model, rng, mf_true)

        eval_steps_true = vmap(eval_true, in_axes=(0), out_axes=(0))

        #fx0true, fx1true = eval_steps_true(train_ts)
        fx0true = eval_steps_true(train_ts)
 
        d = 0.06
        fig, ax = plt.subplots(1)
        ax.set_title("Orthogonal decomposition of difference in loss, {:d} vs {:d} samples".format(J_true, J))
        ax.plot(train_ts, jnp.abs(-fx0[1][:, 0] + fx0true[1][:, 0]), label=r"tangent $|L(\theta) - \hat{L}(\theta)|$")
        ax.plot(train_ts, jnp.abs(-fx0[1][:, 1] + fx0true[1][:, 1]), label=r"perpendicular $|L(\theta) - \hat{L}(\theta)|$")
        ax.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
        ax.set_ylabel("Loss component")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend()
        plt.savefig("losses_t1d.png")
        plt.close()

    else:
        #eval = lambda t: evaluate_step(t, params, rng, mf, score_model, loss_function_t, has_aux=False)
        eval = lambda t: loss_function_t(t, params, score_model, rng, mf)

        eval_steps = vmap(eval, in_axes=(0), out_axes=(0))
        #fx, gx = eval_steps(train_ts)
        fx = eval_steps(train_ts)

        fig, ax = plt.subplots(1)
        ax.set_title("Loss")
        ax.plot(train_ts, fx[:])
        ax.set_ylabel("Loss")
        ax.set_xlabel(r"$t$")
        plt.savefig("losses_t0.png")
        plt.close()

        # eval_true = lambda t: evaluate_step(t, params, rng, mf_true, score_model, loss_function_t, has_aux=True)
        eval_true = lambda t: loss_function_t(t, params, score_model, rng, mf_true)

        eval_steps_true = vmap(eval_true, in_axes=(0), out_axes=(0))

        #fx0true, fx1true = eval_steps_true(train_ts)
        fxtrue = eval_steps_true(train_ts)

        d = 0.06
        fig, ax = plt.subplots(1)
        ax.set_title("Difference in loss, {:d} samples vs {:d} samples".format(J_true, J))
        ax.plot(train_ts, jnp.abs(-fx + fxtrue), label=r"$|L(\theta) - \hat{L}(\theta)|$")
        ax.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
        ax.set_ylabel("Loss component")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend()
        plt.savefig("losses_t0d.png")
        plt.close()

    assert 0

    if decomposition:
        fig, ax = plt.subplots(1)
        ax.set_title("Orthogonal decomposition of losses")
        ax.plot(mean_losses[:, 0], label="tangent")
        ax.plot(mean_losses[:, 1], label="perpendicular")
        ax.set_ylabel("Loss component")
        ax.set_xlabel("Number of epochs")
        plt.legend()
        plt.savefig("losses1.png")
        plt.close()
    else:
        fig, ax = plt.subplots(1)
        ax.set_title("Loss")
        ax.plot(mean_losses[:])
        ax.set_ylabel("Loss")
        ax.set_xlabel("Number of epochs")
        plt.savefig("losses0.png")
        plt.close()

    trained_score = jit(lambda x, t: score_model.apply(params, x, t))

    plot_video(fig, ax, animate, frames, "plot_scorediff")


if __name__ == "__main__":
    main()
