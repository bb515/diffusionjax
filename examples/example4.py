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
sns.set_style("darkgrid")
cm = sns.color_palette("mako_r", as_cmap=True)
from sgm.plot import plot_score_ax, plot_score_diff, plot_video
from sgm.utils import optimizer, sample_hyperplane, retrain_nn
from sgm.linear import ApproximateScoreLinear
from sgm.non_linear import log_hat_pt, nabla_log_pt_scalar_hyperplane, update_step, loss_fn



def main():
    # The data can be for example, pixels in an image, described by a single point in Euclidean space
    J = 50
    M = 1
    N = 2

    # mf = sample_sphere(J, M, N)
    mf = sample_hyperplane(J, M, N)
    plt.scatter(mf[:, 0], mf[:, 1])
    plt.savefig(path + "scatter.png")
    plt.close()

    (train_size, N) = mf.shape

    log_hat_pt_l = lambda x, t: log_hat_pt(x, t, mf)
    # Get a jax grad function, which can be batched with vmap
    nabla_log_hat_pt = jit(vmap(grad(log_hat_pt_l), in_axes=(0, 0), out_axes=(0)))

    # Get a jax function, which can be batched with vmap
    nabla_log_pt = jit(vmap(nabla_log_pt_scalar_hyperplane, in_axes=(0, 0), out_axes=(0)))

    def plot_frame(
            ax, score_diff, nabla_log_pt,
            nabla_log_hat_pt, trained_score):
        # xlim_max = 
        # ylim_min_1 = 
        # ylim_max_3 = 
        # ylim_min_3 = 
        # ylim_max_4 = 
        # ylim_min_4 =

        ((ax1, ax2), (ax3, ax4)) = ax

        plot_score_ax(
            ax1, nabla_log_hat_pt, 0.06, N, -3, 3, fname="plot_scorediff")
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
            ax2, trained_score, 0.06, N, -3, 3, fname="plot_scorediff")
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
            ax3, trained_score, nabla_log_hat_pt, 0.06, N, -3, 3, fname="plot_scorediff")
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
            ax4, nabla_log_pt, 0.06, N, -3, 3, fname="plot_scorediff")
        ax4.set_xlabel("nabla_log_pt")
        # ax4.set_xlabel(r"$x_0$")
        # ax4.set_ylabel(r"$x_1$")
        ax4.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        # plt.savefig(path + "plot_scorediff{}".format(i))
        return ((ax1, ax2), (ax3, ax4))

    # lim_tuple = (xlim_max, ylim_min_1, ylim_max_3, ylim_min_3, ylim_max_4, ylim_min_4)
    _J = 2
    _K = 2
    fig, ax = plt.subplots(_J, _K, figsize=(10,5))

    def animate(i, ax):
        batch_size = 5
        batch_size = min(train_size, batch_size)
        x = jnp.zeros(N*batch_size).reshape((batch_size, N))
        time = jnp.ones((batch_size, 1))
        rng = random.PRNGKey(123)
        rng, step_rng = random.split(rng)
        score_model = ApproximateScoreLinear()
        params = score_model.init(step_rng, x, time)
        opt_state = optimizer.init(params)
        print('frame {} rendered'.format(i))
        # Clear the axis
        for j in range(_J):
            for k in range(_K):
                ax[j, k].clear()
        # True score
        score_model, params, opt_state, mean_losses = retrain_nn(update_step, i, step_rng, mf, score_model, params, opt_state, loss_fn)
        trained_score = jit(lambda x, t: score_model.apply(params, x, t))
        score_diff = lambda x, t: (score_model.apply(params, x, t) - nabla_log_hat_pt(x, t))
        score_diff_normalized = lambda x, t: (score_model.apply(params, x, t) - nabla_log_hat_pt(x, t)) / jnp.linalg.norm(nabla_log_hat_pt(x, t))

        ax = plot_frame(ax, score_diff, nabla_log_pt, nabla_log_hat_pt, trained_score) 
        
        # plt.tight_layout()
        # plt.show()
        return None

    plot_video(fig, ax, animate, 300, "plot_scorediff_t=0.06_nonlinear")


if __name__ == "__main__":
    main()
