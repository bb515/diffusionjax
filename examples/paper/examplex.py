"""
Need to calculate an expectation over E_q_t,
which requires a large number of samples from q_t.
Need to check how many samples it takes to converge.
* Actually have: 0, 1, 2.
* 0. is numerically unstable, probably due to ill conditioned matrix at small times and other reasons
* 1. and 2. are very similar, the difference is due variance due to small N
* Need to be sure that 0. is calculated correctly.

* (p or q) ...
** e. the empirical score
*** ea. score, exact in the limit of large no. samples from true measure
*** eb. score, noisy approximation with emprical measure and minibatch of samples
*** eB. score, appproximation with emprical measure and all samples
** a. the exact score (analytical)
* ... versus (neural net estimate)
** s. approximation to the score (from the neural net),
*** With and without contraints on the architecture

calculate an expectation of the different between these.
* Questions about how each score behaves at values of x
** ea is undefined at t=T
** do ea|x and eb|x point in different directions?
** ea and s should be similar forall t \in [0, T]

* Plot the following, as a function of time
** 0. ||a - s|| expect that both scores are defined, and this does not blow update
*** does not need MC estimate.
* 1. ||eb - s|| what effect should noisy evaluations of e have?
* 2. ||ea - s|| difference between empirical measure score and approximate score
** expect blow up for t=T as emprical score paths collapse, but approximate score do not
* 3. ||ea - a|| difference between empirical measure paths and oracle measure paths
** expect similar behaviour, as generalization implies this


1. is actually evaluated. 0. is oracle, so difference between 1. and 0. is interesting.
Also interesting to see if calculating 1. more precisely 

* Questions about how each loss behaves as a function of time
** expect both 2. and 3. to blow up at t=T?
** expect that 0. does not blow up at t=T
** 2. - 3. expect to be close to zero.
** 1. - 2. is that interesting? Shows one evaluation of the noise in fx after optimization. Better to produce samples to show how noisy minibatching is?


Possible that some values do not need the expectation.
Plotting score matching loss as function of time for a learned hessian S_\theta(t)."""
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
    train_ts, retrain_nn, update_step)
from sgm.linear import (
    Matrix)


def moving_average(a, n=100) :
    a = np.asarray(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def main():
    rng = random.PRNGKey(123)
    rng, step_rng, step_rng2 = random.split(rng, 3)
    # Js = jnp.logspace(2, 9, num=25, base=2).astype(int)

    # The data can be for example, pixels in an image, described by a single point in Euclidean space
    J = 50
    J_true = 3000
    M = 1
    N = 2

    # tangent_basis = jnp.zeros((N, N - M))
    # tangent_basis = tangent_basis.at[jnp.array([[0, 0]])].set(jnp.sqrt(2)/2)
    # tangent_basis = tangent_basis.at[jnp.array([[1, 0]])].set(jnp.sqrt(2)/2)

    # For 1D hyperplane example,
    C_0 = jnp.array([[1, 0], [0, 0]])
    m_0 = jnp.zeros(N)

    # mf = sample_hyperplane_mvn(J, N, C_0, m_0, tangent_basis)
    # mf = sample_multimodal_mvn(J, N, C_0, m_0, weights)
    # mf = sample_multimodal_hyperplane_mvn(J, N, C_0, m_0, weights, tangent_basis)
    # mf = sample_sphere(J, M, N)
    mf_true = sample_hyperplane(J_true, M, N)
    mf = sample_hyperplane(J, M, N)

    plt.scatter(mf[:, 0], mf[:, 1])
    plt.savefig(path + "scatter.png")
    plt.close()

    plt.scatter(mf_true[:, 0], mf_true[:, 1])
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
    score_model = Matrix()
    params = score_model.init(step_rng, time, N)
    opt_state = optimizer.init(params)
    tangent_basis = jnp.zeros((N, N - M))
    tangent_basis = tangent_basis.at[0].set(1.0)
    projection_matrix = jnp.zeros((N, N))
    projection_matrix = projection_matrix.at[0, 0].set(1.0)

    # Get functions that return loss
    decomposition = True
    if decomposition:
        # Plot projected and orthogonal components of loss
        from sgm.utils import orthogonal_loss_fn, orthogonal_loss_fn_t
        from sgm.linear import orthogonal_oracle_loss_fn_t
        loss_function = orthogonal_loss_fn(tangent_basis)
        loss_function_t = orthogonal_loss_fn_t(tangent_basis)
        oracle_loss_function_t = orthogonal_oracle_loss_fn_t(projection_matrix)
    else:
        from sgm.utils import loss_fn
        from sgm.utils import loss_fn_t
        from sgm.linear import oracle_loss_fn_t
        loss_function = loss_fn
        loss_function_t = loss_fn_t
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

        fig, ax = plt.subplots(1)
        ax.set_title("Orthogonal decomposition of losses")
        ax.plot(train_ts, fx1[1][:, 0], label="tangent")
        ax.plot(train_ts, fx1[1][:, 1], label="perpendicular")
        ax.set_ylabel("Loss component")
        ax.set_xlabel(r"$t$")
        plt.legend()
        plt.savefig(path + "losses_1.png")
        plt.close()

        eval_true = lambda t: loss_function_t(t, params, score_model, rng, mf_true)
        eval_steps_true = vmap(eval_true, in_axes=(0), out_axes=(0))
        fx2 = eval_steps_true(train_ts[:])

        fig, ax = plt.subplots(1)
        ax.set_title("Orthogonal decomposition of losses")
        ax.plot(train_ts, fx2[1][:, 0], label="tangent")
        ax.plot(train_ts, fx2[1][:, 1], label="perpendicular")
        ax.set_ylabel("Loss component")
        ax.set_xlabel(r"$t$")
        ylim = ax.get_ylim()
        plt.legend()
        plt.savefig(path + "losses_2.png")
        plt.close()

        d = 0.06
        fig, ax = plt.subplots(1)
        ax.set_title("Orthogonal decomposition of difference in loss, {:d} vs {:d} samples".format(J_true, J))
        ax.plot(train_ts, jnp.abs(-fx1[1][:, 0] + fx2[1][:, 0]), label=r"tangent $|L(\theta) - \hat{L}(\theta)|$")
        ax.plot(train_ts, jnp.abs(-fx1[1][:, 1] + fx2[1][:, 1]), label=r"perpendicular $|L(\theta) - \hat{L}(\theta)|$")
        ax.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
        ax.set_ylabel("Loss component")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend()
        plt.savefig(path + "losses_d12.png")
        plt.close()

        print(oracle_loss_function_t(0.5, params, score_model, m_0, C_0))
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

        d = 0.06
        fig, ax = plt.subplots(1)
        ax.set_title("Orthogonal decomposition of difference in loss, {:d} vs {:d} samples".format(J_true, J))
        ax.plot(train_ts, jnp.abs(-fx0[1][:, 0] + fx1[1][:, 0]), label=r"tangent $|L(\theta) - \hat{L}(\theta)|$")
        ax.plot(train_ts, jnp.abs(-fx0[1][:, 1] + fx1[1][:, 1]), label=r"perpendicular $|L(\theta) - \hat{L}(\theta)|$")
        ax.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
        ax.set_ylabel("Loss component")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend()
        plt.savefig(path + "losses_d01.png")
        plt.close()

        d = 0.06
        fig, ax = plt.subplots(1)
        ax.set_title("Orthogonal decomposition of difference in loss, {:d} vs {:d} samples".format(J_true, J))
        ax.plot(train_ts, jnp.abs(-fx0[1][:, 0] + fx2[1][:, 0]), label=r"tangent $|L(\theta) - \hat{L}(\theta)|$")
        ax.plot(train_ts, jnp.abs(-fx0[1][:, 1] + fx2[1][:, 1]), label=r"perpendicular $|L(\theta) - \hat{L}(\theta)|$")
        ax.plot(train_ts, d * jnp.exp(-2 * train_ts), label=r"${:.2f}\exp (-2t)$".format(d))
        ax.set_ylabel("Loss component")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend()
        plt.savefig(path + "losses_d02.png")
        plt.close()

    else:
        fig, ax = plt.subplots(1)
        ax.set_title("Loss")
        ax.plot(mean_losses[:])
        ax.plot(moving_average(mean_losses))
        ax.set_ylabel("Loss")
        ax.set_xlabel("Number of epochs")
        plt.savefig(path + "losses0.png")
        plt.close()
        # eval = lambda t: loss_function_t(params, score_model, t, m_0, C_0)
        eval = lambda t: loss_function_t(t, params, score_model, rng, mf)

        eval_steps = vmap(eval, in_axes=(0), out_axes=(0))
        #fx, gx = eval_steps(train_ts)
        fx = eval_steps(train_ts[:])
        fig, ax = plt.subplots(1)
        ax.set_title("Loss")
        ax.plot(train_ts[:], fx[:], label=r"$\hat{L}(\theta)$")
        ax.set_ylabel("Loss")
        ax.set_xlabel(r"$t$")
        plt.savefig(path + "losses_t0hat.png")
        plt.close()

        eval_approx_true = lambda t: loss_function_t(t, params, score_model, rng, mf_true)
        eval_steps_approx_true = vmap(eval_approx_true, in_axes=(0), out_axes=(0))
        #fx0true, fx1true = eval_steps_true(train_ts)
        fxapproxtrue = eval_steps_approx_true(train_ts[:])

        eval_oracle = lambda t: oracle_loss_function_t(params, score_model, t, m_0, C_0)
        eval_steps_oracle = vmap(eval_oracle, in_axes=(0), out_axes=(0))
        #fx0oracle, fx1oracle = eval_steps_oracle(train_ts)
        fxoracle = eval_steps_oracle(train_ts[:])

        print(fxapproxtrue)
        print(fxtrue)
        print(fx)

        fig, ax = plt.subplots(1)
        ax.plot(train_ts[100:], fxoracle[100:], label=r"$L(\theta)$")
        ax.set_ylabel("Loss")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend()
        plt.savefig(path + "losses_t0oracle.png")
        plt.close()

        fig, ax = plt.subplots(1)
        ax.plot(train_ts[:], fxapproxtrue, label=r"$\tilde L(\theta)$")
        ax.set_ylabel("Loss")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend()
        plt.savefig(path + "losses_t0approx.png")
        plt.close()

        fig, ax = plt.subplots(1)
        ax.set_title("Difference in loss, {:d} samples vs {:d} samples".format(J_true, J))
        ax.plot(train_ts[100:], (-fx[100:] + fxtrue[100:]), label=r"$|L(\theta) - \hat{L}(\theta)|$")
        ax.set_ylabel("Loss component")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend()
        plt.savefig(path + "losses_t0dtrue.png")
        plt.close()

        # ax.plot(train_ts[:], d * jnp.exp(-2 * train_ts[:]), label=r"${:.2f}\exp (-2t)$".format(d))
  
        fig, ax = plt.subplots(1)
        ax.plot(train_ts[:],  (-fx + fxapproxtrue), label=r"$|L(\theta) - \hat{L}(\theta)|$")
        ax.set_ylabel("Loss component")
        ax.set_xlabel(r"$t$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.legend()
        plt.savefig(path + "losses_t0dapprox.png")
        plt.close()


if __name__ == "__main__":
    main()
