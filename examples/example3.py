from ensurepip import bootstrap
import jax
from jax import jit
import jax.numpy as jnp
# enable 64 precision
# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update()
jax.config.update('jax_platform_name', 'cpu')
import matplotlib.pyplot as plt
import jax.random as random
import seaborn as sns
sns.set_style("darkgrid")
cm = sns.color_palette("mako_r", as_cmap=True)
from sgm.utils import reverse_sde, drift, dispersion, sample_mvn, train_ts, optimizer, retrain_nn
from sgm.non_linear import update_step, ApproximateScore, non_linear_trained_score
from sgm.non_linear import loss_fn as non_linear_loss_fn
from sgm.non_linear import orthogonal_loss_fn as non_linear_orthogonal_loss_fn
from sgm.linear import (
    ApproximateScoreLinear, ApproximateScoreOperatorLinear, linear_trained_score,
    train_linear_nn, linear_loss_fn, sqrt_linear_loss_fn, loss_fn,
    sqrt_linear_trained_score)
from mlkernels import EQ, Linear, Matern52, Matern32, Matern12


BG_ALPHA = 1.0
MG_ALPHA = 0.2
FG_ALPHA = 0.4


def main():
    """Outer loop for training over sample sizes for multivariate normal with some linear covariance function"""
    rng = random.PRNGKey(123) 
    decomposition = False
    # model = "non_linear"
    model = "linear"
    # model = "linear_map"
    # model = "sqrt_linear_map"

    if decomposition:
        assert 0  # not implemented
        if model=="non_linear":
            loss_function = non_linear_loss_fn
        elif model=="linear":
            loss_function = linear_loss_fn
        elif model=="linear_map":
            loss_function = loss_fn
        elif model=="sqrt_linear_map":
            loss_function = sqrt_linear_loss_fn
    else:
        if model=="non_linear":
            loss_function = non_linear_loss_fn
        elif model=="linear":
            loss_function = linear_loss_fn
        elif model=="linear_map":
            loss_function = loss_fn
        elif model=="sqrt_linear_map":
            loss_function = sqrt_linear_loss_fn

    # Dimension
    Ns = jnp.logspace(2, 4, num=3, base=2).astype(int)
    Ns = [128]
    n_batch = 128
    n_samps = 1000
    n_bootstrap_samps = n_samps // 2
    n_bootstraps = 50
    n_epochs = 1000
    n_repeat_experiment = 1

    # Knowing tangent basis possibly only possible with a kernel of form (x^{T}x)^{M}
    error_meanss = []
    error_covss = []
    rng, step_rng, step_rng2 = random.split(rng, 3)
    for N in Ns:
        error_means = []
        error_covs = []
        for r in range(n_repeat_experiment):
            print("At N={:d}, repeat={:d}".format(N, r))
            J = 1024
            error_meanss = []
 
            mf, m_0, C_0, z, marginal_mean, marginal_std = sample_mvn(J, N, kernel=EQ(), m_0=jnp.zeros((N, )))
            # print(marginal_mean, marginal_std)
            # Is it possible to get tangent space by using the cholesky factor... samples are projections onto the basis formed by the Chol factor

            fig = plt.figure()
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax = fig.add_subplot(111)
            ax.plot(z, mf[:128, :].T, alpha=0.6)
            # ax.plot(z, mf[1, :])
            # ax.plot(z, mf[-1, :])
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.grid()
            plt.title("Samples of x, plotted as images")
            plt.ylabel(r"$x_{i}$")
            plt.xlabel("'pixel' space")
            plt.savefig("tmp1.png")
            plt.close()

            fig = plt.figure()
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax = fig.add_subplot(111)
            ax.scatter(mf[:, 0], mf[:, 1])
            ax.scatter(mf[:, 0], mf[:, -1])
            ylim2 = ax.get_ylim()
            xlim2 = ax.get_xlim()
            plt.grid()
            plt.title("mf")
            plt.savefig("tmp0.png")
            plt.close()

            train_size = mf.shape[0]
            N = mf.shape[1]
            batch_size = train_size
            time = jnp.ones((batch_size, 1))
            rng = random.PRNGKey(123)
            rng, step_rng = random.split(rng)

            if model=="non_linear":
                score_model = ApproximateScore()
                x = jnp.zeros(N * batch_size).reshape((batch_size, N))
                params = score_model.init(step_rng, x, time)
            elif model=="linear":
                score_model = ApproximateScoreLinear()
                x = jnp.zeros(N*batch_size).reshape((batch_size, N))
                params = score_model.init(step_rng, x, time)
            elif model=="linear_map":
                score_model = ApproximateScoreOperatorLinear() 
                params = score_model.init(step_rng, time, N)
            elif model=="sqrt_linear_map":
                score_model = ApproximateScoreOperatorLinear()
                params = score_model.init(step_rng, time, N)

            opt_state = optimizer.init(params)

            score_model, params, opt_state, mean_losses = retrain_nn(
                update_step, n_epochs, rng, mf, score_model, params,
                opt_state, loss_function, batch_size=n_batch, decomposition=decomposition)

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

            if model in ["non_linear", "linear"]:
                trained_score = jit(lambda x, t: non_linear_trained_score(score_model, params, t, x))
            elif model == "linear_map":
                trained_score = jit(lambda x, t: linear_trained_score(score_model, params, t, N, x, n_samps))
            elif model == "sqrt_linear_map":
                trained_score = jit(lambda x, t: sqrt_linear_trained_score(score_model, params, t, N, x))

            samples = reverse_sde(step_rng2, N, n_samps, drift, dispersion, trained_score, train_ts)
            samples = samples * marginal_std + marginal_mean  # unnormalize
            fig = plt.figure()
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax = fig.add_subplot(111)
            plt.grid()
            ax.set_xlim(xlim2)
            ax.set_ylim(ylim2)
            ax.scatter(samples[:, 0], samples[:, 1])
            ax.scatter(samples[:, 0], samples[:, 2])
            plt.title("learned mf")
            plt.savefig("tmp2.png")
            plt.close()

            fig = plt.figure()
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax = fig.add_subplot(111)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.plot(z, samples[:128, :].T, alpha=0.6)
            plt.grid()
            plt.title("Samples of x, plotted as images")
            plt.ylabel(r"$x_{i}$")
            plt.xlabel("'pixel' space")
            plt.savefig("tmp3.png")
            plt.close()

            # Calculate the MMD with a polynomial kernel
            print(jnp.shape(samples))
            idx = random.randint(rng, (n_bootstraps, n_bootstrap_samps), 0, n_samps - 1)
            bootstrap_samples = samples[idx, :]
            print(jnp.shape(bootstrap_samples))
            empirical_mean = jnp.mean(bootstrap_samples, axis=1)
            print(jnp.shape(empirical_mean))
            error_mean = jnp.mean(empirical_mean - m_0.reshape(1, -1), axis=1)
            # error_mean = jnp.linalg.norm(empirical_mean - m_0.reshape(1, -1), axis=1)
            print(error_mean)
            print(jnp.shape(error_mean))
            fig = plt.figure()
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax = fig.add_subplot(111)
            ax.hist(error_mean)
            plt.grid()
            plt.title(r"Bootstrap $\ell_2$ norm error in sample mean")
            plt.ylabel("count")
            plt.xlabel(r"$\|E(x) - m_0\|_2$")
            plt.savefig("error_mean.png")
            plt.close()

            empirical_cov = jnp.zeros((n_bootstraps, N, N))
            error_cov = jnp.zeros((n_bootstraps))
            # bootstap_samples = bootstrap_samples - jnp.expand_dims(empirical_mean, axis=1)
            for i in range(n_bootstraps):
                empirical_cov = empirical_cov.at[i].set(jnp.cov((bootstrap_samples[i, :, :]).T))
                error = jnp.linalg.norm(empirical_cov[i] - C_0)
                print(empirical_cov[i])
                error_cov = error_cov.at[i].set(error)
            print(jnp.shape(empirical_cov))
            print("Frobenius norm of C_0 = {}".format(jnp.linalg.norm(C_0)))
            print(C_0)
            print(marginal_std)
            print(marginal_mean)

            fig = plt.figure()
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax = fig.add_subplot(111)
            ax.hist(error_cov)
            plt.grid()
            plt.title(r"Bootstrap Frobenius norm of the error in sample covariance")
            plt.ylabel("count")
            plt.xlabel(r"$\|E(x x^{T}) - C_0 \|_2$")
            plt.savefig("error_cov.png")
            plt.close()
            assert 0
            error_means.append(error_mean)
            error_covs.append(error_cov)
            assert 0 
        error_meanss.append(error_means)
        error_covss.append(error_covs)

    error_meanss = jnp.array(error_meanss)
    error_covss = jnp.array(error_covss)

    error_means_mean = jnp.mean(error_meanss, axis=1)
    error_means_std = jnp.std(error_meanss, axis=1)
    error_covs_mean = jnp.mean(error_covss, axis=1)
    error_covs_std = jnp.std(error_covss, axis=1)

    print("error_means_mean", error_means_mean)
    print("error_covs_mean", error_covs_mean)

    plt.title("error in mean")
    # plt.plot(Ns, error_means_mean, label="Frobenius norm")
    plt.errorbar(Ns, error_means_mean, yerr=error_means_std, label=r"Frobenius norm $m_{0} - m_{est}$")
    plt.legend()
    plt.savefig("error_mean.png")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("error_mean_log.png")
    plt.close()

    plt.title("error in cov")
    # plt.plot(Ns, error_covs, label="Frobenius norm")
    plt.errorbar(Ns, error_covs_mean, yerr=error_covs_std, label=r"Frobenius norm $C_{0} - C_{est}$")
    plt.legend()
    plt.savefig("error_cov.png")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("error_cov_log.png")
    plt.close()

    plt.title("error in mean")
    # plt.plot(Ns, jnp.array(error_means) / jnp.array(Ns), label="Frobenius norm")
    plt.errorbar(Ns, error_means_mean / jnp.array(Ns), yerr=error_means_std / jnp.array(Ns), label=r"Frobenius norm $C_{0} - C_{est}$")
    plt.legend()
    plt.savefig("error_mean_N.png")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("error_mean_N_log.png")
    plt.close()

    plt.title("error in cov")
    # plt.plot(Ns, jnp.array(error_covs) / jnp.array(Ns), label="Frobenius norm")
    plt.errorbar(Ns, error_covs_mean / jnp.array(Ns), yerr=error_covs_std / jnp.array(Ns), label=r"Frobenius norm $C_{0} - C_{est}$")
    plt.legend()
    plt.savefig("error_cov_N.png")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("error_cov_N_log.png")
    plt.close()


if __name__ == "__main__":
    main()
