import jax
from jax import jit
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
from sgm.utils import reverse_sde, drift, dispersion, sample_mvn, train_ts
from sgm.linear import ApproximateScoreLinear, train_linear_nn
from mlkernels import EQ


def main():
    """Outer loop for training over sample sizes for multivariate normal with some linear covariance function"""
    rng = random.PRNGKey(123) 

    Ns = jnp.logspace(2, 4, num=3, base=2).astype(int)

    error_meanss = []
    error_covss = []
    rng, step_rng, step_rng2 = random.split(rng, 3)
    for N in Ns:
        error_means = []
        error_covs = []
        for _ in range(5):
            print("At N=%d" % N)
            J = 128
 
            mf, m_0, C_0, z = sample_mvn(J, N, kernel=EQ(), m_0=jnp.zeros((N, )))
            score_model, params = train_linear_nn(step_rng, mf, batch_size=64, score_model=ApproximateScoreLinear(), N_epochs=2000)
            trained_score = jit(lambda x, t: score_model.apply(params, x, t))
            samples = reverse_sde(step_rng2, N, 10000, drift, dispersion, trained_score, train_ts)
            plt.scatter(samples[:, 0], samples[:, 1])
            plt.title("current samples")
            plt.savefig("tmp.png")
            plt.close()
            plt.scatter(z, samples[0, :])
            plt.scatter(z, samples[1, :])
            plt.scatter(z, samples[2, :])
            plt.title("samples of x plotted over an index")
            plt.ylabel(r"$x_{i}$")
            plt.xlabel("index space")
            plt.savefig("tmp2.png")
            plt.close()
            empirical_mean = jnp.mean(samples, axis=0)
            empirical_cov = jnp.cov(samples.T)

            error_mean = jnp.linalg.norm(empirical_mean - m_0)
            error_cov = jnp.linalg.norm(C_0 - empirical_cov)

            error_means.append(error_mean)
            error_covs.append(error_cov)
            
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
