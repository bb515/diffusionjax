import os
path = os.path.join(os.path.expanduser('~'), 'sync', 'exp/')

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
from sgm.linear import train_linear_nn, ApproximateScoreLinear
from sgm.utils import (
    sample_hyperplane, reverse_sde, sample_hyperplane, drift, dispersion, train_ts,
    w1_dd, average_distance_to_hyperplane, w1_stdnormal)


def main():
    """Outer loop for training over sample sizes"""
    rng = random.PRNGKey(123) 
    Js = jnp.logspace(2, 9, num=25, base=2).astype(int)
    wasserstein_generated_train = []
    average_distance_to_hyperplanes = []
    wasserstein_generated_analytic = []
    wasserstein_train_analytic = []
    rng, step_rng, step_rng2 = random.split(rng, 3)
    score_model = ApproximateScoreLinear()
    for J in Js:
        print("At J=%d" % J)
        N = 3
        mf = sample_hyperplane(J, 1, N)
        score_model, params = train_linear_nn(step_rng, mf, batch_size=jnp.shape(mf)[1], score_model=score_model, N_epochs=100)
        trained_score = jit(lambda x, t: score_model.apply(params, x, t))
        samples = reverse_sde(step_rng2, N, 10000, drift, dispersion, trained_score, train_ts)
        wasserstein_generated_train.append(w1_dd(samples[:, 0], mf[:, 0]))
        average_distance_to_hyperplanes.append(average_distance_to_hyperplane(samples))
        wasserstein_generated_analytic.append(w1_stdnormal(samples[:, 0]))
        wasserstein_train_analytic.append(w1_stdnormal(mf[:, 0]))
    print("wasserstein_generated_train", wasserstein_generated_train)
    print("average_distance_to_hyperplane", average_distance_to_hyperplane)
    print("wasserstein_generated_analytic", wasserstein_generated_analytic)
    print("wasserstein_train_analytic", wasserstein_train_analytic)
    plt.plot(Js, wasserstein_generated_train, label="wasserstein_generated_train")
    plt.plot(Js, average_distance_to_hyperplanes, label="average_distance_to_hyperplane")
    plt.plot(Js, wasserstein_generated_analytic, label="wasserstein_generated_analytic")
    plt.plot(Js, wasserstein_train_analytic, label="twasserstein_train_analytic")
    plt.legend()
    plt.savefig(path + "lineplot.png")
    plt.xscale("log")
    plt.savefig(path + "lineplot_log.png")
    plt.close()

    plt.title(r"$\ell^{2}$ distance to hyperplane")
    plt.plot(Js, wasserstein_generated_train)
    plt.xscale("log")
    plt.savefig(path + "lineplot_log_sm_mf.png")
    plt.close()

    plt.title("Wasserstein distance between empirical distribution and samples")
    plt.plot(Js, average_distance_to_hyperplanes)
    plt.xscale("log")
    plt.savefig(path + "lineplot_log_sm_td.png")
    plt.close()

    plt.title("Wasserstein distance between samples and normal")
    plt.plot(Js, wasserstein_generated_analytic)
    plt.xscale("log")
    plt.savefig(path + "lineplot_log_sm_no.png")
    plt.close()

    plt.title("Wasserstein distance between empirical distribution and normal")
    plt.plot(Js, wasserstein_generated_train)
    plt.xscale("log")
    plt.savefig(path + "lineplot_log_tf_no.png")
    plt.close()


if __name__ == "__main__":
    main()
