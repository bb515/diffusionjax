import jax
# enable 64 precision
# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import matplotlib.pyplot as plt
import jax.random as random
import jax.numpy as jnp
import seaborn as sns
sns.set_style("darkgrid")
cm = sns.color_palette("mako_r", as_cmap=True)
import numpy as np  # for plotting
# For sampling from MVN
from sgm.utils import sample_hyperplane, sample_sphere
from sgm.non_linear import nabla_log_hat_pt, train_nn
from sgm.plot import heatmap_image, plot_score, plot_heatmap, plot_samples


def main():
    """Train neural network for given examples."""
    # The data can be for example, pixels in an image, described by a single point in Euclidean space
    J = 18  # 100 may not be enough, why cant use less than 50? something is still wrong
    M = 1
    N = 2

    # This data lies on sphere
    # Defining it as a global variable makes it accessible to JAX without doing lambda functions
    mf = sample_sphere(J, M, N)
    # mf = mf[jnp.where(mf[:, 0] < 0)]

    
    # mf = sample_hyperplane(J, M, N)
    # # This data lies on hyperplane
    # mf = sample_hyperplane(J, D, N)
    plt.scatter(mf[:, 0], mf[:, 1])
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))
    plt.savefig("scatter.pdf")
    plt.close()
    rng = random.PRNGKey(123)
    score = lambda x, t: nabla_log_hat_pt(x, t, mf)
    plot_score(score, 0.01, N, -3, 3)
    heatmap_image(score, N=N, n_samps=5000, rng=rng)
    # perturbed_score = lambda x, t: nabla_log_hat_pt(x, t) + 1
    # heatmap_image(score=perturbed_score, n_samps=5000)
    samples = train_nn(mf, N)
    plot_heatmap(samples)
    plot_samples(samples, index=[0, 1], lims=((-3, 3), (-3, 3)))
    assert 0


if __name__ == "__main__":
    main()