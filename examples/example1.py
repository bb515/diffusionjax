import os
path = os.path.join(os.path.expanduser('~'), 'sync', 'exp/')

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
from sgm.utils import get_mf, update_step, retrain_nn
from sgm.non_linear import NonLinear, nabla_log_hat_pt
from sgm.plot import heatmap_image, plot_score, plot_heatmap, plot_samples


def main():
    """Train neural network for given examples."""
    rng = random.PRNGKey(123)
    # The data can be for example, pixels in an image, described by a single point in Euclidean space
    J = 18  # 100 may not be enough, why cant use less than 50? something is still wrong
    M = 1
    N = 2
    data_strings = ["hyperplane", "multimodal_hyperplane_mvn", "sample_sphere"]
    data_string = data_strings[2]
    mfs, mf_true, m_0, C_0, tangent_basis, projection_matrix = get_mf(data_string, Js=[J], J_true=J, M=M, N=N)
    mf = mfs["{:d}".format(J)]

    plt.scatter(mf[:, 0], mf[:, 1])
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))
    plt.savefig(path + "scatter.png")
    plt.close()

    score = lambda x, t: nabla_log_hat_pt(x, t, mf)
    plot_score(score, 0.01, N, -3, 3, fname=path + "score.png")
    heatmap_image(score, N=N, n_samps=5000, rng=rng, fname=path + "heatmap.png")
    perturbed_score = lambda x, t: nabla_log_hat_pt(x, t) + 1
    heatmap_image(score=perturbed_score, n_samps=5000, fname=path + "perturbed_heatmap.png")
    
    train_size = mf.shape[0]
    N = mf.shape[1]
    batch_size = train_size  # usually 64 or 128
    x = jnp.zeros(N*batch_size).reshape((batch_size, N))
    time = jnp.ones((batch_size, 1))
    rng = random.PRNGKey(123)
    rng, step_rng = random.split(rng)
    score_model = ApproximateScore()
    params = score_model.init(step_rng, x, time)
    opt_state = optimizer.init(params)
    N_epochs = 2000
    score_model, params, opt_state, mean_losses = retrain_nn(
        update_step,
        N_epochs, step_rng, mf, score_model, params, opt_state,
        loss_function, batch_size, decomposition=decomposition)
    trained_score = lambda x, t: score_model.apply(params, x, t)

    samples = train_nn(mf, N)
    plot_heatmap(samples)
    plot_samples(samples, index=[0, 1], lims=((-3, 3), (-3, 3)))
    assert 0


if __name__ == "__main__":
    main()
