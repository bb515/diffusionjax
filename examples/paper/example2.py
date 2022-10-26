"""File for exploring loss with reverse SDEs."""
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
from sgm.utils import (
    var,
    moving_average,
    drift, dispersion, reverse_sde, get_mf,
    optimizer, train_ts,
    reverse_update_step, update_step, retrain_nn_alt, retrain_nn)
from sgm.non_linear import NonLinear, nabla_log_hat_pt
from sgm.linear import Matrix, Cholesky
from sgm.plot import heatmap_image, plot_score, plot_heatmap, plot_samples


def main():
    """Train neural network for given examples."""
    rng = random.PRNGKey(123)
    # The data can be for example, pixels in an image, described by a single point in Euclidean space
    J = 13
    J_test = 3000
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
    plot_score(score, 1.00, N, -3, 3, fname=path + "score100.png")
    plot_score(score, 0.60, N, -3, 3, fname=path + "score60.png")
    plot_score(score, 0.01, N, -3, 3, fname=path + "score1.png")
    heatmap_image(score, N=N, n_samps=5000, rng=rng, fname=path + "heatmap.png")
    perturbed_score = lambda x, t: nabla_log_hat_pt(x, t, mf) + 1
    heatmap_image(score=perturbed_score, N=N, n_samps=5000, fname=path + "perturbed_heatmap.png")
    train_size = mf.shape[0]
    N = mf.shape[1]
    batch_size = train_size
    time = jnp.ones((batch_size, 1))
    rng = random.PRNGKey(123)
    rng, step_rng = random.split(rng)
    architectures = ["non_linear", "matrix", "cholesky"]
    architecture = architectures[0]
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
    # Get functions that return loss
    decomposition = False
    flipped_loss=False
    if decomposition:
        raise NotImplementedError()
    else:
        if flipped_loss:
            from sgm.utils import flipped_loss_fn as loss_fn
            loss_function = loss_fn
        else:
            from sgm.utils import loss_fn
            from sgm.utils import loss_fn_t
            from sgm.linear import oracle_loss_fn_t
            loss_function = loss_fn
            loss_function_t = loss_fn_t
            if architecture in ["matrix", "cholesky"]:
                oracle_loss_function_t = oracle_loss_fn_t

    opt_state = optimizer.init(params)
    N_epochs = 20000
    score_model, params, opt_state, mean_losses = retrain_nn(
        update_step,
        N_epochs, step_rng, mf, score_model, params, opt_state,
        loss_function, batch_size, decomposition=decomposition)
    likelihood_flag=0  # hack. Remember to change in utils.py
    if likelihood_flag==0:
        # Song's likelihood rescaling
        # model evaluate is h = -\sigma_t s(x_t)
        # Standard training objective without likelihood rescaling
        trained_score = lambda x, t: -score_model.evaluate(params, x, t) / jnp.sqrt(var(t))
        rescaled_score = lambda x, t: -score_model.evaluate(params, x, t)
    elif likelihood_flag==1:
        # What has worked previously for us, which learns a score
        trained_score = lambda x, t: score_model.evaluate(params, x, t)
        rescaled_score = lambda x, t: score_model.evaluate(params, x, t)
    elif likelihood_flag==2:
        # Jakiw training objective - has incorrect
        trained_score = lambda x, t: score_model.evaluate(params, x, t)
        rescaled_score = lambda x, t: score_model.evaluate(params, x, t)
    elif likelihood_flag==3:
        # Not likelihood rescaling
        # model evaluate is s(x_t) errors are then scaled by \beta_t
        trained_score = lambda x, t: score_model.evaluate(params, x, t)
        rescaled_score = lambda x, t: score_model.evaluate(params, x, t)
    plot_score(trained_score, 1.00, N, -3, 3, fname=path + "trainedscore100.png")
    plot_score(trained_score, 0.60, N, -3, 3, fname=path + "trainedscore60.png")
    plot_score(trained_score, 0.01, N, -3, 3, fname=path + "trainedscore1.png")
    q_samples = reverse_sde(step_rng, N, J_test, drift, dispersion, trained_score, train_ts)
    plot_heatmap(q_samples, fname=path+"forward_heatmap.png")
    plot_samples(q_samples, fname=path+"forward_samples_qp.png", index=[0, 1], lims=((-3, 3), (-3, 3)))
    assert 0
    from sgm.utils import flipped_loss_fn
    from sgm.utils import plot_errors
    loss_function = flipped_loss_fn
    print("errors", plot_errors(params, score_model, score, rng, N, 32, fpath=path, likelihood_flag=0))
    print(flipped_loss_fn(params, score_model, score, rng, jnp.ones((32, 2))))
    # # Reset params
    # params = score_model.init(step_rng, mf, time)
    # opt_state = optimizer.init(params)
    N_epochs = 1000
    score_model, params, opt_state, mean_losses = retrain_nn_alt(
        reverse_update_step,
        N_epochs, step_rng, mf, score_model, params, opt_state,
        loss_function, score, batch_size, decomposition=decomposition)
    plt.close()
    fig9, ax9 = plt.subplots(1)
    ax9.set_title("Loss")
    ax9.plot(mean_losses[:])
    ax9.plot(moving_average(mean_losses))
    ax9.set_ylabel("Loss")
    ax9.set_xlabel("Number of epochs")
    plt.savefig(path + "reverse_losses.png")
    plt.close()
    # Since this is a rescaled trained score, shouldn't it be
    # multiplied by \var(t)
    test_rescaling = 0  # hack. remember to change in utilts.py
    if test_rescaling:
        trained_score = lambda x, t: score_model.evaluate(params, x, t) / var(t)
    else:
        trained_score = lambda x, t: score_model.evaluate(params, x, t) / jnp.sqrt(t)
    print("errors", plot_errors(params, score_model, score, rng, N, 32, fpath=path + "reverse"), likelihood_flag=0)
    plot_score(trained_score, 1.00, N, -3, 3, fname=path + "reversetrainedscore100.png")
    plot_score(trained_score, 0.60, N, -3, 3, fname=path + "reversetrainedscore60.png")
    plot_score(trained_score, 0.01, N, -3, 3, fname=path + "reversetrainedscore1.png")
    # DONE Make sure that time is defined in the correct direction
    # * Figure out what is happening to the score function when reverse training
    # ** depends on the likelihood rescaling, in terms of performance
    q_samples = reverse_sde(step_rng, N, J_test, drift, dispersion, trained_score, train_ts)
    plot_heatmap(q_samples, fname=path+"reverse_heatmap.png")
    plot_samples(q_samples, fname=path+"reverse_samples_qp.png", index=[0, 1], lims=((-3, 3), (-3, 3)))


if __name__ == "__main__":
    main()
