"""Config for `examples/example.py`."""
from configs.default_cs_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = 'vpsde'
    # training.sde = 'vesde'
    training.n_iters = 4000
    training.batch_size = 8
    training.likelihood_weighting = False
    training.score_scaling = True
    training.reduce_mean = True
    training.log_epoch_freq = 1
    training.log_step_freq = 8000
    training.pmap = False
    training.n_jitted_steps = 1
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq = 8000
    training.snapshot_freq_for_preemption = 8000
    training.eval_freq = 8000

    # eval
    eval = config.eval
    eval.batch_size = 1000

    # sampling
    sampling = config.sampling
    sampling.denoise = True

    # data
    data = config.data
    data.image_size = 2
    data.num_channels = None

    # model
    model = config.model
    # for vp
    model.beta_min = 0.01
    model.beta_max = 3.
    # for ve
    model.sigma_min = 0.01
    model.sigma_max = 10.

    # solver
    solver = config.solver
    solver.num_outer_steps = 1000
    solver.outer_solver = 'EulerMaruyama'
    solver.inner_solver = None

    # optim
    optim = config.optim
    optim.optimizer = 'Adam'
    optim.lr = 1e-3
    optim.warmup = False
    optim.weight_decay = False
    optim.grad_clip = None

    config.seed = 2023

    return config
