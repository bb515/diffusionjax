import ml_collections


def get_default_configs():
    config = ml_collections.ConfigDict()

    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 64
    training.n_iters = 2400001
    training.snapshot_freq = 50000
    training.log_epochs_freq = 10
    training.log_step_freq = 8
    training.eval_freq = 100
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 5000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.score_scaling = True
    training.n_jitted_steps = 1
    training.pmap = False
    training.reduce_mean = True
    training.pointwise_t = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.stack_samples = False
    sampling.denoise = True

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 128

    # data
    config.data = data = ml_collections.ConfigDict()
    data.num_channels = None
    data.image_size = 2

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'mlp'

    # for vp
    model.beta_min = 0.1
    model.beta_max = 20.

    # for ve
    model.sigma_max = 378.
    model.sigma_min = 0.01

    # solver
    config.solver = solver = ml_collections.ConfigDict()
    solver.num_outer_steps = 1000
    solver.num_inner_steps = 1
    solver.outer_solver = 'EulerMaruyama'
    solver.eta = None  # for DDIM
    solver.inner_solver = None
    solver.dt = None
    solver.epsilon = None
    solver.snr = None

    # optimization
    config.seed = 2023
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.warmup = 5000
    optim.weight_decay = False
    optim.grad_clip = None
    optim.beta1 = 0.9
    optim.eps = 1e-8

    return config
