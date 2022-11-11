"""SDE class"""
import abc
from functools import partial
import jax.numpy as jnp
from jax import vmap, jit


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, n_steps):
        """Construct an SDE.
        Args:
        n_steps: number of discretization time steps.
        """
        super().__init__()
        self.n_steps = n_steps
        # TODO: work out correct way
        # Doesn't this mean we simulate until reverse time t=1.0, or forward time t=0.0?
        self.train_ts = jnp.linspace(0, 1, self.n_steps + 1)[:-1]
        # train_ts = jnp.linspace(0, 1, R + 1)[1:]
        # train_ts = jnp.arange(1, R)/(R-1)
        # Could this be a nonlinear grid over time?
        # train_ts = jnp.logspace(-4, 0, R)

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass


    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_{i} + f_i(x_i) + G_i z_i

        Useful for reverse diffusion sampling and probability flow sampling.
        Defaults to Euler-Maryama discretization.

        Args:
            x: a JAX tensor
            t: a JAX float representing the time step

        Returns:
            f, G
        """
        dt = 1. / self.n_steps
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * jnp.sqrt(dt)
        return f, G


class OU(SDE):
    """Time rescaled OU"""
    def __init__(self, beta_min=0.001, beta_max=3, n_steps=1000):
        # TODO check that beta min and beta max appropriate. other values for beta_max are 2 or 20, beta_min 0.1
        super().__init__(n_steps)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def sde(self, x, t):
        """
        Returns
        forward_drift: drift function of the forward SDE (we implemented it above)
        disperion: dispersion function of the forward SDE (we implemented it above)
        """
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5 * beta_t * x  # batch mul
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    def log_mean_coeff(self, t):
        return -0.5 * t * self.beta_min - 0.25 * t**2 * (self.beta_max - self.beta_min)  # alpha * (-0.5)

    def mean_coeff(self, t):
        return jnp.exp(self.log_mean_coeff(t))

    def variance(self, t):
        return 1.0 - jnp.exp(2 * self.log_mean_coeff(t))

    def marginal_prob(self, x, t):
        m = self.mean_coeff(t)
        mean = m * x
        std = jnp.sqrt(self.variance(t))
        return mean, std

    def forward_potential(self, x_0, x, t):
        mean, std = self.marginal_prob(x_0, t)
        # TODO: is reshape necessary
        return (x.reshape(-1, 1) - mean) / std**2

    def forward_density(self, x_0, x, t):
        mean, std = self.marginal_prob(x_0, t)
        return norm.pdf(x.reshape(-1, 1), loc=mean, scale=std)

    # TODO make stateless or return a function that can be used in generic solver
    def forward_sde_t(initial, rng, N, n_samples, ts):
        """
        Forward numerical solution of the SDE
        rng: random number generator (JAX rng)
        D: dimension in which the reverse SDE runs
        N_initial: How many samples from the initial distribution N(0, I), number
        score: The score function to use as additional drift in the reverse SDE
        ts: a discretization {t_i} of [0, T], shape 1d-array
        """
        def f(carry, params):
            t, dt = params
            x, xs, i, rng = carry
            rng, step_rng = jax.random.split(rng)  # the missing line
            noise = random.normal(step_rng, x.shape)
            t = jnp.ones((x.shape[0], 1)) * t
            drift = forward_drift(x, t)
            x = x + dt * drift + jnp.sqrt(dt) * dispersion(t) * noise
            xs = xs.at[i, :, :].set(x)
            i += 1
            return (x, xs, i, rng), ()
        dts = ts[1:] - ts[:-1]
        params = jnp.stack([ts[:-1], dts], axis=1)
        xs = jnp.empty((jnp.size(ts), n_samples, N))
        (_, xs, i, _), _ = scan(f, (initial, xs, 0, rng), params)
        return xs, i

    #we jit the function, but we have to mark some of the arguments as static,
    #which means the function is recompiled every time these arguments are changed,
    #since they are directly compiled into the binary code. This is necessary
    #since jitted-functions cannot have functions as arguments. But it also 
    #no problem since these arguments will never/rarely change in our case,
    #therefore not triggering re-compilation.
    @partial(jit, static_argnums=[1, 2, 3, 4])
    def reverse_sde(rng, N, n_samples, score):
        # TODO: may not be possible to jit compile with stateful variables, so may have to return a
        # class for the reverse-time SDE which can be run forward
        """
        rng: random number generator (JAX rng)
        D: dimension in which the reverse SDE runs
        N_initial: How many samples from the initial distribution N(0, I), number
        forward_drift: drift function of the forward SDE (we implemented it above)
        disperion: dispersion function of the forward SDE (we implemented it above)
        score: The score function to use as additional drift in the reverse SDE
        ts: a discretization {t_i} of [0, T], shape 1d-array
        """
        def f(carry, params):
            t, dt = params
            x, rng = carry
            rng, step_rng = jax.random.split(rng)  # the missing line
            noise = random.normal(step_rng, x.shape)
            forward_drift, diffusion = self.sde(x, 1 -t)  # sqrt(beta_t),-0.5 * beta_t * x  # batch mul
            t = jnp.ones((x.shape[0], 1)) * t
            drift = -forward_drift + diffusion**2 * score(x, 1 - t)
            x = x + dt * drift + jnp.sqrt(dt) * diffusion * noise
            return (x, rng), ()
        rng, step_rng = random.split(rng)
        initial = random.normal(step_rng, (n_samples, N))
        dts = self.train_ts[1:] - self.train_ts[:-1]
        params = jnp.stack([self.train_ts[:-1], dts], axis=1)
        (x, _), _ = scan(f, (initial, rng), params)
        return x

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE

        Args:
        score_fn: A time-dependent score-based model that takes x and t and returns the score.
        probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        n_steps = self.n_steps
        sde_fn = self.sde
        discretize_fn = self.discretize
        train_ts = self.train_ts  # definately there is a better way of doing this

        class RSDE(self.__class__):

            def __init__(self):
                self.n_steps = n_steps
                self.probability_flow = probability_flow
                self.train_ts = train_ts

            def sde(self, x, t):
                """Create the drift adn diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion**2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs
                diffusion = jnp.zeros_like(diffusion) if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G**2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = jnp.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


    #we jit the function, but we have to mark some of the arguments as static,
    #which means the function is recompiled every time these arguments are changed,
    #since they are directly compiled into the binary code. This is necessary
    #since jitted-functions cannot have functions as arguments. But it also
    #no problem since these arguments will never/rarely change in our case,
    #therefore not triggering re-compilation.
    @partial(jit, static_argnums=[1, 2,3,4,5])  # removed 1 because that's N
    def reverse_sde_t(rng, N, n_samples, score, ts):
        """
        rng: random number generator (JAX rng)
        D: dimension in which the reverse SDE runs
        N_initial: How many samples from the initial distribution N(0, I), number
        forward_drift: drift function of the forward SDE (we implemented it above)
        disperion: dispersion function of the forward SDE (we implemented it above)
        score: The score function to use as additional drift in the reverse SDE
        ts: a discretization {t_i} of [0, T], shape 1d-array
        """
        def f(carry, params):
            t, dt = params
            x, xs, i, rng = carry
            i += 1
            rng, step_rng = jax.random.split(rng)
            noise = random.normal(step_rng, x.shape)
            forward_drift, diffusion = self.sde(x, 1-t)
            t = jnp.ones((x.shape[0], 1)) * t
            drift = -forward_drift + diffusion**2 * score(x, 1 - t)
            x = x + dt * drift + jnp.sqrt(dt) * diffusion * noise
            xs = xs.at[i, :, :].set(x)
            return (x, xs, i, rng), ()
        rng, step_rng = random.split(rng)
        initial = random.normal(step_rng, (n_samples, N))
        dts = ts[1:] - ts[:-1]
        params = jnp.stack([ts[:-1], dts], axis=1)
        xs = jnp.empty((jnp.size(ts), n_samples, N))
        (_, xs, _, _), _ = scan(f, (initial, xs, 0, rng), params)
        return xs

    # @partial(jit, static_argnums=[1,2,3,4,5])  # removed 1 because that's N
    def reverse_sde_outer(rng, N, n_samples, forward_drift, dispersion, score, ts, indices):
        xs = jnp.empty((jnp.size(indices), n_samples, N))
        j = 0
        for i in indices:
            train_ts = ts[:i]
            x = reverse_sde(rng, N, n_samples, forward_drift, dispersion, score, train_ts)
            xs = xs.at[j, :, :].set(x)
            j += 1
            return xs  # (size(indices), n_samples, N)


def get_sde(sde_string):
    if sde_string=="OU":
        return OU()
    else:
        return NotImplementedError()
