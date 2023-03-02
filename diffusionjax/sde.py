"""SDE class."""
import abc
from functools import partial
import jax.numpy as jnp
import jax
from diffusionjax.utils import batch_mul


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, n_steps):
        """Construct an SDE.
        Args:
            n_steps: number of discretization time steps.
        """
        self.n_steps = n_steps

    @abc.abstractmethod
    def sde(self, x, t):
        r"""Return the drift and diffusion coefficients of the SDE.

        Args:
            x: a JAX tensor of the state
            t: JAX float of the time

        Returns:
            drift: drift function of the forward SDE
            diffusion: dispersion function of the forward SDE
        """

    def discretize(self, x, t):
        r"""Discretize the SDE in the form,

        .. math::
            x_{i+1} = x_{i} + f_i(x_i) + G_i z_i

        Useful for diffusion sampling and probability flow sampling.

        Args:
            x: a JAX tensor of the state
            t: a JAX float of the time step

        Returns:
            f, G
        """
        dt = 1. / self.n_steps
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * jnp.sqrt(dt)
        return f, G


class ODLangevin(SDE):
    """Overdamped langevin SDE."""
    def __init__(self, score, dt=1e-3, damping=2e0, L=1.0, n_steps=1000):
        super().__init__(n_steps)
        self.score = score
        self.dt = dt
        self.damping = damping
        self.L = L
        t1 = self.dt * n_steps
        self.ts = jnp.linspace(0, t1, self.n_steps + 1)[:-1].reshape(-1, 1)

    def sde(self, x, t):
        drift = -self.score(x, t)
        diffusion = jnp.ones(x.shape) * jnp.sqrt(2 * self.damping / self.L)
        return drift, diffusion

    def discretize(self, x, xd, t):
        drift, diffusion = sde(x, t)
        G = diffusion * jnp.sqrt(self.dt)
        xdd = drift - self.damping * xd  # / densities, assume density is 1.0
        f = xdd * self.dt
        return f, G


class UDLangevin(SDE):
    """Underdamped Langevin SDE."""
    def __init__(self, score, dt=1e-4, n_steps=10000):
        super().__init__(n_steps)
        self.score = score
        self.dt = dt
        t1 = dt * n_steps
        self.ts = jnp.linspace(0, t1, self.n_steps + 1)[:-1].reshape(-1, 1)

    def sde(self, x, t):
        drift = -self.score(x, t)
        diffusion = jnp.ones(x.shape) * jnp.sqrt(2)
        return drift, diffusion

    def discretize(self, x, t):
        r"""Discretize the SDE in the form,

        .. math::
            x_{i+1} = x_{i} + f_i(x_i) + G_i z_i

        Useful for diffusion sampling and probability flow sampling.
        Defaults to Euler-Maryama discretization.

        Args:
            x: a JAX tensor of the state
            t: a JAX float of the time step

        Returns:
            f, G
        """
        drift, diffusion = self.sde(x, t)
        f = drift * self.dt
        G = diffusion * jnp.sqrt(self.dt)
        return f, G


class AnnealedUDLangevin(UDLangevin):
    """Annealed Underdamped Langevin SDE."""

    def __init__(self, score, r=1e-2, beta_min=0.001, beta_max=3, n_steps=3):
        super().__init__(score, n_steps=n_steps)
        self.score = score
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.r = r

    def log_mean_coeff(self, t):
        return -0.5 * t * self.beta_min - 0.25 * t**2 * (self.beta_max - self.beta_min)

    def sde(self, x, t):
        drift = -self.score(x, t)
        diffusion = jnp.ones(x.shape) * jnp.sqrt(2)
        return drift, diffusion

    def discretize(self, x, t):
        r"""Discretize the SDE in the form,

        .. math::
            x_{i+1} = x_{i} + f_i(x_i) + G_i z_i

        Useful for diffusion sampling and probability flow sampling.
        Defaults to Euler-Maryama discretization.

        Args:
            x: a JAX tensor of the state
            t: a JAX float of the time step

        Returns:
            f, G
        """
        alpha = jnp.exp(2 * self.log_mean_coeff(t))
        drift, diffusion = self.sde(x, t)
        d = x.shape[1]
        grad_norm = jnp.linalg.norm(
            drift.reshape((drift.shape[0], -1)), axis=-1).mean()
        epsilon = 2 * alpha * d * (self.r / grad_norm )**2
        f = batch_mul(drift, epsilon)
        G = batch_mul(diffusion, jnp.sqrt(epsilon))
        return f, G


class OU(SDE):
    """Time rescaled Ohrnstein Uhlenbeck (OU) SDE."""
    def __init__(self, beta_min=0.001, beta_max=3, n_steps=1000):
        super().__init__(n_steps)
        self.ts = jnp.linspace(0, 1, self.n_steps + 1)[:-1].reshape(-1, 1)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def sde(self, x, t):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5 * batch_mul(beta_t, x)
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    def log_mean_coeff(self, t):
        return -0.5 * t * self.beta_min - 0.25 * t**2 * (self.beta_max - self.beta_min)

    def mean_coeff(self, t):
        return jnp.exp(self.log_mean_coeff(t))

    def variance(self, t):
        return 1.0 - jnp.exp(2 * self.log_mean_coeff(t))

    def marginal_prob(self, x, t):
        r"""Parameters to determine the marginal distribution of the SDE,

        .. math::
            p_t(x)

        Args:
            x: a JAX tensor of the state
            t: JAX float of the time
        """
        m = self.mean_coeff(t)
        try:
            mean = batch_mul(m, x)
        except:
            mean = m * x
        std = jnp.sqrt(self.variance(t))
        return mean, std

    def reverse(self, score):
        """Create the reverse-time SDE/ODE

        Args:
            score: A time-dependent score-based model that takes x and t and returns the score.
        """
        ts = self.ts
        sde = self.sde
        discretize = self.discretize
        beta_min = self.beta_min
        beta_max = self.beta_max

        class RSDE(self.__class__):

            def __init__(self):
                self.ts = ts
                self.beta_min = beta_min
                self.beta_max = beta_max

            def sde(self, x, t):
                drift, diffusion = sde(x, t)
                score = score(x, t)
                drift = drift - diffusion**2 * score
                return drift, diffusion

            def discretize(self, x, t):
                f, G = discretize(x, t)
                rev_f = -f + batch_mul(G**2, score(x, t))
                return rev_f, G

        return RSDE()


