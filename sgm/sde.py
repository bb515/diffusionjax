"""SDE class."""
import abc
from functools import partial
import jax.numpy as jnp
from jax import jit
from sgm.utils import batch_mul


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, n_steps):
        """Construct an SDE.
        Args:
            n_steps: number of discretization time steps.
        """
        super().__init__()
        self.n_steps = n_steps
        self.train_ts = jnp.linspace(0, 1, self.n_steps + 1)[:-1].reshape(-1, 1)

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

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        r"""Parameters to determine the marginal distribution of the SDE,

        .. math::
            p_t(x)

        Args:
            x: a JAX tensor of the state
            t: JAX float of the time
        """

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
        dt = 1. / self.n_steps
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * jnp.sqrt(dt)
        return f, G


class OU(SDE):
    """Time rescaled Ohrnstein Uhlenbeck (OU) SDE."""
    def __init__(self, beta_min=0.001, beta_max=3, n_steps=1000):
        super().__init__(n_steps)
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
        m = self.mean_coeff(t)
        try:
            mean = batch_mul(m, x)
        except:
            mean = m * x
        std = jnp.sqrt(self.variance(t))
        return mean, std

    def reverse(self, score_fn):
        """Create the reverse-time SDE/ODE

        Args:
            score_fn: A time-dependent score-based model that takes x and t and returns the score.
        """
        train_ts = self.train_ts
        sde_fn = self.sde
        discretize_fn = self.discretize

        class RSDE(self.__class__):

            def __init__(self):
                self.train_ts = train_ts

            def sde(self, x, t):
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion**2 * score
                return drift, diffusion

            def discretize(self, x, t):
                r"""Discretize the SDE in the form,
                .. math::
                    x_{i+1} = x_{i} + f_i(x_i) + G_i z_i

                Useful for reverse diffusion sampling.
                Defaults to Euler-Maryama discretization.

                Args:
                    x: a JAX tensor
                    t: a JAX float representing the time step

                Returns:
                f, G
                """
                f, G = discretize_fn(x, t)
                rev_f = -f + batch_mul(G**2, score_fn(x, t))
                return rev_f, G

        return RSDE()

