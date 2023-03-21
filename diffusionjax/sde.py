"""SDE class."""
import abc
from functools import partial
import jax.numpy as jnp
import jax
from diffusionjax.utils import batch_mul


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self):
        """Construct an SDE.
        """

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


class ODLangevin(SDE):
    """Overdamped langevin SDE."""
    def __init__(self, score, damping=2e0, L=1.0):
        super().__init__()
        self.score = score
        self.damping = damping
        self.L = L

    def sde(self, x, t):
        drift = -self.score(x, t)
        diffusion = jnp.ones(x.shape) * jnp.sqrt(2 * self.damping / self.L)
        return drift, diffusion


class UDLangevin(SDE):
    """Underdamped Langevin SDE."""
    def __init__(self, score):
        super().__init__()
        self.score = score

    def sde(self, x, t):
        drift = -self.score(x, t)
        diffusion = jnp.ones(x.shape) * jnp.sqrt(2)
        return drift, diffusion


class OU(SDE):
    """Time rescaled Ohrnstein Uhlenbeck (OU) SDE."""
    def __init__(self, beta_min=0.1, beta_max=20.0):
        super().__init__()
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
        sde = self.sde
        beta_min = self.beta_min
        beta_max = self.beta_max

        class RSDE(self.__class__):

            def __init__(self):
                self.beta_min = beta_min
                self.beta_max = beta_max

            def sde(self, x, t):
                drift, diffusion = sde(x, t)
                drift = -drift + batch_mul(diffusion**2, score(x, t))
                return drift, diffusion

        return RSDE()

    def corrector(self, Corrector, score):

        class CSDE(Corrector, self.__class__):
            def __init__(self, score):
                super().__init__(score)

        return CSDE(score)
