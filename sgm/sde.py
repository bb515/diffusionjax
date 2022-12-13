"""SDE class."""
import abc
from functools import partial
import jax.numpy as jnp
from jax import jit
from sgm.utils import batch_mul


from jax.experimental.host_callback import id_print



class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, n_steps):
        """Construct an SDE.
        Args:
            n_steps: number of discretization time steps.
        """
        super().__init__()
        self.n_steps = n_steps
        self.train_ts = jnp.linspace(0, 1, self.n_steps + 1)[:-1]
        # print(self.train_ts.shape)
        # self.train_ts = jnp.linspace(0, 1, self.n_steps + 1)[:-1].reshape(-1, 1)
        print(self.train_ts.shape)

    @abc.abstractmethod
    def sde(self, x, t):
        r"""Parameters to determine the marginal distribution of the SDE,

        .. math::
            p_t(x)

        Args:
            x: a JAX tensor of the state
            t: JAX float of the time
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
        """
        Parameters to determine the marginal distribution of the SDE,

        Args:
            x: a JAX tensor of the state
            t: a JAX float of the time step

        Returns:
            drift: drift function of the forward SDE
            diffusion: dispersion function of the forward SDE
        """
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5 * batch_mul(beta_t, x)
        # drift = -0.5 * beta_t * x  # batch mul
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
        # mean = m * x
        # TODO problem is that sometimes this is used for batch t
        # and other times, scalar t
        id_print(m.shape)
        id_print(x.shape)
        print("Hello")
        print(m.shape)
        print(x.shape)
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
                """
                Parameters to determine the marginal distribution of the reverse SDE,
                
                Args:
                    x: a JAX tensor of the state
                    t: a JAX float of the time step

                Returns:
                    drift: drift function of the forward SDE
                    diffusion: dispersion function of the forward SDE
                """
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
                # rev_f = -f + G**2 * score_fn(x, t)
                return rev_f, G

        return RSDE()


def get_sde(sde_string):
    if sde_string=="OU":
        return OU()
    else:
        return NotImplementedError()

