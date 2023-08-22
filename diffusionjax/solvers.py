"""Solver classes."""
import jax
import jax.numpy as jnp
import jax.random as random
from diffusionjax.utils import batch_mul
import abc


class Solver(abc.ABC):
    """Solver abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, num_steps=1000, dt=None, epsilon=None):
        """Construct an SDE.
        Args:
            num_steps: number of discretization time steps.
            dt: time step duration, float or `None`.
                Optional, if provided then final time, t1 = dt * num_steps.
            epsilon: A small float 0. < `epsilon` << 1. The SDE or ODE are integrated to `epsilon` to avoid numerical issues.
        """
        self.num_steps = num_steps
        if dt is not None:
            self.t1 = dt * num_steps
            if epsilon is not None:
                # Defined in forward time, t \in [epsilon, t1], 0 < epsilon << t1
                ts, step = jnp.linspace(epsilon, t1, num_steps, retstep=True)
                self.ts = ts.reshape(-1, 1)
                assert step == (t1 - epsilon) / num_steps
                self.dt = step
            else:
                # Defined in forward time, t \in [dt , t1], 0 < \epsilon << t1
                step = jnp.linspace(0, t1, num_steps + 1)
                self.ts = ts[1:].reshape(-1, 1)
                assert step == dt
                self.dt = step
        else:
            self.t1 = 1.0
            if epsilon is not None:
                self.ts, step = jnp.linspace(epsilon, 1, num_steps, retstep=True)
                assert step == (1. - epsilon) / num_steps
                self.dt = step
            else:
                # Defined in forward time, t \in [dt, 1.0], 0 < dt << 1
                ts, step = jnp.linspace(0, 1, num_steps + 1, retstep=True)
                self.ts = ts[1:].reshape(-1, 1)
                assert step == 1. / num_steps
                self.dt = step

    @abc.abstractmethod
    def update(self, rng, x, t):
        r"""Return the drift and diffusion coefficients of the SDE.

        Args:
            rng: A JAX random state.
            x: A JAX array of the state.
            t: JAX array of the time.

        Returns:
            x: A JAX array of the next state.
            x_mean: A JAX array of the next state without noise, for denoising.
        """


class EulerMaruyama(Solver):
    """Euler Maruyama numerical solver of an SDE.
    Functions are designed for a mini-batch of inputs."""

    def __init__(self, sde, num_steps=1000, dt=None, epsilon=None):
        """Constructs an Euler-Maruyama Solver.
        Args:
            sde: A valid SDE class.
            num_steps: number of discretization time steps.
        """
        super().__init__(num_steps=num_steps, dt=dt, epsilon=epsilon)
        self.sde = sde

    def update(self, rng, x, t):
        """
        Args:
            rng: A JAX random state.
            x: A JAX array representing the current state.
            t: A JAX array representing the current step.

        Returns:
            x: A JAX array of the next state.
            x_mean: A JAX array. The next state without random noise. Useful for denoising.
        """
        drift, diffusion = self.sde.sde(x, t)
        f = drift * self.dt
        G = diffusion * jnp.sqrt(self.dt)
        noise = random.normal(rng, x.shape)
        x_mean = x + f
        x = x_mean + batch_mul(G, noise)
        return x, x_mean


class Annealed(Solver):
    """Annealed Langevin numerical solver of an SDE.
    Functions are designed for a mini-batch of inputs."""

    def __init__(self, sde, snr=1e-2, num_steps=2, dt=None, epsilon=None):
        """Constructs an Annealed Langevin Solver.
        Args:
            sde: A valid SDE class.
        """
        super().__init__(num_steps, dt=dt, epsilon=epsilon)
        self.sde = sde
        self.snr = snr

    def update(self, rng, x, t):
        """
        Args:
            rng: A JAX random state.
            x: A JAX array representing the current state.
            t: A JAX array representing the current step.

        Returns:
            x: A JAX array of the next state.
            x_mean: A JAX array. The next state without random noise. Useful for denoising.
        """
        grad = self.sde.score(x, t)
        grad_norm = jnp.linalg.norm(
            grad.reshape((grad.shape[0], -1)), axis=-1).mean()
        grad_norm = jax.lax.pmean(grad_norm, axis_name='batch')
        noise = random.normal(rng, x.shape)
        noise_norm = jnp.linalg.norm(
            noise.reshape((noise.shape[0], -1)), axis=-1).mean()
        noise_norm = jax.lax.pmean(noise_norm, axis_name='batch')
        # Note: alpha need not be a mini-batch
        alpha = jnp.exp(2 * self.sde.log_mean_coeff(t))
        dt = (self.snr * noise_norm / grad_norm)**2 * 2 * alpha
        x_mean = x + batch_mul(grad, dt)
        x = x_mean + batch_mul(2 * dt, noise)
        return x, x_mean
