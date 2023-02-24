"""Controllable generation."""
import jax.numpy as jnp
import jax
import jax.random as random
# from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
import functools
from diffusionjax.utils import batch_mul


class Inpainter():
    """Inpainter class."""
    # TODO find out why an inpainter would require access to both predictor and corrector
    # TODO work out if requiring a sampler class in this way is a good thing

    def __init__(self, sampler):
        """Constructs an Inpainter from a sampler.
        Args:
            sampler: A valid sampler class.
        """
        self.sampler = sampler

    def get_update(self):
        sampler_update = self.sampler.get_update()

        def inpaint_update(rng, data, mask, x, t):
            rng, step_rng = random.split(rng)
            vec_t = jnp.ones(data.shape[0]) * t  # 1 - t?
            x, x_mean = update(rng, x, vec_t)
            masked_data_mean, std = self.sampler.sde.marginal_prob(data, vec_t)
            masked_data = masked_data_mean + batch_mul(random.normal(rng, x.shape), std)
            x = x * (1. - mask) + masked_data * mask
            x_mean = x * (1. - mask) + masked_data_mean * mask
            return x, x_mean

        return inpaint_update_fn

    def get_inpainter_enumerate(self, inverse_scaler, snr,
                    n_steps=None, probability_flow=None,
                    denoise=None, epsilon=1e-5):
        """Create an image inpainting function that uses sampler,
        that returns the discretized reverse sde samples.

        Args:
            inverse_scaler: The inverse data normalizer.
            snr: A `float` number. The signal-to-noise ratio for the corrector.
            n_steps: An integer. The number of corrector steps per update of the corrector.
            probability_flow: If `True`, predictor solves the probability flow ODE for sampling.
            denoise: If `True`, add one-step denoising to final samples.
            epsilon: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.
        Returns:
            A pmapped inpainting function.
rng,        """

        update = self.get_update(sampler_update)

        def inpainter(rng, data, mask):
            """Sampler for image inpainting.

            Args:
                rng: A JAX random state.
                data: A JAX array that represents a mini-batch of images to inpaint.
                mask: A {0, 1} array with the same shape of `data`. Value `1` marks known pixels,
                    and value `0` marks pixels that require inpainting.

            Returns:
                Inpainted (complete) images.
            """
            # Initial sample
            rng, step_rng = random.split(rng)
            x = data * mask + random.normal(step_rng, data.shape) * (1. - mask)  # TODO: presuming only taking one sample of inpainted image, no batching over samples
            timesteps = self.sampler.sde.train_ts
            def f(carry, t):
                rng, x, x_mean, i = carry
                vec_t = jnp.ones((1, 1)) * (1 - t)
                rng, step_rng = random.split(rng)
                x, x_mean = update(rng, data, mask, x, t)
                i += 1
                return (rng, x, x_mean, i), ()
            (_, x, _), _ = scan(f, (rng, x, x, 0), ts)
            return x

    def get_inpainter(self, inverse_scaler, snr,
                    n_steps=None, probability_flow=None,
                    denoise=None, epsilon=1e-5):
        """Create an image inpainting function that uses sampler, that returns only the final sample.

        Args:
            inverse_scaler: The inverse data normalizer.
            snr: A `float` number. The signal-to-noise ratio for the corrector.
            n_steps: An integer. The number of corrector steps per update of the corrector.
            probability_flow: If `True`, predictor solves the probability flow ODE for sampling.
            denoise: If `True`, add one-step denoising to final samples.
            epsilon: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.
        Returns:
            A pmapped inpainting function.
        """
        update = self.get_update()

        def inpainter(rng, data, mask, stack_samples=False):
            """Sampler for image inpainting.

            Args:
                rng: A JAX random state.
                data: A JAX array that represents a mini-batch of images to inpaint.
                mask: A {0, 1} array with the same shape of `data`. Value `1` marks known pixels,
                    and value `0` marks pixels that require inpainting.

            Returns:
                Inpainted (complete) images.
            """
            # Initial sample
            rng, step_rng = random.split(rng)
            x = data * mask + random.normal(step_rng, data.shape) * (1. - mask)  # TODO: presuming only taking one sample of inpainted image, no batching over samples
            timesteps = self.sampler.sde.train_ts
            def f(carry, t):
                rng, x, x_mean = carry
                vec_t = jnp.ones((1, 1)) * (1 - t)
                rng, step_rng = random.split(rng)
                x, x_mean = update(step_rng, x, vec_t)
                if not stack_samples:
                    return (rng, x, x_mean), ()
                else:
                    return (rng, x, x_mean), x
            if not stack_samples:
                (_, x, _), _ = scan(f, (rng, x, x), ts)
                return x
            else:
                (_, _, _), xs = scan(f, (rng, x, x), ts)
                return xs
            return sampler
