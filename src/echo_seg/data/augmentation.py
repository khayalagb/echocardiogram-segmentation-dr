"""Ultrasound-appropriate augmentation pipelines.

No horizontal flip: cardiac anatomy in apical views is laterally asymmetric —
left-right structures would be swapped.

Two pipelines:
  build_train_transform — standard augmentation (baseline arm)
  build_dr_transform    — physics-inspired domain randomisation arm

DR custom transforms use mean-1 multiplicative speckle of the form
  multiplier = 1 + s * (R/mean(R) - 1),  R ~ Rayleigh, s ~ Uniform(s_lo, s_hi)
so mean(multiplier) = 1 always and std(multiplier) = s * std(R/mean(R)) ≈ s * 0.52.
"""

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Custom DR transforms (ImageOnlyTransform so masks pass through unchanged)
# ---------------------------------------------------------------------------

class _MultiplicativeSpeckle(ImageOnlyTransform):
    """Mean-1 multiplicative i.i.d. Rayleigh speckle.

    multiplier = 1 + s * (R/mean(R) - 1),  R ~ Rayleigh(0.798),  s ~ Uniform(s_lo, s_hi)
    mean(multiplier) = 1 exactly; std(multiplier) ≈ s * 0.52.
    """

    def __init__(self, s_lo: float = 0.1, s_hi: float = 0.8, p: float = 0.5):
        super().__init__(p=p)
        self.s_lo = s_lo
        self.s_hi = s_hi

    def apply(self, img: np.ndarray, s: float = 0.4, **params) -> np.ndarray:
        noise = np.random.rayleigh(scale=0.798, size=img.shape).astype(np.float32)
        noise /= noise.mean() + 1e-8
        multiplier = 1.0 + s * (noise - 1.0)
        return np.clip(img * multiplier, 0.0, 1.0)

    def get_params(self) -> dict:
        return {"s": float(np.random.uniform(self.s_lo, self.s_hi))}

    def get_transform_init_args_names(self) -> tuple:
        return ("s_lo", "s_hi")


class _AnisotropicBlur(ImageOnlyTransform):
    """Anisotropic Gaussian blur mimicking PSF variation.

    sigma_axial (height/depth) and sigma_lateral (width) sampled independently.
    """

    def __init__(
        self,
        sigma_axial: tuple[float, float] = (0.5, 2.0),
        sigma_lateral: tuple[float, float] = (0.3, 1.0),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.sigma_axial = sigma_axial
        self.sigma_lateral = sigma_lateral

    def apply(self, img: np.ndarray, sa: float = 1.0, sl: float = 0.6, **params) -> np.ndarray:
        return gaussian_filter(img, sigma=[sa, sl]).astype(np.float32)

    def get_params(self) -> dict:
        return {
            "sa": float(np.random.uniform(*self.sigma_axial)),
            "sl": float(np.random.uniform(*self.sigma_lateral)),
        }

    def get_transform_init_args_names(self) -> tuple:
        return ("sigma_axial", "sigma_lateral")


class _SectorBrightness(ImageOnlyTransform):
    """Additive brightness shift applied only inside the ultrasound sector.

    Pixels at or below `sector_threshold` (the black border outside the sector)
    are left unchanged. Prevents the border region from acquiring spurious signal.
    `RandomBrightnessContrast` would add to all pixels uniformly, including zeros.
    """

    def __init__(self, brightness_limit: float = 0.2, sector_threshold: float = 0.01, p: float = 0.5):
        super().__init__(p=p)
        self.brightness_limit = brightness_limit
        self.sector_threshold = sector_threshold

    def apply(self, img: np.ndarray, delta: float = 0.0, **params) -> np.ndarray:
        mask = img > self.sector_threshold
        return np.clip(np.where(mask, img + delta, img), 0.0, 1.0)

    def get_params(self) -> dict:
        return {"delta": float(np.random.uniform(-self.brightness_limit, self.brightness_limit))}

    def get_transform_init_args_names(self) -> tuple:
        return ("brightness_limit", "sector_threshold")


class _DepthAttenuation(ImageOnlyTransform):
    """Depth-linear gain ramp simulating tissue attenuation / TGC variation.

    top_gain ~ Uniform(top_lo, top_hi)  (near-field, row 0)
    bot_gain ~ Uniform(bot_lo, bot_hi)  (far-field, last row)
    Ramp interpolated linearly between top and bottom.
    """

    def __init__(
        self,
        top_range: tuple[float, float] = (1.0, 1.3),
        bot_range: tuple[float, float] = (0.4, 0.8),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.top_range = top_range
        self.bot_range = bot_range

    def apply(self, img: np.ndarray, top: float = 1.15, bot: float = 0.6, **params) -> np.ndarray:
        H = img.shape[0]
        ramp = np.linspace(top, bot, H, dtype=np.float32).reshape(H, 1)
        return np.clip(img * ramp, 0.0, 1.0)

    def get_params(self) -> dict:
        return {
            "top": float(np.random.uniform(*self.top_range)),
            "bot": float(np.random.uniform(*self.bot_range)),
        }

    def get_transform_init_args_names(self) -> tuple:
        return ("top_range", "bot_range")


# ---------------------------------------------------------------------------
# Public pipeline builders
# ---------------------------------------------------------------------------

def build_train_transform(
    image_size: tuple[int, int] = (256, 256),
    rotation_limit: int = 15,
    shift_limit: float = 0.1,
    scale_limit: float = 0.1,
    gamma_limit: tuple[int, int] = (70, 130),
    brightness_limit: float = 0.1,
    noise_var_limit: tuple[float, float] = (0.001, 0.005),
) -> A.Compose:
    return A.Compose(
        [
            A.Affine(
                translate_percent={"x": (-shift_limit, shift_limit), "y": (-shift_limit, shift_limit)},
                scale=(1 - scale_limit, 1 + scale_limit),
                rotate=(-rotation_limit, rotation_limit),
                border_mode=0,  # constant padding
                p=0.8,
            ),
            A.RandomGamma(gamma_limit=gamma_limit, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=0,
                p=0.5,
            ),
            A.GaussNoise(std_range=(noise_var_limit[0] ** 0.5, noise_var_limit[1] ** 0.5), p=0.5),
            A.Resize(*image_size),
            ToTensorV2(),
        ]
    )


def build_dr_transform(
    image_size: tuple[int, int] = (256, 256),
    rotation_limit: int = 15,
    shift_limit: float = 0.1,
    scale_limit: float = 0.1,
    gamma_limit: tuple[int, int] = (40, 160),
    brightness_limit: float = 0.2,
    speckle_s_range: tuple[float, float] = (0.1, 0.8),
    blur_sigma_axial: tuple[float, float] = (0.5, 2.0),
    blur_sigma_lateral: tuple[float, float] = (0.3, 1.0),
    attenuation_top: tuple[float, float] = (1.0, 1.3),
    attenuation_bot: tuple[float, float] = (0.4, 0.8),
) -> A.Compose:
    """Physics-inspired domain randomisation augmentation.

    Replaces Gaussian noise with mean-1 multiplicative Rayleigh speckle,
    adds anisotropic PSF blur and depth-attenuation ramp.
    Widens gamma [40,160] and brightness ±0.20 vs standard [70,130] / ±0.10.
    Affine unchanged. No horizontal flip.

    Eval corruption vs DR training range overlap:
      gain_high  (+50%)  : DR brightness ±0.20 (+20% max) — eval HARDER
      speckle    (std≈0.52): DR s∈[0.1,0.8]→std∈[0.05,0.42] — eval HARDER
      blur       (axial σ=2, lat σ=1): DR axial [0.5,2.0], lat [0.3,1.0] — eval AT edge
      attenuation (1.2→0.5): DR top [1.0,1.3], bot [0.4,0.8] — eval WITHIN range
    """
    return A.Compose(
        [
            A.Affine(
                translate_percent={"x": (-shift_limit, shift_limit), "y": (-shift_limit, shift_limit)},
                scale=(1 - scale_limit, 1 + scale_limit),
                rotate=(-rotation_limit, rotation_limit),
                border_mode=0,
                p=0.8,
            ),
            A.RandomGamma(gamma_limit=gamma_limit, p=0.5),
            _SectorBrightness(brightness_limit=brightness_limit, p=0.5),
            _MultiplicativeSpeckle(s_lo=speckle_s_range[0], s_hi=speckle_s_range[1], p=0.5),
            _AnisotropicBlur(
                sigma_axial=blur_sigma_axial,
                sigma_lateral=blur_sigma_lateral,
                p=0.5,
            ),
            _DepthAttenuation(
                top_range=attenuation_top,
                bot_range=attenuation_bot,
                p=0.5,
            ),
            A.Resize(*image_size),
            ToTensorV2(),
        ]
    )


def build_val_transform(image_size: tuple[int, int] = (256, 256)) -> A.Compose:
    return A.Compose([A.Resize(*image_size), ToTensorV2()])
