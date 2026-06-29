import cv2
import numpy as np


class OutputEnhancer:
    """Post-VAE frame enhancement pipeline for live streaming.
    Applies sharpen, denoise, deblock, temporal smoothing, optional upscale.
    Config-driven — zero-cost when disabled.
    """

    def __init__(self, config: dict):
        cfg = config.get("output_enhancer", {})
        self.enabled = cfg.get("enable", False)
        self.sharpen_strength = cfg.get("sharpen_strength", 0.3)
        self.denoise_strength = cfg.get("denoise_strength", 0)
        self.deblock_strength = cfg.get("deblock_strength", 0)
        self.temporal_strength = cfg.get("temporal_strength", 0.15)
        self.upscale_factor = cfg.get("upscale_factor", 0)
        self.contrast_clip = cfg.get("contrast_clip", 0)
        self._prev_frame = None

    def _sharpen(self, img: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 0:
            return img
        blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
        return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)

    def _deblock(self, img: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 0:
            return img
        d = max(3, int(strength * 8) | 1)
        return cv2.bilateralFilter(img, d, strength * 30, strength * 30)

    def _denoise(self, img: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 0:
            return img
        h = max(1, int(strength * 5))
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

    def _temporal_smooth(self, img: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 0 or self._prev_frame is None:
            return img
        if self._prev_frame.shape != img.shape:
            return img
        return cv2.addWeighted(img, 1.0 - strength, self._prev_frame, strength, 0)

    def _auto_contrast(self, img: np.ndarray, clip: float) -> np.ndarray:
        if clip <= 0:
            return img
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def process(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Enhance a uint8 RGB frame in-place.
        Returns enhanced uint8 RGB frame.
        """
        if not self.enabled:
            self._prev_frame = frame_rgb
            return frame_rgb

        result = frame_rgb.copy()

        # 1. Deblock — bilateral filter (edge-preserving smoothing)
        if self.deblock_strength > 0:
            result = self._deblock(result, self.deblock_strength)

        # 2. Denoise
        if self.denoise_strength > 0:
            result = self._denoise(result, self.denoise_strength)

        # 3. Temporal smoothing (flicker reduction)
        if self.temporal_strength > 0:
            result = self._temporal_smooth(result, self.temporal_strength)

        # 4. Sharpen (unsharp mask)
        if self.sharpen_strength > 0:
            result = self._sharpen(result, self.sharpen_strength)

        # 5. Auto-contrast
        if self.contrast_clip > 0:
            result = self._auto_contrast(result, self.contrast_clip)

        # 6. Upscale
        if self.upscale_factor > 1:
            result = cv2.resize(
                result, None,
                fx=self.upscale_factor, fy=self.upscale_factor,
                interpolation=cv2.INTER_LANCZOS4,
            )
            result = self._sharpen(result, 0.3)

        self._prev_frame = result.copy()
        return result
