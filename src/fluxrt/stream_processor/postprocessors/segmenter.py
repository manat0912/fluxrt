import cv2
import numpy as np


class MultiClassSegmenter:
    """Multi-class selfie segmentation: background, hair, face, body, clothes.

    Uses MediaPipe Tasks (selfie_multiclass.tflite) — auto-downloaded from
    MediaPipe's CDN on first use. Falls back to binary person/background if
    model download fails. Both run at 30+ FPS on GPU.
    """

    CATEGORIES = ["background", "hair", "face", "body", "clothes"]
    _MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "image_segmenter/selfie_multiclass_256x256/float32/latest/"
                  "selfie_multiclass_256x256.tflite")

    def __init__(self, model_path: str | None = None):
        self._seg = None
        self._mp = None
        import os as _os
        mp_path = model_path

        if mp_path is None:
            for _try in ["models/mediapipe/selfie_multiclass.tflite",
                         "app/models/mediapipe/selfie_multiclass.tflite",
                         _os.path.expanduser("~/.cache/mediapipe/selfie_multiclass.tflite")]:
                if _os.path.exists(_try):
                    mp_path = _try
                    break

        if mp_path is None:
            cache_dir = _os.path.expanduser("~/.cache/mediapipe")
            mp_path = _os.path.join(cache_dir, "selfie_multiclass.tflite")
            if not _os.path.exists(mp_path):
                try:
                    _os.makedirs(cache_dir, exist_ok=True)
                    print(f"[Segmenter] Downloading selfie_multiclass model...")
                    import urllib.request
                    urllib.request.urlretrieve(self._MODEL_URL, mp_path)
                    print(f"[Segmenter] Downloaded to {mp_path}")
                except Exception as dl_err:
                    print(f"[Segmenter] Model download failed: {dl_err}")
                    mp_path = None

        if mp_path and _os.path.exists(mp_path):
            try:
                from mediapipe.tasks.python import vision
                from mediapipe.tasks.python.core.base_options import BaseOptions
                from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions
                import mediapipe as mp
                self._mp = mp
                base = BaseOptions(model_asset_path=mp_path)
                opts = ImageSegmenterOptions(base_options=base, output_category_mask=True)
                self._seg = ImageSegmenter.create_from_options(opts)
                print(f"[Segmenter] MediaPipe Tasks multi-class loaded ({mp_path})")
            except Exception as e:
                print(f"[Segmenter] MediaPipe Tasks init failed: {e}")

        if self._seg is None:
            print("[Segmenter] No segmentation model available — using no-op (all pixels pass through)")

    def get_class_mask(self, frame_rgb: np.ndarray, class_name: str) -> np.ndarray:
        """Return uint8 mask (255=class, 0=rest) for a single class."""
        if self._seg is not None:
            mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._seg.segment(mp_img)
            mask = result.category_mask.numpy_view()
            idx = self.CATEGORIES.index(class_name) if class_name in self.CATEGORIES else 0
            return (mask == idx).astype(np.uint8) * 255
        return np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.uint8)

    def get_protect_mask(self, frame_rgb: np.ndarray, protect_classes: list[str]) -> np.ndarray:
        """Return uint8 mask: 255 = pixels to protect, 0 = free for diffusion."""
        if not protect_classes or protect_classes == ["none"]:
            return np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.uint8)

        if self._seg is not None:
            mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._seg.segment(mp_img)
            cat_mask = result.category_mask.numpy_view()
            out = np.zeros_like(cat_mask, dtype=np.uint8)
            for cls in protect_classes:
                if cls in self.CATEGORIES:
                    out |= (cat_mask == self.CATEGORIES.index(cls)).astype(np.uint8)
            return (out * 255).astype(np.uint8)

        return np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.uint8)

    @property
    def available_categories(self) -> list[str]:
        if self._seg is not None:
            return self.CATEGORIES
        return ["person", "background"]


class ClickSegmenter:
    """Click-based segmentation using SAM via Ultralytics.

    Usage:
        model = SAM("mobile_sam.pt")
        result = model(frame, points=[[x,y]], labels=[1])
        mask = result[0].masks.data[0].cpu().numpy()

    Falls back to no-op if Ultralytics SAM is not installed.
    """

    def __init__(self):
        self._model = None
        self._init_sam()

    def _init_sam(self):
        try:
            from ultralytics import SAM
            import os as _os
            _model_path = "mobile_sam.pt"
            if not _os.path.exists(_model_path):
                for _try in ["app/mobile_sam.pt", "../mobile_sam.pt", _os.path.expanduser("~/.cache/ultralytics/mobile_sam.pt")]:
                    if _os.path.exists(_try):
                        _model_path = _try
                        break
            self._model = SAM(_model_path)
            print(f"[Segmenter] Ultralytics SAM ({_model_path}) initialized.")
        except Exception as e:
            print(f"[Segmenter] Ultralytics SAM not available: {e}")

    def set_image(self, frame_rgb: np.ndarray):
        pass  # Ultralytics SAM accepts frame directly per predict call

    def predict(self, pos_points: list[tuple[int, int]],
                neg_points: list[tuple[int, int]] | None = None) -> np.ndarray:
        """Return uint8 mask (255=selected, 0=rest) from click points."""
        if not pos_points:
            return np.zeros((480, 640), dtype=np.uint8)

        if self._model is not None:
            try:
                points = pos_points + (neg_points or [])
                labels = [1] * len(pos_points) + [0] * (len(neg_points or []))
                results = self._model(
                    None,
                    points=points,
                    labels=labels,
                    device="cuda",
                    retina_masks=True,
                    verbose=False,
                )
                if results and results[0].masks is not None:
                    mask = results[0].masks.data[0].cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    return mask
            except Exception as e:
                print(f"[Segmenter] SAM predict failed: {e}")

        return np.zeros((480, 640), dtype=np.uint8)


class MaskTracker:
    """Propagates segmentation masks across frames via optical flow.

    Only runs the upstream segmenter once; subsequent frames update
    the mask by warping with DIS optical flow (runs in ~5ms).
    """

    def __init__(self):
        self._flow = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self._prev_gray = None
        self._mask = None

    def set_mask(self, mask: np.ndarray, frame_rgb: np.ndarray):
        self._mask = mask.copy()
        self._prev_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    def propagate(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self._mask is None or self._prev_gray is None:
            if self._prev_gray is not None:
                self._prev_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            return self._mask if self._mask is not None else np.zeros((1, 1), dtype=np.uint8)

        cur_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        flow = self._flow.calc(self._prev_gray, cur_gray, None)
        h, w = self._mask.shape
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = (xs + flow[..., 0]).clip(0, w - 1)
        map_y = (ys + flow[..., 1]).clip(0, h - 1)
        self._mask = cv2.remap(self._mask, map_x, map_y, cv2.INTER_NEAREST)
        self._prev_gray = cur_gray
        return self._mask


class BackgroundCompositor:
    """Master compositor: combines segmenter + click + optical flow tracking.

    Orchestrates the full pipeline:
      1. Auto multi-class path → re-run the (fast) MediaPipe segmenter every
         ``resegment_interval`` frames so the protected region tracks head/body
         motion directly. Optical flow only fills the gaps between re-segments.
      2. Click (SAM) path → segment on click, then track via optical flow
         (SAM is too slow to run every frame).
      3. Apply mask: keep original subject pixels, replace background with diffused
    """

    def __init__(self, resegment_interval: int = 1, mask_feather: int = 9):
        self.class_segmenter = MultiClassSegmenter()
        self.click_segmenter = ClickSegmenter()
        self.tracker = MaskTracker()
        self.protect_classes = ["face", "hair", "body"]
        self.click_points_pos: list[tuple[int, int]] = []
        self.click_points_neg: list[tuple[int, int]] = []
        self.use_clicks = False
        self._recompute = True
        # How often to re-run the auto segmenter (1 = every frame). MediaPipe
        # selfie segmentation runs at 30+ fps so per-frame is affordable and
        # avoids the optical-flow drift that made the mask "flick back".
        self.resegment_interval = max(1, int(resegment_interval))
        # Odd kernel size for Gaussian edge feathering (0 disables).
        self.mask_feather = int(mask_feather)
        self._frame_count = 0

    def set_protect_classes(self, classes: list[str]):
        self.protect_classes = classes
        self._recompute = True

    def add_click_point(self, x: int, y: int, positive: bool = True):
        (self.click_points_pos if positive else self.click_points_neg).append((x, y))
        self._recompute = True
        if positive and self.click_segmenter._model is not None:
            pass  # set_image called externally

    def clear_clicks(self):
        self.click_points_pos.clear()
        self.click_points_neg.clear()
        self._recompute = True

    def get_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        if self.use_clicks and (self.click_points_pos or self.click_points_neg):
            if self._recompute:
                if self.click_segmenter._model is not None:
                    self.click_segmenter.set_image(frame_rgb)
                mask = self.click_segmenter.predict(self.click_points_pos, self.click_points_neg)
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                self.tracker.set_mask(mask, frame_rgb)
                self._recompute = False
            return self.tracker.propagate(frame_rgb)

        # Auto multi-class path: re-segment on the current frame so the mask
        # follows real motion instead of drifting via optical flow alone.
        self._frame_count += 1
        if (
            self._recompute
            or self.resegment_interval <= 1
            or self._frame_count % self.resegment_interval == 0
        ):
            mask = self.class_segmenter.get_protect_mask(frame_rgb, self.protect_classes)
            self.tracker.set_mask(mask, frame_rgb)
            self._recompute = False
            return mask
        return self.tracker.propagate(frame_rgb)

    def composite(self, original_rgb: np.ndarray, diffused_rgb: np.ndarray) -> np.ndarray:
        mask = self.get_mask(original_rgb)
        if self.mask_feather > 1:
            k = self.mask_feather | 1  # force odd kernel size
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        m3 = np.stack([mask] * 3, axis=-1).astype(np.float32) / 255.0
        return (original_rgb * m3 + diffused_rgb * (1.0 - m3)).astype(np.uint8)

    def force_recompute(self):
        self._recompute = True
