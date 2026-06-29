import argparse
import json
import threading
import time
import sys
import asyncio

def patch_asyncio():
    import asyncio
    import sys
    
    # -- Direct monkey-patch _ProactorBasePipeTransport to suppress WinError 10054 --
    # This is the definitive fix for "Exception in callback
    # _ProactorBasePipeTransport._call_connection_lost(None)" on Windows.
    if sys.platform == "win32":
        try:
            import asyncio.proactor_events as _proactor_events
            _orig_call_connection_lost = _proactor_events._ProactorBasePipeTransport._call_connection_lost
            def _patched_call_connection_lost(self, exc):
                try:
                    _orig_call_connection_lost(self, exc)
                except ConnectionResetError:
                    pass
            _proactor_events._ProactorBasePipeTransport._call_connection_lost = _patched_call_connection_lost
        except Exception:
            pass
    
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    def exception_handler(loop, context):
        exception = context.get("exception")
        if isinstance(exception, ConnectionResetError) or (exception and "[WinError 10054]" in str(exception)):
            return
        message = context.get("message", "")
        if "connection_lost" in message or "_call_connection_lost" in message:
            return
        try:
            loop.default_exception_handler(context)
        except Exception:
            pass

    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(exception_handler)
    except Exception:
        pass

    try:
        orig_new_loop = asyncio.new_event_loop
        def patched_new_loop(*args, **kwargs):
            loop = orig_new_loop(*args, **kwargs)
            try:
                loop.set_exception_handler(exception_handler)
            except Exception:
                pass
            return loop
        asyncio.new_event_loop = patched_new_loop
    except Exception:
        pass

patch_asyncio()

import cv2
import numpy as np
from PIL import Image, ImageDraw

import multiprocessing

is_main_process = (multiprocessing.current_process().name == 'MainProcess' and 
                   (not hasattr(multiprocessing, 'parent_process') or multiprocessing.parent_process() is None))

if is_main_process:
    import streamlit as st
    from streamlit.runtime.scriptrunner import add_script_run_ctx
else:
    class DummySt:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    st = DummySt()
    def add_script_run_ctx(*args, **kwargs):
        pass
from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle

try:
    import sounddevice as sd
except Exception as sd_err:
    sd = None
    print(f"[WARNING] sounddevice import failed: {sd_err}")

DEFAULT_CONFIG = "configs/stream_processor_config.json"
CAM_BACKEND = cv2.CAP_DSHOW
POLL_DELAY = 0.04


def parse_args():
    parser = argparse.ArgumentParser(description="FluxRT Streamlit UI")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config path")
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    _args = parse_args()
    _config_path = _args.config
    _use_int8 = _args.int8
else:
    _config_path = DEFAULT_CONFIG
    _use_int8 = False


def _get_placeholder_image(text: str, width: int = 576, height: int = 320) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(18, 18, 20))
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, width - 2, height - 2], outline=(40, 40, 44), width=1)
    cx, cy = width // 2, height // 2
    text_w = len(text) * 6
    text_h = 10
    draw.text((cx - text_w // 2, cy - text_h // 2), text, fill=(100, 110, 120))
    return img


def _enumerate_cameras() -> list[tuple[str, int]]:
    found = []
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            devices = graph.get_input_devices()
            for idx, name in enumerate(devices):
                found.append((name, idx))
        except Exception as e:
            print(f"[WARNING] pygrabber failed to enumerate cameras: {e}")

    if not found:
        # Fallback to standard cv2 probing
        backends = [cv2.CAP_DSHOW] if sys.platform == "win32" else [None]
        for i in range(8):  # Check up to 8 potential cameras
            for backend in backends:
                try:
                    if backend is None:
                        cap = cv2.VideoCapture(i)
                    else:
                        cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        # Test read a frame to confirm it is actually yielding image data
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            found.append((f"Camera {i}", i))
                            cap.release()
                            break
                        cap.release()
                except Exception:
                    pass
    if not found:
        found = [("Camera 0", 0)]
    return found


def _load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


audio_vol = 0.0

def _background_loop(config_path: str, use_int8: bool, state: "SharedState") -> None:
    """Background: init StreamProcessor, capture camera, write/read tensors."""
    global audio_vol
    audio_vol = 0.0
    audio_stream = None
    
    try:
        config = _load_config(config_path)
        # Apply the dynamic mode selected on startup
        config["use_reference_image"] = state.started_with_img2img
        
        # Resolve resolution from SharedState if specified
        if getattr(state, "resolution_str", None):
            try:
                rw, rh = map(int, state.resolution_str.split("x"))
                config["resolution"]["width"] = rw
                config["resolution"]["height"] = rh
            except Exception:
                pass

        h = config["resolution"]["height"]
        w = config["resolution"]["width"]

        sp = StreamProcessor(config)
        if use_int8:
            sp.enable_quantization()
        sp.start()
        state.sp = sp
        state.res = sp.get_resolution()
        
        # Initialize advanced generation settings from state
        sp.set_param("cfg_scale", state.cfg_scale)
        sp.set_param("distilled_mode", state.distilled_mode)
        sp.set_param("denoise", state.denoise)
        sp.set_param("noise_offset", state.noise_offset)
        sp.set_param("guidance_rescale", state.guidance_rescale)
        sp.set_param("sampler", state.sampler)
        sp.set_param("sigma_min", state.sigma_min)
        sp.set_param("sigma_max", state.sigma_max)
        sp.set_param("eta", state.eta)
        sp.set_param("temporal_smoothing", state.temporal_smoothing)
        sp.set_param("vae_decode_scaling", state.vae_decode_scaling)

        # Push the user's current prompt so the subprocess uses what the UI shows,
        # not just the config's default_prompt (they may differ if the user typed
        # something before clicking Start).
        sp.set_prompt(state.prompt)
        # Push negative prompt if one has been entered.
        if state.negative_prompt:
            sp.set_param("negative_prompt", state.negative_prompt)

        # If there is already an uploaded asset image, pass it to the stream processor
        if state.asset_img is not None:
            sp.set_reference_image(state.asset_img)
            
        input_tensor = sp.get_input_tensor()
        output_tensor = sp.get_output_tensor()

        # Start microphone audio recording via sounddevice
        if sd is not None:
            def audio_callback(indata, frames, time, status):
                global audio_vol
                rms = np.sqrt(np.mean(indata**2))
                audio_vol = float(rms)
            try:
                audio_stream = sd.InputStream(
                    callback=audio_callback,
                    channels=1,
                    samplerate=16000,
                    blocksize=1024
                )
                audio_stream.start()
                print("[Streamlit] Audio input stream started successfully.")
            except Exception as e:
                print(f"[WARNING] Could not start audio input stream: {e}")

        # On Windows, use DirectShow (DSHOW) only to ensure maximum compatibility and avoid MSMF driver conflicts
        backends = [cv2.CAP_DSHOW] if sys.platform == "win32" else [None]
        cap = None
        for backend in backends:
            try:
                if backend is None:
                    c = cv2.VideoCapture(state.cam_idx)
                else:
                    c = cv2.VideoCapture(state.cam_idx, backend)
                if c.isOpened():
                    # Initialize camera focus settings based on SharedState
                    try:
                        c.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if state.autofocus else 0.0)
                        if not state.autofocus:
                            c.set(cv2.CAP_PROP_FOCUS, float(state.manual_focus))
                        print(f"[Streamlit] Camera focus initialized (Autofocus={state.autofocus}, FocusValue={state.manual_focus})")
                    except Exception as autofocus_err:
                        print(f"[WARNING] Failed to initialize camera focus: {autofocus_err}")
                    
                    # Test read a frame to confirm we actually get frames
                    ret, frame = c.read()
                    if ret and frame is not None:
                        cap = c
                        print(f"[Streamlit] Successfully opened camera {state.cam_idx} with backend {backend}")
                        break
                    else:
                        c.release()
            except Exception as e:
                print(f"[WARNING] Failed to open camera with backend {backend}: {e}")
        if cap is None or not cap.isOpened():
            state.error = f"Cannot open camera {state.cam_idx}"
            state.running = False
            if audio_stream is not None:
                try:
                    audio_stream.stop()
                    audio_stream.close()
                except Exception:
                    pass
            return

        while state.running and not state.stop_event.is_set():
            # Check model subprocess status to detect crashes immediately
            if state.sp is not None:
                status_dict = state.sp.get_model_status()
                status = status_dict.get("status", "")
                error_msg = status_dict.get("error", "")
                if status == "Error":
                    raise RuntimeError(error_msg or "Model subprocess crashed.")

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            cropped = crop_maximal_rectangle(frame, h, w)
            
            # Dynamic param updates
            if state.prompt_dirty:
                sp.set_prompt(state.prompt)
                state.prompt_dirty = False
            if state.negative_prompt_dirty:
                sp.set_param("negative_prompt", state.negative_prompt)
                state.negative_prompt_dirty = False
            if state.steps_dirty:
                sp.set_steps(state.steps)
                state.steps_dirty = False

            # Dynamic advanced param updates
            if state.params_dirty:
                sp.set_param("cfg_scale", state.cfg_scale)
                sp.set_param("distilled_mode", state.distilled_mode)
                sp.set_param("denoise", state.denoise)
                sp.set_param("noise_offset", state.noise_offset)
                sp.set_param("guidance_rescale", state.guidance_rescale)
                sp.set_param("sampler", state.sampler)
                sp.set_param("sigma_min", state.sigma_min)
                sp.set_param("sigma_max", state.sigma_max)
                sp.set_param("eta", state.eta)
                sp.set_param("temporal_smoothing", state.temporal_smoothing)
                sp.set_param("vae_decode_scaling", state.vae_decode_scaling)
                state.params_dirty = False

            # Segmentation & enhancer dynamic updates
            if state.enhancer_dirty:
                sp.set_param("segmentation_enabled", state.segmentation_enabled)
                sp.set_param("adaptive_interp", state.adaptive_interp)
                sp.set_param("enhancer_sharpen", state.enhancer_sharpen)
                sp.set_param("enhancer_deblock", state.enhancer_deblock)
                sp.set_param("enhancer_denoise", state.enhancer_denoise)
                sp.set_param("enhancer_temporal", state.enhancer_temporal)
                sp.set_param("enhancer_contrast", state.enhancer_contrast)
                sp.set_param("protect_classes", state.protect_classes)
                sp.set_param("use_click_mask", state.use_click_mask)
                state.enhancer_dirty = False

            # Click mask points update
            if state.click_dirty:
                if state.use_click_mask and state.click_points_pos:
                    import json
                    payload = json.dumps({"pos": state.click_points_pos, "neg": state.click_points_neg})
                    sp.set_param("click_points", payload)
                else:
                    sp.set_param("clear_clicks", True)
                state.click_dirty = False

            # Dynamic focus updates
            if state.open_settings:
                try:
                    print("[Streamlit] Opening native camera settings dialog...")
                    cap.set(cv2.CAP_PROP_SETTINGS, 1.0)
                except Exception as settings_err:
                    print(f"[WARNING] Failed to open camera settings dialog: {settings_err}")
                state.open_settings = False

            if state.focus_dirty:
                try:
                    if state.autofocus:
                        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0)
                        print("[Streamlit] Enabled camera autofocus.")
                    else:
                        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
                        cap.set(cv2.CAP_PROP_FOCUS, float(state.manual_focus))
                        print(f"[Streamlit] Set manual focus to {state.manual_focus}.")
                except Exception as focus_err:
                    print(f"[WARNING] Failed to update camera focus settings: {focus_err}")
                state.focus_dirty = False
                
            # Lipsync & face swap settings
            lip_active = (state.asset_mode == "Change Face (Face Swap)") or state.lip_active
            sp.set_lip_transfer(lip_active)
            sp.set_param("swap_face", state.asset_mode == "Change Face (Face Swap)")
            sp.set_param("audio_volume", audio_vol)
            
            # Dynamic asset image update
            if state.asset_dirty:
                sp.set_reference_image(state.asset_img)
                state.asset_dirty = False

            input_tensor.copy_from(cropped)
            output_bgr = output_tensor.to_numpy()
            state.frame_in = cropped
            state.frame_out = output_bgr
            state.frame_id += 1
            state.ready = True

        cap.release()
        if audio_stream is not None:
            try:
                audio_stream.stop()
                audio_stream.close()
                print("[Streamlit] Audio input stream closed.")
            except Exception:
                pass
        sp.stop()
        state.sp = None
    except Exception as exc:
        import traceback
        traceback.print_exc()
        state.error = str(exc)
        state.running = False
        if audio_stream is not None:
            try:
                audio_stream.stop()
                audio_stream.close()
            except Exception:
                pass


# ── Page config (MUST be first st.* call) ───────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(page_title="FluxRT", page_icon="🎨", layout="wide")

# ── Session init ────────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.running = False
        self.stop_event = threading.Event()
        self.frame_in = None
        self.frame_out = None
        self.frame_id = 0
        self.ready = False
        self.error = ""
        self.sp = None
        self.res = None
        
        # Inputs from UI
        self.prompt = ""
        self.prompt_dirty = False
        self.negative_prompt = ""
        self.negative_prompt_dirty = False
        self.steps = 2
        self.steps_dirty = False
        
        self.asset_mode = "None"
        self.asset_img = None
        self.asset_dirty = False
        
        self.lip_active = False
        self.cam_idx = 0
        self.started_with_img2img = False

        # Focus controls
        self.autofocus = True
        self.manual_focus = 38
        self.focus_dirty = False
        self.open_settings = False

        # Advanced generation settings
        self.cfg_scale = 4.5
        self.distilled_mode = False
        self.denoise = 0.55
        self.noise_offset = 0.2
        self.guidance_rescale = 0.35
        self.sampler = "Euler"
        self.sigma_min = 0.1
        self.sigma_max = 6.0
        self.eta = 0.1
        self.temporal_smoothing = 0.35
        self.vae_decode_scaling = 0.85
        self.resolution_str = "576x320"
        self.params_dirty = False

        # Segmentation & output enhancer
        self.segmentation_enabled = False
        self.adaptive_interp = False
        self.enhancer_sharpen = 0.5
        self.enhancer_deblock = 0.2
        self.enhancer_denoise = 0.0
        self.enhancer_temporal = 0.0
        self.enhancer_contrast = 0.0
        self.enhancer_dirty = False

        # Multi-class mask & click points
        self.protect_classes = ["face", "hair", "body"]
        self.use_click_mask = False
        self.click_points_pos: list = []
        self.click_points_neg: list = []
        self.click_dirty = False

if __name__ == "__main__":
    if "_shared" not in st.session_state:
        st.session_state._shared = SharedState()
        st.session_state._cam_list = _enumerate_cameras()
        st.session_state._thread = None
        st.session_state._last_asset_name = None

    # ── Sidebar ─────────────────────────────────────────────────────────────────
    st.sidebar.title("FluxRT")

    config_path = st.sidebar.text_input("Config path", _config_path)
    try:
        cfg = _load_config(config_path)
    except Exception:
        cfg = {}

    default_prompt = cfg.get("default_prompt", "")
    default_steps = cfg.get("default_steps", 5)
    lip_enabled = cfg.get("lip_transfer", {}).get("enable", False)

    prompt = st.sidebar.text_area("Prompt", default_prompt, height=80)
    negative_prompt = st.sidebar.text_area("Negative Prompt", st.session_state._shared.negative_prompt, height=60, placeholder="e.g. blurry, low quality, distorted")
    steps = st.sidebar.slider("Inference steps", 1, 30, default_steps)
    exp = st.sidebar.slider("Interpolation", 0, 5, cfg.get("interpolation_exp", 2))
    adaptive_interp = st.sidebar.checkbox("Adaptive Interpolation (reduce on motion)", value=st.session_state._shared.adaptive_interp)

    if prompt != st.session_state._shared.prompt:
        st.session_state._shared.prompt = prompt
        st.session_state._shared.prompt_dirty = True

    if negative_prompt != st.session_state._shared.negative_prompt:
        st.session_state._shared.negative_prompt = negative_prompt
        st.session_state._shared.negative_prompt_dirty = True

    if steps != st.session_state._shared.steps:
        st.session_state._shared.steps = steps
        st.session_state._shared.steps_dirty = True

    # Camera select layout
    cam_col, ref_col = st.sidebar.columns([3, 1])
    with cam_col:
        available = st.session_state.get("_cam_list", [("Camera 0", 0)])
        cam_labels = [item[0] for item in available]
        if not cam_labels:
            cam_labels = ["Camera 0"]
        sel = st.selectbox("Camera", cam_labels, index=0)
        cam_idx = 0
        for name, idx in available:
            if name == sel:
                cam_idx = idx
                break
    with ref_col:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄", help="Scan for cameras", width="stretch"):
            st.session_state._cam_list = _enumerate_cameras()
            st.rerun()

    # Focus controls in UI
    st.sidebar.subheader("Camera Focus Controls")
    autofocus = st.sidebar.checkbox("Autofocus", value=st.session_state._shared.autofocus)
    if autofocus != st.session_state._shared.autofocus:
        st.session_state._shared.autofocus = autofocus
        st.session_state._shared.focus_dirty = True

    if not autofocus:
        manual_focus = st.sidebar.slider("Manual Focus Distance", 0, 255, st.session_state._shared.manual_focus, step=5)
        if manual_focus != st.session_state._shared.manual_focus:
            st.session_state._shared.manual_focus = manual_focus
            st.session_state._shared.focus_dirty = True

    if sys.platform == "win32":
        if st.sidebar.button("⚙️ Open Camera Driver Settings", help="Open native Windows camera settings window to manually adjust focus, exposure, and white balance"):
            st.session_state._shared.open_settings = True

    if lip_enabled:
        st.session_state._shared.lip_active = st.sidebar.checkbox("Enable lip transfer (LivePortrait)", value=st.session_state._shared.lip_active)

    # TensorRT status indicator badge
    try:
        import onnxruntime as _ort
        if "TensorrtExecutionProvider" in _ort.get_available_providers():
            st.sidebar.success("⚡ TensorRT ONNX Accelerator Active")
        else:
            st.sidebar.warning("⚠️ TensorRT Accelerator Unavailable")
    except Exception as e:
        st.sidebar.error(f"Error checking TensorRT status: {e}")

    # Advanced settings expander
    st.sidebar.divider()
    with st.sidebar.expander("Advanced Settings", expanded=False):
        distilled_mode = st.checkbox("Distilled Mode (guidance_scale=1.0)", value=st.session_state._shared.distilled_mode, help="When ON, forces CFG scale to 1.0 (no guidance). Use for turbo/distilled models. When OFF, uses the CFG Scale slider value below.")
        cfg_scale = st.slider("CFG Scale", 1.0, 10.0, st.session_state._shared.cfg_scale, 0.1, disabled=distilled_mode)
        denoise = st.slider("Denoise", 0.0, 1.0, st.session_state._shared.denoise, 0.05)
        noise_offset = st.slider("Noise Offset", 0.0, 1.0, st.session_state._shared.noise_offset, 0.05)
        guidance_rescale = st.slider("Guidance Rescale", 0.0, 1.0, st.session_state._shared.guidance_rescale, 0.05)
        
        sampler = st.selectbox(
            "Sampler",
            ["Euler", "Heun", "Euler Ancestral", "DPM++ 2M"],
            index=["Euler", "Heun", "Euler Ancestral", "DPM++ 2M"].index(st.session_state._shared.sampler)
        )
        
        sigma_min = st.slider("Sigma Min", 0.0, 2.0, st.session_state._shared.sigma_min, 0.05)
        sigma_max = st.slider("Sigma Max", 1.0, 20.0, st.session_state._shared.sigma_max, 0.5)
        eta = st.slider("ETA", 0.0, 1.0, st.session_state._shared.eta, 0.05)
        temporal_smoothing = st.slider("Temporal Smoothing", 0.0, 1.0, st.session_state._shared.temporal_smoothing, 0.05)
        vae_decode_scaling = st.slider("VAE Decode Scaling", 0.1, 1.5, st.session_state._shared.vae_decode_scaling, 0.05)
        
        st.divider()
        st.markdown("**Segmentation Mask**")
        segmentation_enabled = st.checkbox("Person-preserving background composite", value=st.session_state._shared.segmentation_enabled, help="Keeps person pixels untouched; only background is diffused")
        use_click_mask = st.checkbox("Use click-based mask (SAM)", value=st.session_state._shared.use_click_mask, help="Click points on the input frame to define what to protect")
        protect_classes = st.multiselect(
            "Auto-protect regions",
            ["face", "hair", "body", "clothes"],
            default=st.session_state._shared.protect_classes,
            help="Regions to preserve from the original webcam frame",
        )
        
        st.divider()
        st.markdown("**Output Enhancer (post-processing)**")
        enhancer_sharpen = st.slider("Sharpen", 0.0, 2.0, st.session_state._shared.enhancer_sharpen, 0.05)
        enhancer_deblock = st.slider("Deblock (bilateral filter)", 0.0, 1.0, st.session_state._shared.enhancer_deblock, 0.05)
        enhancer_denoise = st.slider("Denoise", 0.0, 1.0, st.session_state._shared.enhancer_denoise, 0.05)
        enhancer_temporal = st.slider("Temporal Smooth", 0.0, 1.0, st.session_state._shared.enhancer_temporal, 0.05)
        enhancer_contrast = st.slider("CLAHE Contrast", 0.0, 5.0, st.session_state._shared.enhancer_contrast, 0.1)
        
        resolutions = ["576x320", "640x360", "768x432", "512x512"]
        res_str = st.selectbox(
            "Resolution",
            resolutions,
            index=resolutions.index(st.session_state._shared.resolution_str)
        )

        if st.session_state._shared.running:
            if res_str != st.session_state._shared.resolution_str:
                st.sidebar.warning("⚠️ Restart needed to apply new resolution.")
        else:
            st.session_state._shared.resolution_str = res_str

    if distilled_mode != st.session_state._shared.distilled_mode:
        st.session_state._shared.distilled_mode = distilled_mode
        st.session_state._shared.params_dirty = True
    if cfg_scale != st.session_state._shared.cfg_scale:
        st.session_state._shared.cfg_scale = cfg_scale
        st.session_state._shared.params_dirty = True
    if denoise != st.session_state._shared.denoise:
        st.session_state._shared.denoise = denoise
        st.session_state._shared.params_dirty = True
    if noise_offset != st.session_state._shared.noise_offset:
        st.session_state._shared.noise_offset = noise_offset
        st.session_state._shared.params_dirty = True
    if guidance_rescale != st.session_state._shared.guidance_rescale:
        st.session_state._shared.guidance_rescale = guidance_rescale
        st.session_state._shared.params_dirty = True
    if sampler != st.session_state._shared.sampler:
        st.session_state._shared.sampler = sampler
        st.session_state._shared.params_dirty = True
    if sigma_min != st.session_state._shared.sigma_min:
        st.session_state._shared.sigma_min = sigma_min
        st.session_state._shared.params_dirty = True
    if sigma_max != st.session_state._shared.sigma_max:
        st.session_state._shared.sigma_max = sigma_max
        st.session_state._shared.params_dirty = True
    if eta != st.session_state._shared.eta:
        st.session_state._shared.eta = eta
        st.session_state._shared.params_dirty = True
    if temporal_smoothing != st.session_state._shared.temporal_smoothing:
        st.session_state._shared.temporal_smoothing = temporal_smoothing
        st.session_state._shared.params_dirty = True
    if vae_decode_scaling != st.session_state._shared.vae_decode_scaling:
        st.session_state._shared.vae_decode_scaling = vae_decode_scaling
        st.session_state._shared.params_dirty = True
    if adaptive_interp != st.session_state._shared.adaptive_interp:
        st.session_state._shared.adaptive_interp = adaptive_interp
        st.session_state._shared.enhancer_dirty = True
    if segmentation_enabled != st.session_state._shared.segmentation_enabled:
        st.session_state._shared.segmentation_enabled = segmentation_enabled
        st.session_state._shared.enhancer_dirty = True
    if use_click_mask != st.session_state._shared.use_click_mask:
        st.session_state._shared.use_click_mask = use_click_mask
        st.session_state._shared.enhancer_dirty = True
    if protect_classes != st.session_state._shared.protect_classes:
        st.session_state._shared.protect_classes = protect_classes
        st.session_state._shared.enhancer_dirty = True
    if enhancer_sharpen != st.session_state._shared.enhancer_sharpen:
        st.session_state._shared.enhancer_sharpen = enhancer_sharpen
        st.session_state._shared.enhancer_dirty = True
    if enhancer_deblock != st.session_state._shared.enhancer_deblock:
        st.session_state._shared.enhancer_deblock = enhancer_deblock
        st.session_state._shared.enhancer_dirty = True
    if enhancer_denoise != st.session_state._shared.enhancer_denoise:
        st.session_state._shared.enhancer_denoise = enhancer_denoise
        st.session_state._shared.enhancer_dirty = True
    if enhancer_temporal != st.session_state._shared.enhancer_temporal:
        st.session_state._shared.enhancer_temporal = enhancer_temporal
        st.session_state._shared.enhancer_dirty = True
    if enhancer_contrast != st.session_state._shared.enhancer_contrast:
        st.session_state._shared.enhancer_contrast = enhancer_contrast
        st.session_state._shared.enhancer_dirty = True

    st.sidebar.divider()
    start_label = "⏹ Stop" if st.session_state._shared.running else "▶ Start"
    if st.sidebar.button(start_label, width="stretch"):
        if st.session_state._shared.running:
            st.session_state._shared.stop_event.set()
            st.session_state._shared.running = False
            st.session_state._shared.ready = False
            if getattr(st.session_state, "_thread", None) is not None:
                st.session_state._thread.join(timeout=2.0)
                st.session_state._thread = None
        else:
            # Just in case, join any lingering previous thread to release camera
            if getattr(st.session_state, "_thread", None) is not None:
                if st.session_state._thread.is_alive():
                    st.session_state._shared.stop_event.set()
                    st.session_state._thread.join(timeout=2.0)
                st.session_state._thread = None

            st.session_state._shared.cam_idx = cam_idx
            st.session_state._shared.prompt = prompt
            st.session_state._shared.prompt_dirty = False
            st.session_state._shared.steps = steps
            st.session_state._shared.steps_dirty = False
            st.session_state._shared.resolution_str = res_str
            st.session_state._shared.started_with_img2img = (st.session_state._shared.asset_mode == "Change Clothes (Img2Img)")
            st.session_state._shared.enhancer_dirty = True
            st.session_state._shared.stop_event.clear()
            st.session_state._shared.running = True
            st.session_state._shared.error = ""
            st.session_state._shared.frame_in = None
            st.session_state._shared.frame_out = None
            t = threading.Thread(
                target=_background_loop,
                args=(config_path, _use_int8, st.session_state._shared),
                daemon=True,
            )
            st.session_state._thread = t
            t.start()

    if st.session_state._shared.error:
        st.sidebar.error(st.session_state._shared.error)
        st.session_state._shared.error = ""

    # ── Main area ───────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1.2, 2])
    w = cfg.get("resolution", {}).get("width", 576)
    h = cfg.get("resolution", {}).get("height", 320)
    if st.session_state._shared.running and st.session_state._shared.res is not None:
        w = st.session_state._shared.res.get("width", w)
        h = st.session_state._shared.res.get("height", h)
    else:
        try:
            w, h = map(int, st.session_state._shared.resolution_str.split("x"))
        except Exception:
            pass

    with col1:
        st.subheader("Input")
        with st.container(border=True):
            in_placeholder = st.empty()
            if not st.session_state._shared.running:
                placeholder_img = _get_placeholder_image("Camera Feed (Offline)", w, h)
                in_placeholder.image(placeholder_img, width="stretch")

    with col2:
        st.subheader("Assets")
        with st.container(border=True):
            asset_mode = st.selectbox(
                "Asset Mode",
                ["None", "Change Clothes (Img2Img)", "Change Face (Face Swap)"],
                index=["None", "Change Clothes (Img2Img)", "Change Face (Face Swap)"].index(st.session_state._shared.asset_mode)
            )
            if asset_mode != st.session_state._shared.asset_mode:
                st.session_state._shared.asset_mode = asset_mode

            if st.session_state._shared.running:
                started_img2img = st.session_state._shared.started_with_img2img
                current_is_img2img = (asset_mode == "Change Clothes (Img2Img)")
                if started_img2img != current_is_img2img:
                    st.warning("⚠️ Restart needed to switch Clothes mode.")

            uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        
            if uploaded_file is not None:
                st.image(uploaded_file, width="stretch")
                if st.session_state.get("_last_asset_name", None) != uploaded_file.name:
                    st.session_state._last_asset_name = uploaded_file.name
                    try:
                        asset_pil = Image.open(uploaded_file).convert("RGB")
                        st.session_state._shared.asset_img = np.array(asset_pil)
                        st.session_state._shared.asset_dirty = True
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
            else:
                if st.session_state.get("_last_asset_name", None) is not None:
                    st.session_state._last_asset_name = None
                    st.session_state._shared.asset_img = None
                    st.session_state._shared.asset_dirty = True
            
                placeholder_img = _get_placeholder_image("No Asset Uploaded", 280, 200)
                st.image(placeholder_img, width="stretch")

    with col3:
        st.subheader("Output")
        with st.container(border=True):
            out_placeholder = st.empty()
            if not st.session_state._shared.running:
                placeholder_img = _get_placeholder_image("AI Output (Offline)", w, h)
                out_placeholder.image(placeholder_img, width="stretch")

    if st.session_state._shared.running:
        st.caption("🟢 Running")
    else:
        st.caption("⏸ Stopped — press Start in the sidebar")

    # ── Click mask canvas (shown when click mode is active) ────────────────────
    if st.session_state._shared.running and st.session_state._shared.use_click_mask:
        with st.expander("🎯 Click Segmentation Mask", expanded=True):
            st.caption("Click + to mark regions to protect, − to mark regions to ignore. Mask generated via SAM + optical flow tracking.")
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
            with col_btn1:
                if st.button("🧹 Clear All Points"):
                    st.session_state._shared.click_points_pos = []
                    st.session_state._shared.click_points_neg = []
                    st.session_state._shared.click_dirty = True
            with col_btn2:
                if st.button("📤 Apply Mask Now"):
                    st.session_state._shared.click_dirty = True
            # Render clickable canvas with current input frame
            frame_in = st.session_state._shared.frame_in
            if frame_in is not None:
                try:
                    import base64
                    import streamlit.components.v1 as components
                    frame_rgb = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
                    _, buf = cv2.imencode('.jpg', frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    b64 = base64.b64encode(buf).decode()
                    with open("scripts/click_canvas.html") as f:
                        html = f.read()
                    html = html.replace("{{IMG_B64}}", b64)
                    html = html.replace("{{API_KEY}}", "click_canvas")
                    html = html.replace("{{EXISTING_POS}}", __import__('json').dumps(st.session_state._shared.click_points_pos))
                    html = html.replace("{{EXISTING_NEG}}", __import__('json').dumps(st.session_state._shared.click_points_neg))
                    value = components.html(html, height=520, key="click_canvas")
                    if value and isinstance(value, str):
                        try:
                            data = __import__('json').loads(value)
                            new_pos = [tuple(p) for p in data.get("pos", [])]
                            new_neg = [tuple(p) for p in data.get("neg", [])]
                            if new_pos != st.session_state._shared.click_points_pos or new_neg != st.session_state._shared.click_points_neg:
                                st.session_state._shared.click_points_pos = new_pos
                                st.session_state._shared.click_points_neg = new_neg
                                st.session_state._shared.click_dirty = True
                        except Exception:
                            pass
                except Exception as canvas_err:
                    st.error(f"Canvas error: {canvas_err}")

    # ── Frame rendering loop ────────────────────────────────────────────────────
    if st.session_state._shared.running:
        limit = 300
        last_frame_id = -1
        for _ in range(limit):
            if not st.session_state._shared.running or st.session_state._shared.stop_event.is_set():
                break
            
            # Reduce WebSocket load by only sending images when a new frame is generated
            current_frame_id = st.session_state._shared.frame_id
            if current_frame_id != last_frame_id:
                inp = st.session_state._shared.frame_in
                out = st.session_state._shared.frame_out
                if inp is not None:
                    in_placeholder.image(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                if out is not None:
                    out_placeholder.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                last_frame_id = current_frame_id
                
            time.sleep(POLL_DELAY)
        st.rerun()
