import torch
import time
import cv2
import numpy as np
import os
import json
from safetensors.torch import load_file
from multiprocessing import Process, Value, Manager
from queue import Empty
from PIL import Image

# -- Device configuration for model offloading --
DEVICE_TRANSFORMER = "cuda"
DEVICE_VAE = "cuda:0"
DEVICE_TEXT_ENCODER = "cpu"

# -- Performance flags ---------------------------------------------------------
import torch as _torch_perf
_torch_perf.set_float32_matmul_precision("high")
_torch_perf.backends.cudnn.benchmark = True
_torch_perf.backends.cuda.matmul.allow_tf32 = True
_torch_perf.backends.cudnn.allow_tf32 = True
# -----------------------------------------------------------------------------

# -- DLL search path and TensorrtExecutionProvider monkey-patch -----------------------
try:
    import os as _os
    import sys as _sys
    if _sys.platform == "win32":
        _site_pkgs = _os.path.join(_sys.prefix, "Lib", "site-packages")
        _dll_dirs = [
            _os.path.join(_site_pkgs, "torch", "lib"),
            _os.path.join(_site_pkgs, "tensorrt_libs"),
        ]
        
        # Add to PATH and DLL search path
        _existing_path = _os.environ.get("PATH", "")
        _new_paths = []
        for _d in _dll_dirs:
            if _os.path.isdir(_d):
                _new_paths.append(_d)
                if hasattr(_os, "add_dll_directory"):
                    try:
                        _os.add_dll_directory(_d)
                    except Exception:
                        pass
        
        if _new_paths:
            _os.environ["PATH"] = _os.pathsep.join(_new_paths) + _os.pathsep + _existing_path

    import onnxruntime as _ort
    _orig_InferenceSession = _ort.InferenceSession
    
    class _PatchedInferenceSession(_orig_InferenceSession):
        def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
            _cache_dir = "models/trt_cache"
            if not _os.path.exists(_cache_dir) and _os.path.exists("app/models/trt_cache"):
                _cache_dir = "app/models/trt_cache"
            elif not _os.path.exists(_cache_dir):
                try:
                    _os.makedirs(_cache_dir, exist_ok=True)
                except Exception:
                    pass
            
            _available = _ort.get_available_providers()
            if "TensorrtExecutionProvider" in _available:
                _trt_options = {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": _cache_dir,
                    "trt_fp16_enable": True
                }
                if providers is None:
                    providers = [
                        ("TensorrtExecutionProvider", _trt_options),
                        ("CUDAExecutionProvider", {"device_id": 0}),
                        "CPUExecutionProvider"
                    ]
                else:
                    _new_providers = []
                    _has_trt = False
                    for _prov in providers:
                        _prov_name = _prov[0] if isinstance(_prov, tuple) else _prov
                        _prov_opts = _prov[1] if isinstance(_prov, tuple) else {}
                        
                        if _prov_name == "TensorrtExecutionProvider":
                            _has_trt = True
                            _merged_opts = _trt_options.copy()
                            _merged_opts.update(_prov_opts)
                            _new_providers.append((_prov_name, _merged_opts))
                        else:
                            _new_providers.append((_prov_name, _prov_opts))
                            
                    if not _has_trt:
                        _new_providers.insert(0, ("TensorrtExecutionProvider", _trt_options))
                    
                    providers = _new_providers
            
            super().__init__(path_or_bytes, sess_options=sess_options, providers=providers, provider_options=provider_options, **kwargs)
            try:
                _active = self.get_providers()
                print(f"[FluxRT] [TensorRT] Loaded ONNX session for {path_or_bytes}. Resolved execution providers: {_active}")
            except Exception as _e:
                print(f"[FluxRT] [WARNING] Failed to query resolved providers for {path_or_bytes}: {_e}")
    
    _ort.InferenceSession = _PatchedInferenceSession
    print("[FluxRT] Successfully monkey-patched ONNX Runtime for TensorRT execution provider support.")
except Exception as _ort_err:
    print(f"[FluxRT] ONNX Runtime monkey-patch failed: {_ort_err}")
# -------------------------------------------------------------------------------------

# -- triton_key monkey-patch ----------------------------------------------------
# PyTorch 2.7's inductor expects triton.compiler.compiler.triton_key() but
# triton-windows 3.6 ships get_cache_key() instead. Inject the shim BEFORE
# any torch.compile / inductor code path runs.
try:
    import triton.compiler.compiler as _triton_compiler

    if not hasattr(_triton_compiler, "triton_key"):
        import triton as _triton_mod

        _triton_compiler.triton_key = lambda: getattr(
            _triton_mod, "__version__", "unknown"
        )
        print(f"[FluxRT] Patched triton_key -> triton {_triton_compiler.triton_key()}")
except Exception as _e:
    print(f"[FluxRT] triton_key patch skipped: {_e}")
# -------------------------------------------------------------------------------

# -- Monkey-patch diffusers single_file_model for custom Flux2Transformer2DModel GGUF loader --
try:
    import diffusers.loaders.single_file_model as _sf_model
    _orig_get_mapping = _sf_model._get_single_file_loadable_mapping_class
    def _patched_get_mapping(cls):
        if cls.__name__ == "Flux2Transformer2DModel":
            return "Flux2Transformer2DModel"
        return _orig_get_mapping(cls)
    _sf_model._get_single_file_loadable_mapping_class = _patched_get_mapping
    print("[FluxRT] Successfully monkey-patched single_file_model mapping class for GGUF loading.")
except Exception as e:
    print(f"[FluxRT] single_file_model monkey-patch failed: {e}")
# ---------------------------------------------------------------------------------------------

# -- RMSNorm shape mismatch monkey-patch ----------------------------------------
try:
    import torch as _torch
    import torch.nn as _nn
    if hasattr(_nn, "RMSNorm"):
        _orig_rmsnorm_forward = _nn.RMSNorm.forward
        def _patched_rmsnorm_forward(self, input):
            weight = self.weight
            if weight is not None:
                norm_dim = self.normalized_shape[0]
                weight_dim = weight.shape[0]
                if weight_dim != norm_dim:
                    if weight_dim > norm_dim:
                        weight = weight[:norm_dim]
                    else:
                        repeats = (norm_dim + weight_dim - 1) // weight_dim
                        weight = weight.repeat(repeats)[:norm_dim]
            return _torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)
        _nn.RMSNorm.forward = _patched_rmsnorm_forward
        print("[FluxRT] Successfully monkey-patched RMSNorm to prevent GGUF dimension mismatches.")
except Exception as _e:
    print(f"[FluxRT] RMSNorm monkey-patch skipped: {_e}")
# -------------------------------------------------------------------------------

# -- CompiledKernel.launch_enter_hook monkey-patch --------------------------------
# Some PyTorch 2.7 + triton-windows builds are missing the launch_enter_hook
# attribute on CompiledKernel, which causes torch.compile to crash.
try:
    import torch._inductor as _inductor_mod
    import triton.compiler.compiler as _triton_cc
    if not hasattr(_inductor_mod.CompiledKernel, "launch_enter_hook"):
        _inductor_mod.CompiledKernel.launch_enter_hook = None
    if not hasattr(_triton_cc.CompiledKernel, "launch_enter_hook"):
        _triton_cc.CompiledKernel.launch_enter_hook = None
except Exception:
    pass
# -------------------------------------------------------------------------------

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

# -- FluxRT torch.compile wrapper -------------------------------------------------
# Gracefully fall back to eager mode when torch.compile is not supported (e.g.
# missing CompiledKernel.launch_enter_hook on triton-windows).
_orig_torch_compile = torch.compile
def _fluxrt_safe_compile(*args, **kwargs):
    try:
        return _orig_torch_compile(*args, **kwargs)
    except Exception as e:
        print(f"[FluxRT] torch.compile failed: {e}. Falling back to eager mode.")
        return args[0] if args else None
torch.compile = _fluxrt_safe_compile
# -------------------------------------------------------------------------------
from diffusers.models import AutoencoderKLFlux2
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM, AutoConfig
from accelerate import init_empty_weights

from fluxrt.stream_processor.interpolation_model import IFNet
from fluxrt.stream_processor.transformer_flux2 import Flux2Transformer2DModel
from fluxrt.utils.shared_tensor import SharedTensor
from fluxrt.stream_processor.pipeline import Flux2KleinPipeline
from fluxrt.stream_processor.update_controller import UpdateController
from fluxrt.stream_processor.postprocessors import (
    BasePostProcessor,
    LivePortraitPostProcessor,
    OutputEnhancer,
)


class ModelInferenceSubprocess:
    def __init__(
        self,
        config: dict,
        input_shared_tensor_name: str,
        output_batch_shared_tensor_name: str,
        pack_is_ready,
        last_processing_time,
    ):
        self.running = Value("b", False)
        self.memory_reserved = Value("i", 0)
        self.process = None
        self.config = config
        self.height = self.config["resolution"]["height"]
        self.width = self.config["resolution"]["width"]
        self.resolution = self.config["resolution"]
        self.prompt = self.config["default_prompt"]
        self.logging = self.config.get("logging", True)
        self.input_shared_tensor_name = input_shared_tensor_name
        self.output_batch_shared_tensor_name = output_batch_shared_tensor_name
        self.pack_is_ready = pack_is_ready
        self.last_processing_time = last_processing_time

        manager = Manager()
        self.command_queue = manager.Queue()
        self.shared_state = manager.dict()
        self.interpolation_exp = self.config.get("interpolation_exp", 1)

    def enable_quantization(self):
        """
        Should be called before the subprocess is started.
        """
        self.config["enable_int8_quantization"] = True

    def init_process_state(self):
        self.device = "cuda"
        self.process_state = {
            "prompt": self.config["default_prompt"],
            "steps": self.config["default_steps"],
            "seed": self.config["default_seed"],
        }

    def load_models_without_quantization(self):
        device = self.device
        dtype = torch.bfloat16

        models_path = self.config["models_path"]
        gguf_path = self.config.get(
            "gguf_model_path",
            "models/gguf/flux-2-klein-base-4b-Q6_K.gguf"
        )
        
        # Check relative to current working directory and resolve absolute paths
        if not os.path.exists(gguf_path):
            if os.path.exists(os.path.abspath(gguf_path)):
                gguf_path = os.path.abspath(gguf_path)
            elif os.path.exists(os.path.abspath(f"app/{gguf_path}")):
                gguf_path = os.path.abspath(f"app/{gguf_path}")

        # Fallback to scanning the directory for any .gguf file
        if not os.path.exists(gguf_path):
            gguf_dir = "models/gguf"
            if not os.path.exists(gguf_dir) and os.path.exists("app/models/gguf"):
                gguf_dir = "app/models/gguf"
            
            if os.path.exists(gguf_dir):
                files = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")]
                if files:
                    selected_file = files[0]
                    gguf_path = os.path.abspath(os.path.join(gguf_dir, selected_file))
                    print(f"[Subprocess] Configured GGUF model path not found. Automatically fell back to: {gguf_path}")
        
        # Final validation to raise a clean error
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(
                f"No GGUF model file found. Please place a .gguf model in models/gguf/ directory."
            )

        print(f"[Subprocess] Loading scheduler, VAE, text_encoder from: {models_path}...")
        
        self.scheduler_config = FlowMatchEulerDiscreteScheduler.load_config(
            f"{models_path}/scheduler", local_files_only=True
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler_config)
        
        print(f"[Subprocess] Loading AutoencoderKLFlux2 (VAE) on {DEVICE_VAE}...")
        self.vae = AutoencoderKLFlux2.from_pretrained(
            f"{models_path}/vae", local_files_only=True, torch_dtype=dtype
        )
        self.vae = self.vae.to(DEVICE_VAE, dtype=dtype)
        # TRT VAE disabled: engine has fixed 320x576 shape and channel mismatch (64 vs 32)
        # Use PyTorch eager mode for VAE
        if not hasattr(self.vae, "_orig_encode"):
            self.vae._orig_encode = self.vae.encode
        if not hasattr(self.vae, "_orig_decode"):
            self.vae._orig_decode = self.vae.decode
        print("[FluxRT] VAE running in PyTorch eager mode.")
        
        print(f"[Subprocess] Loading Qwen3ForCausalLM (text encoder) on {DEVICE_TEXT_ENCODER}...")
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            f"{models_path}/text_encoder", local_files_only=True, torch_dtype=dtype
        )
        self.text_encoder = self.text_encoder.to(DEVICE_TEXT_ENCODER, dtype=dtype)
        
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            f"{models_path}/tokenizer", local_files_only=True
        )
        
        print(f"[Subprocess] Loading Flux2Transformer2DModel from GGUF: {gguf_path}")
        config_path = os.path.abspath(f"{models_path}/transformer")
        if not os.path.exists(config_path) and os.path.exists(os.path.abspath(f"app/{models_path}/transformer")):
            config_path = os.path.abspath(f"app/{models_path}/transformer")
        from diffusers.quantizers.quantization_config import GGUFQuantizationConfig
        self.transformer = Flux2Transformer2DModel.from_single_file(
            gguf_path,
            config=config_path,
            subfolder="",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            local_files_only=True
        )
        
        # Dequantize all GGUF Q4_0 parameters to plain BF16 tensors so the transformer
        # runs with standard matmuls. On Windows there are no CUDA GGUF kernels, so keeping
        # the model quantized causes on-the-fly dequantization via a slow Python fallback
        # that produces incorrect results. Dequantizing once at load time is both faster
        # and numerically correct.
        _dequant_count = 0
        _linear_count = 0
        try:
            from diffusers.quantizers.gguf.utils import GGUFParameter, GGUFLinear, dequantize_gguf_tensor
            import torch.nn as _nn_mod
            # Phase 1: Dequant all GGUFParameters (Q4_0, BF16, etc.) to plain BF16 tensors
            for _name, _param in list(self.transformer.named_parameters()):
                if isinstance(_param, GGUFParameter):
                    _dq = dequantize_gguf_tensor(_param)
                    if hasattr(_dq, "as_tensor"):
                        _dq = _dq.as_tensor()
                    _dq = _dq.to(device=_param.device, dtype=torch.bfloat16)
                    _parts = _name.split(".")
                    _parent = self.transformer
                    for _part in _parts[:-1]:
                        _parent = getattr(_parent, _part)
                    setattr(_parent, _parts[-1], torch.nn.Parameter(_dq, requires_grad=False))
                    _dequant_count += 1
            # Phase 2: Replace GGUFLinear modules with standard nn.Linear
            for _name, _mod in list(self.transformer.named_modules()):
                if isinstance(_mod, GGUFLinear):
                    _w = _mod.weight
                    if hasattr(_w, "as_tensor"):
                        _w = _w.as_tensor()
                    _w = _w.to(torch.bfloat16)
                    _new_linear = _nn_mod.Linear(
                        _mod.in_features, _mod.out_features,
                        bias=_mod.bias is not None,
                        device=_w.device,
                        dtype=torch.bfloat16,
                    )
                    with torch.no_grad():
                        _new_linear.weight.copy_(_w)
                        if _mod.bias is not None:
                            _new_linear.bias.copy_(_mod.bias.to(torch.bfloat16))
                    _parts = _name.split(".")
                    _parent = self.transformer
                    for _part in _parts[:-1]:
                        _parent = getattr(_parent, _part)
                    setattr(_parent, _parts[-1], _new_linear)
                    _linear_count += 1
        except Exception as _dq_err:
            print(f"[FluxRT] GGUF dequant/linear replacement skipped: {_dq_err}")
        # Cast any remaining float32 params (norms, biases) to BF16
        for _name, _param in list(self.transformer.named_parameters()):
            if _param.dtype == torch.float32:
                _param.data = _param.data.to(torch.bfloat16)
                _dequant_count += 1
        print(f"[FluxRT] Dequantized {_dequant_count} GGUF params, replaced {_linear_count} GGUFLinear → nn.Linear (BF16).")

        self.transformer.to(DEVICE_TRANSFORMER)
        print(f"[Subprocess] Transformer on {DEVICE_TRANSFORMER}, VAE on {DEVICE_VAE}, text_encoder on {DEVICE_TEXT_ENCODER}")

    def load_quantized_models(self):
        from optimum.quanto import requantize
        from fluxrt.stream_processor.quantized_flux2 import (
            QuantizedFlux2Transformer2DModel,
        )

        device = self.device
        dtype = torch.bfloat16

        models_path = self.config["models_path"]
        int8_models_path = self.config["int8_models_path"]

        qtransformer = QuantizedFlux2Transformer2DModel.from_pretrained(
            int8_models_path, local_files_only=True
        )
        qtransformer.to(device=device, dtype=dtype)
        self.transformer = qtransformer._wrapped

        self.scheduler_config = FlowMatchEulerDiscreteScheduler.load_config(
            f"{models_path}/scheduler", local_files_only=True
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler_config)
        self.vae = AutoencoderKLFlux2.from_pretrained(
            f"{models_path}/vae", local_files_only=True
        ).to(device, dtype)

        config = AutoConfig.from_pretrained(
            f"{int8_models_path}/text_encoder", local_files_only=True
        )
        with init_empty_weights():
            text_encoder = Qwen3ForCausalLM(config)

        with open(f"{int8_models_path}/text_encoder/quanto_qmap.json", "r") as f:
            qmap = json.load(f)
        state_dict = load_file(f"{int8_models_path}/text_encoder/model.safetensors")
        requantize(text_encoder, state_dict=state_dict, quantization_map=qmap)
        text_encoder.eval()
        text_encoder.to(device, dtype=dtype)
        self.text_encoder = text_encoder

        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            f"{int8_models_path}/tokenizer", local_files_only=True
        )

    def load_models(self):
        self.interpolation_model = IFNet()
        self.interpolation_model.load_state_dict(
            load_file("RIFE-safetensors/flownet.safetensors")
        )
        self.interpolation_model.to("cuda", dtype=torch.float16)
        self.interpolation_model.eval()

        if self.config.get("enable_int8_quantization", False):
            self.load_quantized_models()
        else:
            self.load_models_without_quantization()

        if self.config.get("compile_models", False):
            self.transformer = torch.compile(
                self.transformer,
            )
            self.vae = torch.compile(
                self.vae,
            )
            self.interpolation_model = torch.compile(
                self.interpolation_model,
            )

        reference_image_seq_len = None
        if self.config["use_reference_image"]:
            reference_image_res = self.config["reference_image_resolution"]
            reference_image_seq_len = (reference_image_res["width"] // 16) * (
                reference_image_res["height"] // 16
            )

        self.update_controller = UpdateController(
            self.config,
            self.height,
            self.width,
            compression_ratio=16,
            text_seq_len=self.config.get("max_sequence_length", 512),
            reference_image_seq_len=reference_image_seq_len,
            reset_period=self.config.get("mask_reset_period", None),
        )

        self.pipe = Flux2KleinPipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            update_controller=self.update_controller,
            subprocess_config=self.config,
        )
        self.pipe.to(self.device)
        self.text_encoder = self.text_encoder.to(DEVICE_TEXT_ENCODER)
        self.vae = self.vae.to(DEVICE_VAE)

        if self.config.get("use_lora", False):
            self.pipe.load_lora_weights(self.config.get("lora_weights_path", ""))

        self.lip_processor: BasePostProcessor | None = None
        self.lip_active = False
        lp_cfg = self.config.get("lip_transfer", {})
        if lp_cfg.get("enable", False):
            self.lip_processor = LivePortraitPostProcessor(
                models_dir=lp_cfg["models_dir"]
            )

        self.output_enhancer = OutputEnhancer(self.config)

    def recreate_scheduler(self):
        sampler_name = self.process_state.get("sampler", "Euler")
        from diffusers import (
            FlowMatchEulerDiscreteScheduler,
            FlowMatchHeunDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        )
        if sampler_name == "Heun":
            self.scheduler = FlowMatchHeunDiscreteScheduler.from_config(self.scheduler_config)
        elif sampler_name == "Euler Ancestral":
            self.scheduler = EulerAncestralDiscreteScheduler.from_config(self.scheduler_config)
        elif sampler_name == "DPM++ 2M":
            self.scheduler = DPMSolverMultistepScheduler.from_config(self.scheduler_config)
        else:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler_config)
        if hasattr(self, "pipe") and self.pipe is not None:
            self.pipe.scheduler = self.scheduler

    def update_negative_prompt_embeds(self, negative_prompt: str):
        try:
            self.negative_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=negative_prompt,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=self.config.get("max_sequence_length", 512),
                text_encoder_out_layers=(9, 18, 27),
            )
        except Exception as e:
            print(f"[FluxRT] Failed to encode negative prompt: {e}")
            self.negative_prompt_embeds = None

    def update_prompt_embeds(self, prompt):
        self.prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=self.config.get("max_sequence_length", 512),
            text_encoder_out_layers=(9, 18, 27),
        )
        self.update_controller.reset_cache()
        # Clear the pipeline's spatial_cache so it is rebuilt with the correct
        # dimensions on the next frame (prompt length change can alter text_seq_len).
        if hasattr(self, "pipe") and self.pipe is not None:
            self.pipe.spatial_cache = {}

    def init_shared_tensors(self):
        h, w = self.resolution["height"], self.resolution["width"]

        self.input_shared_tensor = SharedTensor(
            (h, w, 3),
            name=self.input_shared_tensor_name,
        )

        # All interpolated then one original
        output_batch_size = 2**self.interpolation_exp
        self.output_batch_shared_tensor = SharedTensor(
            (output_batch_size, h, w, 3),
            name=self.output_batch_shared_tensor_name,
        )

    def process_init(self):
        """
        Initializes all resources required by the inference subprocess.
        """
        self.init_process_state()
        self.init_shared_tensors()
        self.load_models()
        self.update_prompt_embeds(self.process_state["prompt"])
        self.negative_prompt_embeds = None
        self.previous_frame = None

        if self.config.get("use_reference_image", False):
            path = self.config.get("reference_image_path", "")
            image = None
            if path and os.path.exists(path):
                image = cv2.imread(path)

            resolution = self.config.get("reference_image_resolution")
            if image is None:
                image = np.zeros(
                    (resolution["height"], resolution["width"], 3), dtype=np.uint8
                )
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (resolution["width"], resolution["height"]))
            self.reference_image = Image.fromarray(image)

        target_fps = self.config.get("target_fps", None)
        self.target_base_processing_time = None
        if target_fps is not None:
            target_base_fps = target_fps / (2**self.interpolation_exp)
            self.target_base_processing_time = 1 / target_base_fps

        # Segmentation & enhancer defaults
        self.segmentation_enabled = False
        self.adaptive_interp = False
        self.segmenter = None
        self.compositor = None
        self.protect_classes = ["face", "hair", "body"]

    def _init_segmenter(self):
        try:
            from fluxrt.stream_processor.postprocessors.segmenter import BackgroundCompositor
            self.compositor = BackgroundCompositor()
            print("[Subprocess] BackgroundCompositor with multi-class segmenter initialized.")
        except Exception as e:
            print(f"[Subprocess] BackgroundCompositor unavailable: {e}")
            self.compositor = None

    def start(self):
        self.running.value = True
        self.process = Process(target=self.process_main)
        self.process.start()

    def stop(self):
        self.running.value = False
        if self.process:
            self.process.join()

    def set_param(self, name: str, value) -> None:
        self.command_queue.put(("set_param", (name, value)))

    def set_reference_image(self, image: np.ndarray | None) -> None:
        """
        Update the reference image on the fly.
        image: numpy uint8 RGB array
        Only valid when use_reference_image is true in config.
        """
        if not self.config.get("use_reference_image", False):
            raise ValueError(
                "set_reference_image called but use_reference_image is not enabled in the stream processor config"
            )
        self.command_queue.put(("set_reference_image", image))

    def set_mask(self, mask) -> None:
        """
        Update the mask on the fly.
        mask: numpy uint8 array of shape (h // compression_ratio, w // compression_ratio).
        Only valid when mask_calculation_method is set to manual in config.
        """
        if self.config.get("mask_calculation_method", "auto") != "manual":
            raise ValueError(
                "set_mask called but mask_calculation_method is not set to manual in the config"
            )
        self.command_queue.put(("set_mask", mask))

    def set_lip_transfer(self, enabled: bool) -> None:
        self.command_queue.put(("set_lip_transfer", enabled))

    def set_gguf_model(self, gguf_path: str) -> None:
        self.command_queue.put(("set_gguf_model", gguf_path))

    def update_process_state(self) -> None:
        """
        Called by the internal process
        """
        try:
            while True:
                cmd, payload = self.command_queue.get_nowait()
                if cmd == "set_param":
                    name, value = payload
                    self.process_state[name] = value
                    if name == "prompt":
                        self.update_prompt_embeds(value)
                    elif name == "negative_prompt":
                        self.update_negative_prompt_embeds(value)
                    elif name == "sampler":
                        self.recreate_scheduler()
                        self.update_controller.reset_cache()
                    elif name in ["steps", "seed", "cfg_scale", "distilled_mode", "denoise", "noise_offset", "guidance_rescale", "sigma_min", "sigma_max", "eta", "temporal_smoothing", "vae_decode_scaling"]:
                        self.update_controller.reset_cache()
                    elif name == "segmentation_enabled":
                        self.segmentation_enabled = value
                        if value and self.compositor is None:
                            self._init_segmenter()
                    elif name == "protect_classes":
                        self.protect_classes = value
                        if self.compositor is not None:
                            self.compositor.set_protect_classes(value)
                    elif name == "click_points":
                        if self.compositor is not None:
                            try:
                                data = __import__('json').loads(value)
                                self.compositor.clear_clicks()
                                for x, y in data.get("pos", []):
                                    self.compositor.add_click_point(int(x), int(y), True)
                                for x, y in data.get("neg", []):
                                    self.compositor.add_click_point(int(x), int(y), False)
                            except Exception:
                                pass
                    elif name == "clear_clicks":
                        if self.compositor is not None:
                            self.compositor.clear_clicks()
                    elif name == "use_click_mask":
                        if self.compositor is not None:
                            self.compositor.use_clicks = bool(value)
                    elif name == "adaptive_interp":
                        self.adaptive_interp = value
                    elif name.startswith("enhancer_"):
                        if hasattr(self, 'output_enhancer') and self.output_enhancer is not None:
                            attr = name[len("enhancer_"):]
                            setattr(self.output_enhancer, attr, value)
                            if attr == "sharpen_strength":
                                self.output_enhancer.enabled = any([
                                    self.output_enhancer.sharpen_strength > 0,
                                    self.output_enhancer.deblock_strength > 0,
                                    self.output_enhancer.denoise_strength > 0,
                                    self.output_enhancer.temporal_strength > 0,
                                    self.output_enhancer.contrast_clip > 0,
                                    self.output_enhancer.upscale_factor > 1,
                                ])
                elif cmd == "reset_cache":
                    # reset_cache command handler
                    self.update_controller.reset_cache()
                elif cmd == "set_reference_image":
                    image = payload  # numpy uint8 RGB array or None
                    resolution = self.config["reference_image_resolution"]
                    if image is not None:
                        image = cv2.resize(
                            image, (resolution["width"], resolution["height"])
                        )
                        self.reference_image = Image.fromarray(image)
                    else:
                        self.reference_image = Image.fromarray(
                            np.zeros(
                                (resolution["height"], resolution["width"], 3),
                                dtype=np.uint8,
                            )
                        )
                    self.update_controller.reset_cache()

                elif cmd == "set_mask":
                    mask = payload  # numpy uint8 array of shape (h // compression_ratio, w // compression_ratio)
                    mask_tensor = (
                        torch.from_numpy(mask)
                        .unsqueeze(0)
                        .to(self.update_controller.device)
                    )
                    self.update_controller.set_mask(mask_tensor)

                elif cmd == "set_lip_transfer":
                    self.lip_active = payload
                    if self.lip_active and self.lip_processor is None:
                        try:
                            lp_cfg = self.config.get("lip_transfer", {})
                            models_dir = lp_cfg.get("models_dir", "LivePortrait-code/pretrained_models")
                            print(f"[Subprocess] [TensorRT] Dynamically initializing LivePortraitPostProcessor from {models_dir}...")
                            self.lip_processor = LivePortraitPostProcessor(models_dir=models_dir)
                            print("[Subprocess] [TensorRT] Successfully initialized LivePortraitPostProcessor.")
                        except Exception as e:
                            print(f"[Subprocess] [ERROR] Failed to initialize LivePortrait: {e}")
                            self.lip_active = False
                    
                elif cmd == "set_gguf_model":
                    gguf_path = payload
                    print(f"[DEBUG] Loading GGUF model from {gguf_path}")
                    if hasattr(self, 'transformer'):
                        del self.transformer
                    torch.cuda.empty_cache()
                    self.transformer = Flux2Transformer2DModel.from_single_file(gguf_path)
                    self.transformer.to(self.device, torch.bfloat16)
                    self.pipe.transformer = self.transformer
                    self.update_controller.reset_cache()
                    print(f"[DEBUG] Successfully loaded GGUF model from {gguf_path}")

        except Empty:
            pass

    def receive_frame(self):
        """
        Reads frame from input shared memory, converts to RGB float16 GPU tensors.
        """
        frame = self.input_shared_tensor.to_numpy()
        frame_gpu = (
            torch.from_numpy(frame)
            .to(self.device)
            .to(torch.float16)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255)
        )
        return frame_gpu

    def interpolate_frames(self, frame):
        """
        Takes one new generated frame (torch tensor, RGB, on GPU, float16)
        Interpolates according to interpolation_exp times.
        Batches to [interpolated frames, new frame].
        """
        if self.previous_frame is None:
            self.previous_frame = frame

        if self.interpolation_exp == 0:
            frames_out = frame
        else:
            frames = torch.cat([self.previous_frame, frame], dim=0)
            with torch.no_grad():
                for _ in range(self.interpolation_exp):
                    B = frames.size(0)
                    prevs = frames[:-1]
                    nexts = frames[1:]
                    mids = self.interpolation_model(torch.cat([prevs, nexts], dim=1))
                    H, W = frames.shape[2:]
                    new_frames = torch.empty(
                        2 * B - 1, 3, H, W, device=frames.device, dtype=frames.dtype
                    )
                    new_frames[0::2] = frames
                    new_frames[1::2] = mids
                    frames = new_frames
            frames_out = frames[1:]

        frames_cpu = (
            frames_out.mul(255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )

        self.previous_frame = frame

        return frames_cpu[..., ::-1]

    def send_frames(self, frames):
        self.output_batch_shared_tensor.copy_from(frames)

    def sync_fps_and_send(self, prev_time, frames):
        now = time.time()
        processing_time = now - prev_time

        if self.target_base_processing_time is not None:
            sleep_time = max(0, self.target_base_processing_time - processing_time)
            time.sleep(sleep_time)
            now = time.time()

        processing_time = now - prev_time

        self.last_processing_time.value = processing_time
        self.send_frames(frames)
        self.pack_is_ready.value = True
        self.memory_reserved.value = torch.cuda.memory_reserved() // (1024 * 1024)

        if self.logging:
            print(
                f"base fps: {(1 / processing_time):.2f}, interpolated fps: {(1 / processing_time * 2**self.interpolation_exp):.2f}"
            )
        return now

    def process_frame_with_pipeline(self, frame):
        """
        Takes frame as np uint8 RGB array
        Returns frame as np uint8 RGB array
        """
        input_frame = Image.fromarray(frame)

        reference_list = [input_frame]
        if self.config["use_reference_image"]:
            reference_list.append(self.reference_image)

        start_t = time.time()
        steps = self.process_state["steps"]
        sigma_min = self.process_state.get("sigma_min", 0.0)
        sigma_max = self.process_state.get("sigma_max", 1.0)
        # FLUX.2 uses a flow-matching scheduler whose sigmas must stay within
        # (0, 1] (1.0 == pure noise, 0 == clean). When the sliders are at their
        # full-range default, defer to the scheduler's native flow schedule
        # (sigmas=None) — this is the proven default. Only build a custom schedule
        # for partial denoising, and keep it inside the valid range.
        if sigma_max >= 1.0 and sigma_min <= 0.0:
            sigmas = None
        else:
            sigma_max = min(sigma_max, 1.0)
            sigma_min = max(sigma_min, 0.0)
            sigmas = np.linspace(sigma_max, sigma_min, steps).tolist()

        cfg_scale = self.process_state.get("cfg_scale", 4.5)
        if self.process_state.get("distilled_mode", True):
            cfg_scale = 1.0
        noise_offset = self.process_state.get("noise_offset", 0.0)
        guidance_rescale = self.process_state.get("guidance_rescale", 0.0)
        eta = self.process_state.get("eta", 0.0)
        temporal_smoothing = self.process_state.get("temporal_smoothing", 0.0)
        vae_decode_scaling = self.process_state.get("vae_decode_scaling", 1.0)
        denoise = self.process_state.get("denoise", 0.55)

        out = self.pipe(
            prompt_embeds=self.prompt_embeds,
            negative_prompt_embeds=self.negative_prompt_embeds,
            image=reference_list,
            height=self.resolution["height"],
            width=self.resolution["width"],
            guidance_scale=cfg_scale,
            num_inference_steps=steps,
            sigmas=sigmas,
            num_images_per_prompt=1,
            generator=torch.Generator(device=self.device).manual_seed(
                self.process_state["seed"]
            ),
            output_type="np",
            noise_offset=noise_offset,
            guidance_rescale=guidance_rescale,
            eta=eta,
            temporal_smoothing=temporal_smoothing,
            vae_decode_scaling=vae_decode_scaling,
            denoise=denoise,
        )
        end_t = time.time()
        out_image = out.images[0]
        out_image = out_image * 255
        out_image = out_image.astype(np.uint8)
        print(f"[DEBUG] FLUX.2 generated frame in {end_t - start_t:.2f}s. Image mean pixel value: {out_image.mean():.2f}")
        return out_image

    def convert_np_to_torch(self, frame):
        frame = (
            torch.from_numpy(frame)
            .to(self.device)
            .to(torch.float16)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255)
        )
        return frame

    def process_main(self):
        self.process_init()
        prev_time = time.time()
        while self.running.value:
            self.update_process_state()
            original_frame = self.input_shared_tensor.to_numpy()
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            frame = self.process_frame_with_pipeline(original_frame)
            if self.lip_processor is not None and self.lip_active:
                # Note: we are getting the latest input frame again after flux processing to reduce latency.
                original_frame = self.input_shared_tensor.to_numpy()
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                frame = self.lip_processor.process(frame, original_frame)
            # Output enhancement: sharpen, denoise, deblock, temporal smooth, optional upscale
            if hasattr(self, 'output_enhancer') and self.output_enhancer.enabled:
                frame = self.output_enhancer.process(frame)
            # Multi-class background composite (person-preserving via click / auto-segmentation)
            if self.segmentation_enabled and self.compositor is not None:
                try:
                    composite_in = self.input_shared_tensor.to_numpy()
                    composite_in = cv2.cvtColor(composite_in, cv2.COLOR_BGR2RGB)
                    frame = self.compositor.composite(composite_in, frame)
                except Exception as seg_err:
                    print(f"[Subprocess] Background composite failed: {seg_err}")
            frame = self.convert_np_to_torch(frame)
            frames = self.interpolate_frames(frame)
            prev_time = self.sync_fps_and_send(prev_time, frames)
