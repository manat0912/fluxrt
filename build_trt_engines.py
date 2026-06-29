"""
Export VAE encoder/decoder to ONNX for TensorRT via ONNX Runtime.
Run once: `python build_trt_engines.py`
"""
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

MODELS_PATH = "FLUX.2-klein-4B"
TRT_CACHE = "models/trt_cache"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

os.makedirs(TRT_CACHE, exist_ok=True)

from diffusers.models import AutoencoderKLFlux2

print("Loading VAE...")
vae = AutoencoderKLFlux2.from_pretrained(
    f"{MODELS_PATH}/vae", local_files_only=True, torch_dtype=DTYPE
)
vae = vae.to(DEVICE, dtype=DTYPE)
vae.eval()

# ── Encoder ─────────────────────────────────────────────
# The raw encoder outputs a 64-channel latent parameter vector.
# The 64 channels are split into mean (first 32) and logvar (last 32).
print("Exporting encoder to ONNX...")
encoder_in = torch.randn(1, 3, 320, 576, device=DEVICE, dtype=DTYPE)

class EncoderWrapper(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
    def forward(self, x):
        return self.enc(x)  # returns (B, 64, H//8, W//8) latent params

encoder_wrapper = EncoderWrapper(vae.encoder).eval()

torch.onnx.export(
    encoder_wrapper,
    encoder_in,
    f"{TRT_CACHE}/vae_encoder.onnx",
    input_names=["image"],
    output_names=["latent_params"],
    opset_version=17,
)
print("Encoder ONNX saved.")

# Verify
with torch.no_grad():
    latent_params_ref = encoder_wrapper(encoder_in)
import onnx
onnx_model = onnx.load(f"{TRT_CACHE}/vae_encoder.onnx")
onnx.checker.check_model(onnx_model)
print(f"Encoder ONNX verified. Latent params shape: {latent_params_ref.shape}")

# ── Decoder ─────────────────────────────────────────────
print("Exporting decoder to ONNX...")
decoder_in = torch.randn(1, 32, 40, 72, device=DEVICE, dtype=DTYPE)

class DecoderWrapper(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec
    def forward(self, x):
        return self.dec(x)

decoder_wrapper = DecoderWrapper(vae.decoder).eval()

torch.onnx.export(
    decoder_wrapper,
    decoder_in,
    f"{TRT_CACHE}/vae_decoder.onnx",
    input_names=["latents"],
    output_names=["image"],
    opset_version=17,
)
print("Decoder ONNX saved.")

# Verify
onnx_model = onnx.load(f"{TRT_CACHE}/vae_decoder.onnx")
onnx.checker.check_model(onnx_model)
print(f"Decoder ONNX verified.")

# ── Build TRT engines via TensorRT Python API ──────────
print("Building TensorRT engines (this may take a few minutes)...")
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def build_engine(onnx_path, engine_name, builder, precision="fp16"):
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(err)}")
            raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1 << 30))  # 2GB
    
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"  FP16 enabled for {engine_name}")
    
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Failed to build engine for {engine_name}")
    
    engine_path = f"{TRT_CACHE}/{engine_name}.engine"
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  Engine saved: {engine_path}")
    return engine_path

build_engine(f"{TRT_CACHE}/vae_encoder.onnx", "vae_encoder", builder)
build_engine(f"{TRT_CACHE}/vae_decoder.onnx", "vae_decoder", builder)

print(f"\nDone! TRT engines saved to {TRT_CACHE}/")
print("Files:")
for f in os.listdir(TRT_CACHE):
    print(f"  {f}")
