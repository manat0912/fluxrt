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

import argparse
import sys
import threading
import time
import os
import psutil
from PIL import Image
import cv2
import numpy as np
import gradio as gr
import torch

from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle

default_prompt = "Turn this image into cyberpunk night, red and blue neon lamps, bokeh"

stream_processor = None
input_tensor = None
output_tensor = None
resolution = None
use_int8 = False

processor_lock = threading.Lock()

# Video & Image State
current_video_id = 0
current_video_id_lock = threading.Lock()
local_current_frame = None
local_processed_frame = None
local_frame_lock = threading.Lock()

# Webcam State
is_playing = False

def get_processor():
    global stream_processor, input_tensor, output_tensor, resolution
    if stream_processor is None:
        stream_processor = StreamProcessor("configs/config_with_reference.json")
        if use_int8:
            stream_processor.enable_quantization()
        stream_processor.start()
        stream_processor.set_prompt(default_prompt)

        input_tensor = stream_processor.get_input_tensor()
        output_tensor = stream_processor.get_output_tensor()
        resolution = stream_processor.get_resolution()

    return stream_processor, input_tensor, output_tensor, resolution

def to_bgr(frame):
    if frame is None: return None
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def to_rgb(frame):
    if frame is None: return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def process_frame(frame):
    _, input_tensor, output_tensor, resolution = get_processor()
    frame = crop_maximal_rectangle(frame, resolution["height"], resolution["width"])
    with processor_lock:
        input_tensor.copy_from(frame)
        processed = output_tensor.to_numpy()
    return frame, processed

def set_prompt(prompt: str):
    sp, _, _, _ = get_processor()
    sp.set_prompt(prompt)

def set_reference_image_ui(image):
    sp, _, _, _ = get_processor()
    if image is not None:
        sp.config["use_reference_image"] = True
    sp.set_reference_image(image)

def set_lip_transfer_ui(enabled: bool):
    sp, _, _, _ = get_processor()
    if hasattr(sp, 'set_lip_transfer'):
        sp.set_lip_transfer(enabled)

def toggle_play():
    global is_playing
    is_playing = not is_playing
    return gr.update(value="Stop" if is_playing else "Start Animation", variant="stop" if is_playing else "primary")

def get_vram_recommendation():
    if not torch.cuda.is_available():
        return "⚠️ **CUDA is not available! Running on CPU will be extremely slow. Distilled or GGUF Q4_0 models are highly recommended.**"
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    device_name = torch.cuda.get_device_name(0)
    rec = f"🖥️ **GPU Detected:** {device_name} ({total_vram:.1f} GB VRAM)\n\n"
    if total_vram <= 6.5:
        rec += "⚠️ **Low VRAM detected (< 6.5 GB).** Quantized models (GGUF Q4_0 or int8 quantized) are **required** to prevent Out of Memory (OOM) errors. Please do NOT load full-precision models."
    elif total_vram <= 8.5:
        rec += "💡 **Medium VRAM detected (6.5 - 8.5 GB).** Quantized GGUF models (Q4_0 or Q8_0) or int8 quantized models are highly recommended for fast real-time inference."
    else:
        rec += "✅ **High VRAM detected (> 8.5 GB).** You can successfully run Full‑precision FluxRT, GGUF, or distilled models at peak performance."
    return rec

def load_variant_ui(variant, progress=gr.Progress()):
    global stream_processor, input_tensor, output_tensor, resolution, use_int8
    sp, _, _, _ = get_processor()
    
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if variant == "Full-precision FluxRT" and total_vram <= 6.5:
            return f"❌ **Error: Loading Full-precision FluxRT requires > 6.5 GB VRAM (You have {total_vram:.1f} GB). Loading prevented to avoid Out of Memory crash. Please select a quantized or GGUF model instead.**"
            
    progress(0.2, desc="Preparing model unloading...")
    try:
        if variant == "Full-precision FluxRT":
            progress(0.4, desc="Unloading existing model...")
            sp.config["enable_int8_quantization"] = False
            use_int8 = False
            progress(0.6, desc="Reloading Stream Processor in Full-precision mode...")
            sp.stop()
            stream_processor = None
            get_processor()
            return "Successfully loaded Full-precision FluxRT model!"
            
        elif variant == "Quantized (int8)":
            progress(0.4, desc="Enabling int8 mode...")
            sp.config["enable_int8_quantization"] = True
            use_int8 = True
            progress(0.6, desc="Reloading Stream Processor in int8 quantized mode...")
            sp.stop()
            stream_processor = None
            get_processor()
            return "Successfully loaded int8 Quantized model!"
            
        return f"Unknown variant: {variant}"
    except Exception as e:
        return f"❌ Error loading variant: {e}"

def download_and_apply_gguf(repo_id, gguf_filename, progress=gr.Progress()):
    if not gguf_filename:
        return "Please select a GGUF model to download.", gr.update()
    if not repo_id:
        return "Please enter a HuggingFace Repository ID.", gr.update()
        
    progress(0, desc="Starting download...")
    try:
        from huggingface_hub import hf_hub_download
        import os
        
        local_dir = "models/gguf"
        os.makedirs(local_dir, exist_ok=True)
        
        progress(0.1, desc=f"Downloading {gguf_filename} from {repo_id}...")
        
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=gguf_filename,
            local_dir=local_dir,
            token=False
        )
        
        progress(0.8, desc="Applying GGUF model...")
        sp, _, _, _ = get_processor()
        sp.set_gguf_model(file_path)
        progress(1.0, desc="Done!")
        
        # Refresh local models dropdown
        models = list_local_gguf_models()
        local_update = gr.update(choices=models, value=gguf_filename)
        return f"Successfully downloaded and applied GGUF model: {gguf_filename}", local_update
    except Exception as e:
        return f"❌ Error downloading/applying model: {e}", gr.update()

def list_local_gguf_models():
    import os
    local_dir = "models/gguf"
    if not os.path.exists(local_dir):
        return ["No local GGUF models found"]
    files = [f for f in os.listdir(local_dir) if f.endswith(".gguf")]
    if not files:
        return ["No local GGUF models found"]
    return sorted(files)

def load_local_gguf(filename, progress=gr.Progress()):
    if filename == "No local GGUF models found" or not filename:
        return "❌ Error: Please download a GGUF model first."
    progress(0.2, desc="Checking local GGUF path...")
    try:
        import os
        filepath = os.path.join("models/gguf", filename)
        if not os.path.exists(filepath):
            return f"❌ Error: Local file {filepath} not found."
            
        progress(0.5, desc=f"Loading local model {filename}...")
        sp, _, _, _ = get_processor()
        sp.set_gguf_model(filepath)
        progress(1.0, desc="Loaded successfully!")
        return f"Successfully loaded local GGUF model: {filename}"
    except Exception as e:
        return f"❌ Error loading model: {e}"

def refresh_local_gguf_list():
    models = list_local_gguf_models()
    return gr.update(choices=models, value=models[0] if models else None)

def process_webcam(frame):
    if frame is None:
        return frame
    _, processed = process_frame(to_bgr(frame))
    return to_rgb(processed)

def _image_to_video_loop(image: np.ndarray, video_id: int):
    global local_current_frame, local_processed_frame
    try:
        while True:
            with current_video_id_lock:
                if current_video_id != video_id:
                    break
            start = time.time()
            input_frame, processed = process_frame(to_bgr(image))
            with local_frame_lock:
                local_current_frame = to_rgb(input_frame)
                local_processed_frame = to_rgb(processed)
            time.sleep(max(0, 0.04 - (time.time() - start)))
    except Exception as e:
        print(f"Image loop error: {e}")

def start_image_to_video(image: np.ndarray | None):
    global current_video_id, local_current_frame, local_processed_frame
    with current_video_id_lock:
        current_video_id += 1
        my_id = current_video_id
    with local_frame_lock:
        local_current_frame = None
        local_processed_frame = None
    if image is None: return
    t = threading.Thread(target=_image_to_video_loop, args=(image, my_id), daemon=True)
    t.start()

def poll_local_video():
    with local_frame_lock:
        return local_current_frame, local_processed_frame

def switch_mode(mode: str):
    global current_video_id, is_playing
    
    # Stop video processing loops
    with current_video_id_lock:
        current_video_id += 1
        
    # Stop webcam processing
    is_playing = False

    is_cam = mode == "Cam-to-Live-Stream"
    is_image = mode == "Image-to-Image"
    is_video = mode == "Image-to-Video"

    return (
        gr.update(visible=is_cam),
        gr.update(visible=not is_cam),
        gr.update(visible=is_image or is_video),
        gr.update(active=is_video),
        gr.update(visible=not is_cam, value="Start Animation", variant="primary"),
        gr.update(visible=not is_cam)
    )

def on_generate_click(mode, image):
    if mode == "Image-to-Image":
        if image is None: return None
        
        _, input_tensor, output_tensor, resolution = get_processor()
        frame = crop_maximal_rectangle(to_bgr(image), resolution["height"], resolution["width"])
        
        with processor_lock:
            stream_processor.reset_cache()
            stream_processor.pack_is_ready.value = False
            stream_processor.frame_written.value = False
            input_tensor.copy_from(frame)

        # Wait for inference to finish
        while not stream_processor.pack_is_ready.value:
            time.sleep(0.01)
            
        # Wait for the output scheduler to write it to the output tensor
        while not stream_processor.frame_written.value:
            time.sleep(0.01)
            
        with processor_lock:
            processed = output_tensor.to_numpy()
            
        return to_rgb(processed)
    elif mode == "Image-to-Video":
        if image is not None: start_image_to_video(image)
        return None
    return None

def get_memory_stats():
    """Fetch RAM and VRAM usage."""
    ram = psutil.virtual_memory()
    ram_gb = ram.used / (1024**3)
    ram_total = ram.total / (1024**3)
    
    vram_mb = 0
    if stream_processor:
        vram_mb = stream_processor.get_reserved_memory()
    vram_gb = vram_mb / 1024
    
    sp_status = "Unknown"
    sp_error = ""
    if stream_processor and hasattr(stream_processor, "get_model_status"):
        sp_info = stream_processor.get_model_status()
        sp_status = sp_info.get("status", "Unknown")
        sp_error = sp_info.get("error", "")
        
    status_str = f"| **Inference Status:** {sp_status}"
    if sp_error:
        status_str += f" (⚠️ {sp_error})"
        
    return f"**System RAM:** {ram_gb:.1f} GB / {ram_total:.1f} GB | **GPU VRAM:** {vram_gb:.1f} GB {status_str}"

def on_model_type_change(model_type_val):
    is_gguf = model_type_val == "GGUF Quantized"
    return gr.update(visible=is_gguf), gr.update(visible=not is_gguf)

def main():
    global use_int8
    parser = argparse.ArgumentParser(description="Run FluxRT Gradio demo.")
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    args, _ = parser.parse_known_args()
    use_int8 = args.int8

    get_processor()
    use_reference_image = stream_processor.config.get("use_reference_image", False)

    css = """
    .gradio-container {
        transition: none !important;
    }
    #webcam_output, #webcam_input, .gradio-image {
        transition: none !important;
        animation: none !important;
    }
    .loading {
        display: none !important;
    }
    """

    with gr.Blocks(
        title="FluxRT - Real-Time Image & Video Transformer",
        css=css
    ) as demo:
        gr.Markdown(
            "## FluxRT - Real-Time Style Transfer & Animation\n"
            "Transform webcam, images, and videos in real-time. Upload a **reference image** to sync character appearance."
        )

        with gr.Row():
            memory_display = gr.Markdown(value="Fetching memory stats...", elem_id="memory_display")
            timer = gr.Timer(value=1)
            timer.tick(
                fn=get_memory_stats,
                inputs=[],
                outputs=memory_display,
            )

        with gr.Tabs() as tabs:
            with gr.Tab("Real-Time Generation"):
                mode = gr.Radio(
                    choices=["Cam-to-Live-Stream", "Image-to-Image", "Image-to-Video"],
                    value="Cam-to-Live-Stream",
                    label="Mode",
                )

                with gr.Row():
                    with gr.Column(visible=True) as webcam_output_col:
                        webcam_input = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam Input")
                        webcam_output = gr.Image(streaming=True, label="Animated Output")

                    with gr.Column(visible=False) as local_output_col:
                        local_output = gr.Image(label="Processed Output")

                local_timer = gr.Timer(value=0.04, active=False)

                with gr.Row():
                    with gr.Column(visible=False) as image_input_col:
                        image_file = gr.Image(label="Upload Image", type="numpy", sources=["upload"])
                    local_input = gr.Image(label="Input stream", visible=False)

                with gr.Row():
                    with gr.Column(scale=2):
                        prompt = gr.Textbox(value=default_prompt, label="Prompt", lines=3)
                    with gr.Column(scale=1):
                        ref_image_input = gr.Image(label="Reference Image (character/accessory sync)", type="numpy", sources=["upload"], image_mode="RGB")
                    with gr.Column(scale=1):
                        enable_lip_transfer = gr.Checkbox(label="Enable LivePortrait (Lip Sync)", value=False)

                with gr.Row():
                    start_btn = gr.Button("Start Animation", variant="primary", size="lg", visible=False)
                    generate_btn = gr.Button("Generate", variant="primary", size="lg", visible=False)

            with gr.Tab("Flux Models"):
                gr.Markdown("### Browse and Load FluxRT Model Variants")
                vram_recommendation = gr.Markdown(value=get_vram_recommendation())
                
                with gr.Row():
                    model_type = gr.Radio(
                        choices=["Full-precision FluxRT", "GGUF Quantized", "Quantized (int8)"],
                        value="GGUF Quantized",
                        label="Model Variant",
                    )
                
                with gr.Column(visible=True) as gguf_container:
                    with gr.Tabs():
                        with gr.Tab("Download GGUF"):
                            with gr.Row():
                                gguf_repo_id = gr.Textbox(
                                    value="leejet/FLUX.2-klein-base-4B-GGUF",
                                    label="HuggingFace Repo ID",
                                    info="The repository containing the GGUF files."
                                )
                                gguf_selector = gr.Dropdown(
                                        choices=[
                                            "flux-2-klein-base-4b-Q4_0.gguf",
                                            "flux-2-klein-base-4b-Q6_K.gguf",
                                            "flux-2-klein-base-4b-Q8_0.gguf"
                                        ],
                                    value="flux-2-klein-base-4b-Q6_K.gguf",
                                    label="Select GGUF Model to Download",
                                    info="Choose a GGUF model file."
                                )
                                apply_gguf_btn = gr.Button("Download & Load GGUF", variant="primary")
                                
                        with gr.Tab("Load Local GGUF"):
                            with gr.Row():
                                local_gguf_selector = gr.Dropdown(
                                    choices=list_local_gguf_models(),
                                    value=list_local_gguf_models()[0] if list_local_gguf_models() else None,
                                    label="Select Downloaded GGUF Model",
                                    info="Choose from locally stored .gguf models."
                                )
                                refresh_gguf_btn = gr.Button("🔄 Refresh List", variant="secondary")
                                load_local_gguf_btn = gr.Button("Load Local GGUF", variant="primary")
                
                with gr.Column(visible=False) as standard_container:
                    apply_variant_btn = gr.Button("Load Selected Variant", variant="primary")
                    
                model_status_box = gr.Textbox(label="Status / Compatibility Validation Logs", interactive=False)

        model_type.change(
            on_model_type_change,
            inputs=[model_type],
            outputs=[gguf_container, standard_container]
        )

        mode.change(
            switch_mode,
            inputs=mode,
            outputs=[
                webcam_output_col, local_output_col, image_input_col,
                local_timer, start_btn, generate_btn
            ]
        )

        start_btn.click(toggle_play, inputs=None, outputs=start_btn)

        webcam_input.stream(
            process_webcam,
            inputs=webcam_input,
            outputs=[webcam_output],
            stream_every=0.04,
            concurrency_limit=1,
        )

        generate_btn.click(
            on_generate_click,
            inputs=[mode, image_file],
            outputs=local_output,
        )

        local_timer.tick(poll_local_video, outputs=[local_input, local_output])
        prompt.change(set_prompt, inputs=prompt, outputs=None)

        ref_image_input.change(set_reference_image_ui, inputs=ref_image_input, outputs=None)
        enable_lip_transfer.change(set_lip_transfer_ui, inputs=enable_lip_transfer, outputs=None)

        apply_gguf_btn.click(
            download_and_apply_gguf,
            inputs=[gguf_repo_id, gguf_selector],
            outputs=[model_status_box, local_gguf_selector]
        )
        
        load_local_gguf_btn.click(
            load_local_gguf,
            inputs=[local_gguf_selector],
            outputs=[model_status_box]
        )
        
        refresh_gguf_btn.click(
            refresh_local_gguf_list,
            inputs=[],
            outputs=[local_gguf_selector]
        )
        
        apply_variant_btn.click(
            load_variant_ui,
            inputs=[model_type],
            outputs=[model_status_box]
        )

    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"))

if __name__ == "__main__":
    main()