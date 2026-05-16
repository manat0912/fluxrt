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

import argparse
import threading
import time

import cv2
import numpy as np
import gradio as gr

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
    sp.set_reference_image(image)

def set_lip_transfer_ui(enabled: bool):
    sp, _, _, _ = get_processor()
    if hasattr(sp, 'set_lip_transfer'):
        sp.set_lip_transfer(enabled)

def toggle_play():
    global is_playing
    is_playing = not is_playing
    return gr.update(value="Stop" if is_playing else "Start Animation", variant="stop" if is_playing else "primary")

def download_and_apply_gguf(repo_id, gguf_filename, progress=gr.Progress()):
    if not gguf_filename:
        return "Please select a GGUF model to download."
    if not repo_id:
        return "Please enter a HuggingFace Repository ID."
        
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
        
        return f"Successfully downloaded and applied: {gguf_filename}"
    except Exception as e:
        return f"Error downloading model: {e}"

def process_webcam(frame):
    global is_playing
    if not is_playing or frame is None:
        return frame
    _, processed = process_frame(to_bgr(frame))
    return to_rgb(processed)

def _video_loop(video_path: str, video_id: int):
    global local_current_frame, local_processed_frame
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_time = 1.0 / fps
    try:
        while True:
            with current_video_id_lock:
                if current_video_id != video_id:
                    break
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            start = time.time()
            input_frame, processed = process_frame(frame)
            with local_frame_lock:
                local_current_frame = to_rgb(input_frame)
                local_processed_frame = to_rgb(processed)
            time.sleep(max(0, frame_time - (time.time() - start)))
    finally:
        cap.release()

def start_local_video(video_path: str | None):
    global current_video_id, local_current_frame, local_processed_frame
    with current_video_id_lock:
        current_video_id += 1
        my_id = current_video_id
    with local_frame_lock:
        local_current_frame = None
        local_processed_frame = None
    if not video_path: return
    t = threading.Thread(target=_video_loop, args=(video_path, my_id), daemon=True)
    t.start()

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

    is_cam = mode == "cam to live stream"
    is_video = mode == "video to video"
    is_image = mode == "image to image"
    is_edit = mode == "edit image to video"

    return (
        gr.update(visible=is_cam),
        gr.update(visible=not is_cam),
        gr.update(visible=is_video),
        gr.update(visible=is_image or is_edit),
        gr.update(active=(is_video or is_edit)),
        gr.update(visible=is_cam, value="Start Animation", variant="primary"),
        gr.update(visible=not is_cam)
    )

def on_generate_click(mode, video_path, image):
    if mode == "image to image":
        if image is None: return None
        
        _, input_tensor, output_tensor, resolution = get_processor()
        frame = crop_maximal_rectangle(to_bgr(image), resolution["height"], resolution["width"])
        
        with processor_lock:
            input_tensor.copy_from(frame)
            stream_processor.frame_written.value = False

        # Wait for the first write (could be an old frame finishing)
        while not stream_processor.frame_written.value:
            time.sleep(0.05)
            
        stream_processor.frame_written.value = False
        
        # Wait for the second write (guaranteed to be the new frame)
        while not stream_processor.frame_written.value:
            time.sleep(0.05)
            
        with processor_lock:
            processed = output_tensor.to_numpy()
            
        return to_rgb(processed)
    elif mode == "video to video":
        if video_path: start_local_video(video_path)
        return None
    elif mode == "edit image to video":
        if image is not None: start_image_to_video(image)
        return None
    return None

def main():
    global use_int8
    parser = argparse.ArgumentParser(description="Run FluxRT Gradio demo.")
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    args, _ = parser.parse_known_args()
    use_int8 = args.int8

    get_processor()
    use_reference_image = stream_processor.config.get("use_reference_image", False)

    with gr.Blocks(
        title="FluxRT - Real-Time Image & Video Transformer",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    ) as demo:
        gr.Markdown(
            "## FluxRT - Real-Time Style Transfer & Animation\n"
            "Transform webcam, images, and videos in real-time. Upload a **reference image** to sync character appearance."
        )

        mode = gr.Radio(
            choices=["cam to live stream", "video to video", "image to image", "edit image to video"],
            value="cam to live stream",
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
            with gr.Column(visible=False) as local_input_col:
                video_file = gr.File(label="Choose Local Video", file_count="single", file_types=["video"], type="filepath")
            with gr.Column(visible=False) as image_input_col:
                image_file = gr.Image(label="Upload Image", type="numpy", sources=["upload"])
            local_input = gr.Image(label="Input stream", visible=False)

        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(value=default_prompt, label="Prompt", lines=3)
            if use_reference_image:
                with gr.Column(scale=1):
                    ref_image_input = gr.Image(label="Reference Image (character sync)", type="numpy", sources=["upload"], image_mode="RGB")
            with gr.Column(scale=1):
                enable_lip_transfer = gr.Checkbox(label="Enable LivePortrait (Lip Sync)", value=False)

        with gr.Row():
            start_btn = gr.Button("Start Animation", variant="primary", size="lg", visible=True)
            generate_btn = gr.Button("Generate", variant="primary", size="lg", visible=False)

        with gr.Accordion("Model Management", open=False):
            with gr.Row():
                gguf_repo_id = gr.Textbox(
                    value="leejet/FLUX.2-klein-4B-GGUF",
                    label="HuggingFace Repo ID",
                    info="The repository containing the GGUF files."
                )
                gguf_selector = gr.Dropdown(
                    choices=[
                        "FLUX.2-klein-4B-Q4_K_M.gguf",
                        "FLUX.2-klein-4B-F16.gguf",
                        "FLUX.2-klein-4B-BF16.gguf",
                        "FLUX.2-klein-4B-Q8_0.gguf",
                        "FLUX.2-klein-4B-Q5_K_M.gguf",
                        "FLUX.2-klein-4B-Q3_K_M.gguf",
                        "FLUX.2-klein-4B-Q2_K.gguf"
                    ],
                    label="Select GGUF Model",
                    info="Choose a GGUF model. If you click apply, it will download if missing, then apply dynamically."
                )
                apply_gguf_btn = gr.Button("Download & Apply Model")
            gguf_status = gr.Textbox(label="Status", interactive=False)

        mode.change(
            switch_mode,
            inputs=mode,
            outputs=[
                webcam_output_col, local_output_col, local_input_col, image_input_col,
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
            inputs=[mode, video_file, image_file],
            outputs=local_output,
        )

        local_timer.tick(poll_local_video, outputs=[local_input, local_output])
        prompt.change(set_prompt, inputs=prompt, outputs=None)

        if use_reference_image:
            ref_image_input.change(set_reference_image_ui, inputs=ref_image_input, outputs=None)

        enable_lip_transfer.change(set_lip_transfer_ui, inputs=enable_lip_transfer, outputs=None)

        apply_gguf_btn.click(
            download_and_apply_gguf,
            inputs=[gguf_repo_id, gguf_selector],
            outputs=[gguf_status]
        )

    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1")

if __name__ == "__main__":
    main()