import sys
import os
import time
import numpy as np

# Force current working directory to be 'app'
app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(app_dir)
sys.path.insert(0, "src")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    from fluxrt import StreamProcessor
    
    config_path = "configs/stream_processor_config.json"
    print(f"Loading config from {config_path}...")
    sp = StreamProcessor(config_path)
    
    print("Starting StreamProcessor...")
    sp.start()
    
    # We will write dummy frames to trigger processing and test the scheduler copy
    input_tensor = sp.get_input_tensor()
    output_tensor = sp.get_output_tensor()
    
    h = sp.resolution["height"]
    w = sp.resolution["width"]
    dummy_frame = np.ones((h, w, 3), dtype=np.uint8) * 128  # gray frame
    
    print("Monitoring status and output for up to 120 seconds...")
    success = False
    
    for i in range(240):
        time.sleep(1.0)
        
        # Write dummy frame
        input_tensor.copy_from(dummy_frame)
        
        status = sp.get_model_status()
        print(f"[{i}s] Status: {status.get('status')} | Error: {status.get('error')}")
        
        # Read from output tensor
        out_frame = output_tensor.to_numpy()
        mean_val = out_frame.mean()
        print(f"[{i}s] Output frame mean pixel value: {mean_val:.4f}")
        
        if status.get("status") == "Error":
            print(f"FAILED with error: {status.get('error')}")
            break
            
        # If output frame mean is non-zero, it means the scheduler process has successfully
        # copied the generated frames to the output tensor!
        if mean_val > 5.0 and (status.get("status") == "Processing" or status.get("status") == "Ready"):
            success = True
            print("SUCCESS! Output frame is active and non-black.")
            # Save the frame to check
            import cv2
            cv2.imwrite("test_output_frame.png", out_frame)
            print("Saved test output frame to test_output_frame.png")
            break
            
    print("Stopping StreamProcessor...")
    sp.stop()
    print("Done.")
    sys.exit(0 if success else 1)
