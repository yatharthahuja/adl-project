"""
Threading utilities for handling inference in separate threads
"""

import queue
import threading
import time
from models.vision_model import get_openvla_output

def setup_inference_thread(processor, model):
    """Set up threaded inference to avoid blocking the simulation"""
    inference_queue = queue.Queue()
    output_queue    = queue.Queue()
    track_queue     = queue.Queue()  # For tracking input-output relationships
    stop_event      = threading.Event()

    # Perform model warm-up before starting the inference thread
    def warmup_model():
        print("Warming up model with a dummy inference...")
        try:
            # Create a small dummy image for warm-up
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            dummy_instruction = "pick up the ball"
            _ = get_openvla_output(dummy_img, dummy_instruction, processor, model)
            print("Model warm-up complete")
        except Exception as e:
            print(f"Model warm-up failed: {e!r}")

    # Run warm-up
    warmup_model()

    # Define the worker function for the inference thread
    def inference_worker():
        # Flag to optimize first frame processing
        is_first_frame = True
        
        while True:
            try:
                # For first frame: use a very small timeout to reduce latency
                timeout = 0.001 if is_first_frame else 0.1
                item = inference_queue.get(timeout=timeout)
                is_first_frame = False
            except queue.Empty:
                # if we've been asked to stop, break out now
                if stop_event.is_set():
                    break
                else:
                    continue

            # shutdown sentinel
            if item is None:
                inference_queue.task_done()
                break

            # For first frame, skip the draining process
            if is_first_frame:
                latest = item
            else:
                # Drain everything else, keeping only the very last frame
                latest = item
                while True:
                    try:
                        nxt = inference_queue.get_nowait()
                        if nxt is None:
                            # re‑enqueue the sentinel so we still shut down cleanly
                            inference_queue.put(None)
                            inference_queue.task_done()
                            latest = None
                            break
                        latest = nxt
                        inference_queue.task_done()
                    except queue.Empty:
                        break

                # if we saw a shutdown sentinel in the drain, quit
                if latest is None:
                    break

            # Run inference on the latest frame
            frame_idx, rgb, instruction = latest
            try:
                t0 = time.time()
                print(f"Starting inference for input frame {frame_idx} at time {t0:.4f}")

                # Record input-frame timestamp
                track_queue.put(("input", frame_idx, t0))

                # Run the model
                openvla_output = get_openvla_output(rgb, instruction, processor, model)

                t1 = time.time()
                inf_time = t1 - t0

                # Record output-frame timestamp
                track_queue.put(("output", frame_idx, t1))
                output_queue.put((frame_idx, openvla_output, inf_time))

                print(f"Completed inference for frame {frame_idx} in {inf_time:.4f}s")
            except Exception as e:
                print(f"[Inference] ERROR on frame {frame_idx}: {e!r}")
            finally:
                inference_queue.task_done()

    # Start the thread
    inference_thread = threading.Thread(target=inference_worker)
    inference_thread.start()

    return inference_queue, output_queue, inference_thread, track_queue, stop_event