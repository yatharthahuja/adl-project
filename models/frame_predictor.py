import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import sys

# Add SwinLSTM to path
current_dir = Path(__file__).parent.absolute()
swinlstm_dir = current_dir / "SwinLSTM"
sys.path.append(str(swinlstm_dir))

# Import SwinLSTM modules
from SwinLSTM_D import SwinLSTM

class BallFramePredictor:
    def __init__(self, ckpt_path, args):
        """Initialize the frame predictor with model and parameters"""
        self.device = args.device
        self.model = SwinLSTM(
            img_size=64,  # Smaller image size for just the ball
            patch_size=2,
            in_chans=1,
            embed_dim=128,
            depths_downsample=[2, 6],
            depths_upsample=[6, 2],
            num_heads=[4, 8],
            window_size=4
        ).to(self.device)
        
        # Load weights if available, otherwise will need training
        if Path(ckpt_path).exists():
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()
        
        # Initialize states and parameters
        self.states_down = [None] * len(args.depths_down)
        self.states_up = [None] * len(args.depths_down)
        self.n_past = args.n_past
        self.n_future = args.n_future
        self.buffer = []
        
        # Ball detection parameters
        self.ball_color_lower = np.array([0, 0, 100])  # Adjust for red ball detection (BGR)
        self.ball_color_upper = np.array([80, 80, 255])
        self.background = None
        self.ball_size = (64, 64)  # Size for ball patches

    def detect_ball(self, frame):
        """Detect the ball in the frame using color thresholding"""
        # Convert to BGR if grayscale
        if len(frame.shape) == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame
            
        # Create mask for ball based on color
        mask = cv2.inRange(frame_bgr, self.ball_color_lower, self.ball_color_upper)
        
        # Find contours of the ball
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (presumably the ball)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Extract ball patch with some margin
            margin = 10
            x_min = max(0, x - margin)
            y_min = max(0, y - margin)
            x_max = min(frame.shape[1], x + w + margin)
            y_max = min(frame.shape[0], y + h + margin)
            
            ball_patch = frame[y_min:y_max, x_min:x_max]
            ball_position = (x_min, y_min, x_max, y_max)
            
            # Update background if not set
            if self.background is None:
                self.background = frame.copy()
                # Remove ball from background by filling with nearby pixels
                cv2.fillConvexPoly(self.background, largest_contour, 
                                  np.median(frame_bgr, axis=(0,1)).astype(np.uint8))
            
            return ball_patch, ball_position, mask
        
        return None, None, None

    def add_frame(self, frame):
        """Extract ball from frame and add to buffer"""
        # Detect ball in the frame
        ball_patch, ball_position, mask = self.detect_ball(frame)
        
        if ball_patch is not None:
            # Resize ball patch to model's expected size
            ball_patch_resized = cv2.resize(ball_patch, self.ball_size)
            
            # Convert to grayscale if needed
            if len(ball_patch_resized.shape) == 3:
                ball_patch_gray = cv2.cvtColor(ball_patch_resized, cv2.COLOR_BGR2GRAY)
            else:
                ball_patch_gray = ball_patch_resized
            
            # Store both the processed patch and its position
            self.buffer.append((ball_patch_gray, ball_position, mask))
            
            if len(self.buffer) > self.n_past:
                self.buffer.pop(0)
                
            return True
        
        return False

    def ready(self):
        """Check if enough frames are collected for prediction"""
        return len(self.buffer) == self.n_past

    def predict(self):
        """Predict future ball positions and reconstruct full frames"""
        assert self.ready(), f"Need {self.n_past} frames first"
        
        # Prepare input tensor from ball patches
        ball_patches = [item[0] for item in self.buffer]
        x = np.stack(ball_patches).astype(np.float32) / 255.0
        x = torch.from_numpy(x)[None, ..., None].to(self.device)  # [1,T,H,W,1]
        x = x.permute(0, 4, 1, 2, 3)  # [1,1,T,H,W]
        
        # Get the latest mask and position
        _, last_position, last_mask = self.buffer[-1]
        
        # Run inference
        with torch.no_grad():
            # Process past frames
            for t in range(self.n_past):
                out, self.states_down, self.states_up = self.model(
                    x[:, :, t], self.states_down, self.states_up
                )
                last = out
            
            # Predict future frames
            future_frames = []
            for t in range(self.n_future):
                # Predict next ball patch
                out, self.states_down, self.states_up = self.model(
                    last, self.states_down, self.states_up
                )
                
                # Process output frame
                ball_pred = out.squeeze().cpu().numpy()
                ball_pred = (ball_pred * 255.0).clip(0, 255).astype(np.uint8)
                
                # Assume the ball position is similar to the last known position
                # In a real implementation, you might want to also predict position changes
                
                # Create full frame by adding predicted ball to background
                full_frame = self.background.copy()
                
                # Get the size of the predicted ball patch
                x_min, y_min, x_max, y_max = last_position
                patch_height = y_max - y_min
                patch_width = x_max - x_min
                
                # Resize predicted ball patch to match original size
                ball_pred_resized = cv2.resize(ball_pred, (patch_width, patch_height))
                
                # Create a mask for the ball (you might need to adjust this)
                # For simplicity, we'll use the last known mask, but ideally this would be predicted too
                ball_mask = cv2.resize(last_mask, (patch_width, patch_height))
                ball_mask_3ch = cv2.merge([ball_mask, ball_mask, ball_mask])
                
                # Place the ball on the background using the mask
                roi = full_frame[y_min:y_max, x_min:x_max]
                roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(ball_mask))
                
                # Create foreground by applying mask to ball prediction
                if len(ball_pred_resized.shape) == 2:
                    ball_pred_resized = cv2.cvtColor(ball_pred_resized, cv2.COLOR_GRAY2BGR)
                    
                ball_fg = cv2.bitwise_and(ball_pred_resized, ball_pred_resized, mask=ball_mask)
                
                # Combine background and foreground
                dst = cv2.add(roi_bg, ball_fg)
                full_frame[y_min:y_max, x_min:x_max] = dst
                
                future_frames.append(full_frame)
                
                last = out
                
            return future_frames