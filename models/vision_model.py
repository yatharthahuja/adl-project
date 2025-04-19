"""
Vision model loader and inference functions
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

def load_vision_model():
    """Load and configure the OpenVLA vision-language-action model"""
    print("Loading OpenVLA vision-language-action model...")
    
    try:
        processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b", 
            trust_remote_code=True
        )
        
        model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        ).to("cuda:0")
        
        print("Vision model loaded successfully.")
        return processor, model
        
    except Exception as e:
        print(f"Error loading vision model: {e}")
        raise

def get_openvla_output(image_input, instruction, processor, model):
    """
    Get action prediction from OpenVLA model based on image and instruction
    
    Args:
        image_input: PIL Image or numpy array or path to image
        instruction: Text instruction for the robot
        processor: OpenVLA processor
        model: OpenVLA model
        
    Returns:
        action: Array of predicted action parameters
    """
    # Load and preprocess image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input.astype(np.uint8))
    else:
        raise ValueError("Unsupported input type. Must be file path or numpy array.")

    # Resize image to model input size
    image = image.resize((224, 224))
    
    # Format the prompt for the model
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    # Process inputs and predict action
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    return action