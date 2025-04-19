"""
Logging utilities for recording simulation data.
"""

import os
import json
from config import LOG_DIR

def prepare_inference_stats(input_frames, output_frames, input_output_pairs, sim_params=None, sim_idx=None):
    """Prepare inference statistics without writing to file"""
    # Calculate statistics
    latencies = [lat for *_, lat in input_output_pairs]
    avg_latency = sum(latencies)/len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    
    # Create statistics dictionary
    stats = {
        'simulation_index': sim_idx,
        'parameters': sim_params,
        'num_input_frames': len(input_frames),
        'num_output_frames': len(output_frames),
        'num_paired_frames': len(input_output_pairs),
        'avg_latency': avg_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'input_output_pairs': [[i, o, float(l)] for i, o, l in input_output_pairs],  # Convert to serializable format
        'input_frames': {str(k): float(v) for k, v in input_frames.items()},  # Convert to serializable format
        'output_frames': {str(k): float(v) for k, v in output_frames.items()}  # Convert to serializable format
    }
    
    return stats

def write_inference_stats(stats, log_dir=LOG_DIR):
    """Write prepared inference statistics to files"""
    print("Write inference stats")
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    sim_idx = stats.get('simulation_index')
    sim_params = stats.get('parameters')
    
    # Determine file name
    if sim_idx is not None:
        filename = f"inference_stats_sim_{sim_idx}.json"
    else:
        filename = "inference_stats.json"
    
    # Write to JSON file
    with open(os.path.join(log_dir, filename), 'w') as f:
        json.dump(stats, f, indent=2)
    
    return os.path.join(log_dir, filename)

# For backward compatibility
def log_inference_data(input_frames, output_frames, input_output_pairs, sim_params=None, sim_idx=None, log_dir=LOG_DIR):
    """Log inference timing data to a file - wrapper for backward compatibility"""
    stats = prepare_inference_stats(input_frames, output_frames, input_output_pairs, sim_params, sim_idx)
    log_file = write_inference_stats(stats, log_dir)
    return stats, log_file
