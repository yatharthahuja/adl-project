import imageio.v2
import os

# Define the folder containing your images
folder_path = "simulation_frames/"  # Replace with the actual folder path
output_gif_path = "simulation.gif"  # Replace with the desired output path

# Read all image files in the folder
images = []
filenames = sorted(os.listdir(folder_path))  # Sort filenames for correct frame order

for filename in filenames:
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image formats
        file_path = os.path.join(folder_path, filename)
        images.append(imageio.imread(file_path))

# Create and save the GIF
imageio.mimsave(output_gif_path, images, duration=0.5) 