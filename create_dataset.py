import numpy as np
import cv2
import argparse
import os

def process_images(npy_path):
    # Load images with pickle support
    dict = np.load(npy_path, allow_pickle=True).item()

    videos = np.array(dict['videos'])
    
    images = []
    for video in videos:
        for img in video:
            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # resise to 28x28
            img = cv2.resize(img, (64, 64))
            images.append(img)

    train_images = images[int(len(images) * 0.7):]
    val_images = images[int(len(images) * 0.7):int(len(images) * 0.85)]
    test_images = images[int(len(images) * 0.85):]
    
    # convert to dictionary
    train_images = np.array(train_images)
    val_images = np.array(val_images)
    test_images = np.array(test_images)
    
    np.save("./data/train_images.npy", train_images)
    np.save("./data/val_images.npy", val_images)
    np.save("./data/test_images.npy", test_images)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from a .npy file.")
    parser.add_argument("--npy_path", type=str, help="Path to the .npy file.", default="./color_dataset.npy")

    args = parser.parse_args()
    process_images(args.npy_path)
