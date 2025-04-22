# https://github.com/XiYe20/VPTR?tab=readme-ov-file 
# https://github.com/lllyasviel/FramePack


import numpy as np
import cv2
import argparse
import os

def process_images(npy_path, crop_dims, save_flag, save_dir):
    # Load images from .npy file
    images = np.load(npy_path)
    print(f"Loaded {len(images)} images from {npy_path}")

    # Ensure save directory exists if saving is enabled
    if save_flag and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, img in enumerate(images):
        cv2.imshow("Original", img)
        cv2.waitKey(500)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Crop the image: crop_dims = (start_x, start_y, width, height)
        x, y, w, h = crop_dims
        cropped = gray[y:y+h, x:x+w]

        cv2.imshow("Processed", cropped)
        cv2.waitKey(500)

        # Save if flag is True
        if save_flag:
            filename = os.path.join(save_dir, f"processed_{idx}.png")
            cv2.imwrite(filename, cropped)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from npy file.")
    parser.add_argument("--npy_path", type=str, required=True, help="Path to the .npy file containing images.")
    parser.add_argument("--crop_dims", type=int, nargs=4, metavar=('x', 'y', 'w', 'h'), required=True,
                        help="Crop dimensions as x y w h.")
    parser.add_argument("--save", action='store_true', help="Flag to save processed images.", default=False)
    parser.add_argument("--save_dir", type=str, default="./processed_images", help="Directory to save processed images.")

    args = parser.parse_args()

    process_images(args.npy_path, args.crop_dims, args.save, args.save_dir)


# Example usage:
# python pre-process.py --npy_path ./data/000000.npy --crop_dims 0 0 512 512 --save --save_dir ./processed_images