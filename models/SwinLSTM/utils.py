import os
import time
import torch
import random
import logging
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
import imageio
from PIL import Image, ImageDraw, ImageFont

matplotlib.use('agg')

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True



def visualize(inputs, targets, outputs, epoch, idx, cache_dir, fps=5):
    """
    inputs, targets, outputs: torch tensors [B, T, 1, H, W]
    Saves a static PNG grid plus a comparison GIF with overlaid labels.
    """
    os.makedirs(cache_dir, exist_ok=True)
    B, T, _, H, W = targets.shape
    assert B >= 1, "Need at least one video in the batch"

    # ==== 1) Static PNG grid: Real vs Predicted ====
    fig, ax = plt.subplots(2, T, figsize=(T * 2, 4))
    # Row labels
    fig.text(0.04, 0.75, "Real",      va="center", rotation="vertical", fontsize=14)
    fig.text(0.04, 0.25, "Predicted", va="center", rotation="vertical", fontsize=14)

    for t in range(T):
        real_img = targets[0, t, 0].cpu().numpy()
        pred_img = outputs[0, t, 0].cpu().numpy()

        ax[0, t].imshow(real_img, cmap="gray")
        ax[1, t].imshow(pred_img, cmap="gray")

        # Column title
        ax[0, t].set_title(f"Frame {t}", fontsize=10)

        ax[0, t].axis("off")
        ax[1, t].axis("off")

    plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])
    png_path = os.path.join(cache_dir, f"{epoch:03d}-{idx:03d}.png")
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # ==== 2) Side‑by‑side GIF with text overlays ====
    # Prepare raw uint8 sequences
    real_seq = (targets[0, :, 0].cpu().numpy() * 255).astype(np.uint8)
    pred_seq = (outputs[0, :, 0].cpu().numpy() * 255).astype(np.uint8)

    frames = []
    font = ImageFont.load_default()
    for t, (r, p) in enumerate(zip(real_seq, pred_seq)):
        # make 3‑channel
        left  = np.stack([r]*3, axis=-1)
        right = np.stack([p]*3, axis=-1)
        combo = np.concatenate([left, right], axis=1)  # [H, 2W, 3]

        # overlay text
        img = Image.fromarray(combo)
        draw = ImageDraw.Draw(img)
        # positions
        draw.text((10, 10),           "Real",      font=font, fill=(255,255,255))
        draw.text((W+10, 10),         "Predicted", font=font, fill=(255,255,255))
        draw.text((combo.shape[1]//2 - 30, 10), f"Frame {t}",
                  font=font, fill=(255,255,255))

        frames.append(np.array(img))

    gif_path = os.path.join(cache_dir, f"{epoch:03d}-{idx:03d}.gif")
    imageio.mimsave(gif_path, frames, fps=fps)

    print(f"Saved PNG → {png_path}")
    print(f"Saved GIF → {gif_path}")



def plot_loss(loss_records, loss_type, epoch, plot_dir, step):
    plt.plot(range((epoch + 1) // step), loss_records, label=loss_type)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, '{}_loss_records.png'.format(loss_type)))
    plt.close()

def MAE(pred, true):
    return np.mean(np.abs(pred - true), axis=(0, 1)).sum()
    
def MSE(pred, true):
    return np.mean((pred - true) ** 2, axis=(0, 1)).sum()

# cite the 'PSNR' code from E3D-LSTM, Thanks!
# https://github.com/google/e3d_lstm/blob/master/src/trainer.py line 39-40
def PSNR(pred, true):
    mse = np.mean((np.uint8(pred * 255) - np.uint8(true * 255)) ** 2)
    return 20 * np.log10(255) - 10 * np.log10(mse)

def compute_metrics(predictions, targets):
    targets = targets.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    predictions = predictions.permute(0, 1, 3, 4, 2).detach().cpu().numpy()

    batch_size = predictions.shape[0]
    Seq_len = predictions.shape[1]

    ssim = 0

    for batch in range(batch_size):
        for frame in range(Seq_len):
            ssim += structural_similarity(
                        targets[batch, frame].squeeze(),
                        predictions[batch, frame].squeeze(),
                        data_range=1.0 if targets.max() <= 1.0 else 255.0,  # Adjust based on your normalization
                        channel_axis=None  # Required for grayscale images in newer skimage versions
                    )

    ssim /= (batch_size * Seq_len)

    mse = MSE(predictions, targets)

    return mse, ssim

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_dir(args):

    cache_dir = os.path.join(args.res_dir, 'cache')
    check_dir(cache_dir)

    model_dir = os.path.join(args.res_dir, 'model')
    check_dir(model_dir)

    log_dir = os.path.join(args.res_dir, 'log')
    check_dir(log_dir)

    return cache_dir, model_dir, log_dir

def init_logger(log_dir):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(log_dir, time.strftime("%Y_%m_%d") + '.log'),
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging
