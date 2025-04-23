import gzip
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class Moving_MNIST(Dataset):

    def __init__(self, args, split):

        super(Moving_MNIST, self).__init__()

        with gzip.open(args.train_data_dir, 'rb') as f:
            self.datas = np.frombuffer(f.read(), np.uint8, offset=16)
            self.datas = self.datas.reshape(-1, *args.image_size)

        if split == 'train':
            self.datas = self.datas[args.train_samples[0]: args.train_samples[1]]
        else:
            self.datas = self.datas[args.valid_samples[0]: args.valid_samples[1]]

        self.image_size = args.image_size
        self.input_size = args.input_size
        self.step_length = args.step_length
        self.num_objects = args.num_objects

        self.num_frames_input = args.num_frames_input
        self.num_frames_output = args.num_frames_output
        self.num_frames_total = args.num_frames_input + args.num_frames_output

        print('Loaded {} {} samples'.format(self.__len__(), split))

    def _get_random_trajectory(self, seq_length):

        assert self.input_size[0] == self.input_size[1]
        assert self.image_size[0] == self.image_size[1]

        canvas_size = self.input_size[0] - self.image_size[0]

        x = random.random()
        y = random.random()

        theta = random.random() * 2 * np.pi

        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)

        for i in range(seq_length):

            y += v_y * self.step_length
            x += v_x * self.step_length

            if x <= 0.: x = 0.; v_x = -v_x;
            if x >= 1.: x = 1.; v_x = -v_x
            if y <= 0.: y = 0.; v_y = -v_y;
            if y >= 1.: y = 1.; v_y = -v_y

            start_y[i] = y
            start_x[i] = x

        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)

        return start_y, start_x

    def _generate_moving_mnist(self, num_digits=2):

        data = np.zeros((self.num_frames_total, *self.input_size), dtype=np.float32)

        for n in range(num_digits):

            start_y, start_x = self._get_random_trajectory(self.num_frames_total)
            ind = np.random.randint(0, self.__len__())
            digit_image = self.datas[ind]

            for i in range(self.num_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.image_size[0]
                right = left + self.image_size[1]
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]

        return data

    def __getitem__(self, item):

        num_digits = random.choice(self.num_objects)
        images = self._generate_moving_mnist(num_digits)

        inputs = torch.from_numpy(images[:self.num_frames_input]).permute(0, 3, 1, 2).contiguous()
        targets = torch.from_numpy(images[self.num_frames_output:]).permute(0, 3, 1, 2).contiguous()

        return inputs / 255., targets / 255.

    def __len__(self):
        return self.datas.shape[0]


class Moving_MNIST_Test(Dataset):
    def __init__(self, args):
        super(Moving_MNIST_Test, self).__init__()

        self.num_frames_input = args.num_frames_input
        self.num_frames_output = args.num_frames_output
        self.num_frames_total = args.num_frames_input + args.num_frames_output

        self.dataset = np.load(args.test_data_dir)
        self.dataset = self.dataset[..., np.newaxis]
        
        print('Loaded {} {} samples'.format(self.__len__(), 'test'))
        
    def __getitem__(self, index):
        images =  self.dataset[:, index, ...]

        inputs = torch.from_numpy(images[:self.num_frames_input]).permute(0, 3, 1, 2).contiguous()
        targets = torch.from_numpy(images[self.num_frames_output:]).permute(0, 3, 1, 2).contiguous()

        return inputs / 255., targets / 255.

    def __len__(self):
        return len(self.dataset[1])
        

class CustomDataset(Dataset):
    def __init__(self, split):
        super(CustomDataset, self).__init__()

        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        data_path = {
            'train': "../data/train_images.npy",
            'val': "../data/val_images.npy",
            'test': "../data/test_images.npy"
        }[split]

        # Load flat frame array: shape (N, H, W)
        self.datas = np.load(data_path, allow_pickle=True)  # now matches original attribute
        total_frames = self.datas.shape[0]
        H, W = self.datas.shape[1:]

        self.image_size = (H, W)
        self.input_size = (64, 64)  # unused, preserved
        self.step_length = 1
        self.num_objects = 2        # unused, preserved

        self.num_frames_input = 5
        self.num_frames_output = 1
        self.num_frames_total = self.num_frames_input + self.num_frames_output

        # Create frame sequences with sliding window
        self.samples = []
        for i in range(0, total_frames - self.num_frames_total + 1, self.step_length):
            seq = self.datas[i:i + self.num_frames_total]
            self.samples.append(seq)

        self.samples = np.stack(self.samples)  # shape: (num_seq, T, H, W)

        print(f'Loaded {len(self.samples)} {split} sequences from {data_path}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]  # shape: (T, H, W)
        seq = seq.astype(np.float32) / 255.0
        seq = np.expand_dims(seq, axis=1)  # (T, 1, H, W)
        input_seq = seq[:self.num_frames_input]
        target_seq = seq[self.num_frames_input:]
        return torch.from_numpy(input_seq), torch.from_numpy(target_seq)
