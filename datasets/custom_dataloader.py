import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
from PIL import Image

from os import listdir
from os.path import join
import re


# converts string to integers when possible
def atoi(text):
    return int(text) if text.isdigit() else text


# applies atoi to a string
def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", str(text))]


class VideoDataset(data.Dataset):
    def __init__(
        self,
        folder,
        num_segments=1,
        new_length=1,
        image_tmpl="{:05d}.jpg",
        transform=None,
        random_shift=True,
        test_mode=False,
        index_bias=1,
    ):

        self.base_dir = folder
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = False
        self.index_bias = index_bias

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1

        self.classes_names = sorted(listdir(folder))
        self.classes = []
        self.video_list = []

        # select all videos with enough frames
        for y, c in enumerate(self.classes_names):
            self.classes.append([int(y), str(c)])
            d = join(self.base_dir, c)
            videos = listdir(d)
            for video in videos:
                video = join(d, video)
                if len(self.find_frames(video)) >= self.num_segments:
                    self.video_list.append((video, y))

        self.initialized = False

    def _load_image(self, directory, idx):

        return [
            Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert(
                "RGB"
            )
        ]

    @property
    def total_length(self):
        return self.num_segments * self.seg_length

    # selects frames from input sequence
    def find_frames(self, video):
        frames = [join(video, f) for f in listdir(video) if self.is_img(f)]
        return frames

    # checks if input is image
    def is_img(self, f):
        return str(f).lower().endswith("jpg") or str(f).lower().endswith("jpeg")

    def _sample_indices(self, video):
        offsets = list()
        ticks = [
            i * len(self.find_frames(video)) // self.num_segments
            for i in range(self.num_segments + 1)
        ]

        for i in range(self.num_segments):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= self.seg_length:
                tick += randint(tick_len - self.seg_length + 1)
            offsets.extend([j for j in range(tick, tick + self.seg_length)])
        return np.array(offsets) + self.index_bias

    def _get_val_indices(self, video):
        if self.num_segments == 1:
            return (
                np.array([len(self.find_frames(video)) // 2], dtype=np.int)
                + self.index_bias
            )

        if len(self.find_frames(video)) <= self.total_length:
            if self.loop:
                return (
                    np.mod(np.arange(self.total_length), len(self.find_frames(video)))
                    + self.index_bias
                )
            return (
                np.array(
                    [
                        i * len(self.find_frames(video)) // self.total_length
                        for i in range(self.total_length)
                    ],
                    dtype=np.int,
                )
                + self.index_bias
            )
        offset = (
            len(self.find_frames(video)) / self.num_segments - self.seg_length
        ) / 2.0
        return (
            np.array(
                [
                    i * len(self.find_frames(video)) / self.num_segments + offset + j
                    for i in range(self.num_segments)
                    for j in range(self.seg_length)
                ],
                dtype=np.int,
            )
            + self.index_bias
        )

    def __getitem__(self, index):
        video, label = self.video_list[index]
        segment_indices = (
            self._sample_indices(video)
            if self.random_shift
            else self._get_val_indices(video)
        )
        return self.get(video, label, segment_indices)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def get(self, video, label, indices):
        images = list()

        # find frames
        frame_paths = self.find_frames(video)
        frame_paths.sort(key=natural_keys)

        for i, seg_ind in enumerate(indices):
            p = int(seg_ind) - 1
            try:
                seg_imgs = [Image.open(frame_paths[p]).convert("RGB")]
            except OSError:
                print('ERROR: Could not read image "{}"'.format(video))
                print("invalid indices: {}".format(indices))
                raise
            images.extend(seg_imgs)
        process_data = self.transform(images)
        return process_data, label

    def __len__(self):
        return len(self.video_list)
