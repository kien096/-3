import os
import json
from glob import glob
from typing import Tuple
from random import choice

import torch
import librosa
import numpy as np
import albumentations as A
from joblib import Memory
from torch.utils.data import Dataset
from torch.nn.functional import normalize

from config import INPUT_SHAPE_IMAGE, JSON_NAME, AUGMENTATION_DATA, FREQUENCY_DISCREDITING
from src.augmentation_audio import (PitchShift, FreqMask, TimeMask, Resample, PadTranc, StretchAudio,
                                    OverlayAudio, AddNoise, PolarityInversion, Gain, SpeedTuning, MakeQuiter)
from src.utils import show_spectogramm, write_sound

cache_dir = 'data/cash'
mem = Memory(cache_dir, mmap_mode='c')

LABEL_UNIQUE = ['fma' ,'fma-western-art', 'hd-classical' ,'jamendo' ,'rfm']
def load_audio(wav_path: str):
    wav, sr = librosa.load(wav_path, sr=FREQUENCY_DISCREDITING)

    return wav, sr


class CustomDataset(Dataset):
    def __init__(self, df, augmentation_data: bool = AUGMENTATION_DATA,
                 image_shape: Tuple[int, int, int] = INPUT_SHAPE_IMAGE, is_train: bool = True,
                 shuffle_data: bool = False, overlay_audio: bool = False):
        """
        Data generator for prepare input data.
        :param data_path: a path to the folder where the data is stored.
        :param json_name: the name of the json file that contains information about the files to download.
        :param augmentation_data: if this parameter is True, then augmentation is applied to the training dataset.
        :param image_shape: this is image shape (channels, height, width).
        :param is_train: if is_train = True, then we work with val images, otherwise with test.
        :param shuffle_data: if this parameter is True, then data will be shuffled every time.
        """

        self.image_shape = image_shape
        self.overlay_audio = overlay_audio
        self.load = mem.cache(load_audio)

       
        if overlay_audio:
            self.norm_noise = self.augmentation_audio(augm=False)
            self.make_overlay = OverlayAudio(p=1.0, intensity_overlay=(6, 7))

        
        self.data = df
        self.all_audio_noise = self.data[self.data['types']=='music'].file.to_list()
        self.labels = self.data[self.data['types']=='music'].label.to_list()
        # augmentation data
        if is_train:
            self.aug_image = self.augmentation_images(augmentation_data)
            self.aug_audio = self.augmentation_audio(augmentation_data)
            self.noise = 105
        else:
            self.aug_image = self.augmentation_images(augmentation_data)
            self.aug_audio = self.augmentation_audio(augmentation_data)
            self.noise = 7

        # self.data = list(self.data.items())
        if shuffle_data:
            self.on_epoch_end()

    def on_epoch_end(self) -> None:
        """
        Random shuffling of data at the end of each epoch.
        """
        np.random.shuffle(self.all_audio_noise)

    def __len__(self) -> int:
        return len(self.all_audio_noise)

    def __getitem__(self, idx) -> Tuple[torch.tensor, torch.tensor]:
        """
        This function prepares the spectogram and label.
        :param idx: a number of the element to load.
        :return: image tensor and label tensor.
        """
        wav_path, class_name_index = self.all_audio_noise[idx], self.labels[idx]
        self.wav_path = wav_path
        wav, sr = self.load(wav_path)
        spec = self.prepare_audio(wav, sr)
        lable = torch.tensor(int(LABEL_UNIQUE.index(class_name_index)))
        # print(lable.shape)
        return spec, lable

    def prepare_audio(self, wav, sr):
        data = wav, sr
        aug_wav = self.aug_audio(data=data)
        try:
            wav, sr = aug_wav['data']
        except TypeError:
            print(self.wav_path)

        if self.overlay_audio:
            path_noise = choice(self.all_audio_noise)
            wav_noise, sr_noise = self.load(path_noise)
            data_noise = wav_noise, sr_noise
            wav_noise, sr_noise = self.norm_noise(data=data_noise)['data']
            data_overlay = ((wav, sr), (wav_noise, sr_noise))
            overlay = self.make_overlay(data=data_overlay)
            wav, sr = overlay['data']

            if isinstance(wav, tuple):
                wav, sr = wav[0], wav[1]

        spec_first = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmin=20,
                                                    fmax=8300)
        spec_first = librosa.power_to_db(spec_first, top_db=100)

        resize_spec = self.aug_image(image=spec_first)
        spec = resize_spec['image']
        spec = torch.from_numpy(spec)
        spec = normalize(spec).float()
        spec = spec.unsqueeze(0)

        return spec

    def augmentation_images(self, augm: bool = False) -> A.Compose:
        """
        This function performs data augmentation for a spectogram.
        :return: augment data
        """
        if augm is True:
            aug = A.Compose([
                FreqMask(p=0.3),
                TimeMask(p=0.3),
                A.Cutout(p=0.4, num_holes=12, max_h_size=6, max_w_size=6),
                A.Resize(height=self.image_shape[1], width=self.image_shape[2])
            ])
        else:
            aug = A.Compose([
                A.Resize(height=self.image_shape[1], width=self.image_shape[2])
            ])

        return aug

    def augmentation_audio(self, augm: bool = False) -> A.Compose:
        """
        This function performs data augmentation for an audio.
        """
        if augm is True:
            aug = A.Compose([
                Resample(always_apply=True),
                StretchAudio(p=0.2),
                PitchShift(p=0.2),
                SpeedTuning(p=0.2),
                MakeQuiter(p=0.5, quiter=None),
                PadTranc(always_apply=True),
                PolarityInversion(p=0.2),
                Gain(p=0.2),
                AddNoise(p=0.2)
            ])

        else:
            aug = A.Compose([
                            Resample(always_apply=True),
                            PadTranc(always_apply=True)
            ])

        return aug
