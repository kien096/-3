import random
from typing import Union, Tuple

import cv2
import librosa
import numpy as np
from albumentations.core.transforms_interface import BasicTransform

from config import AUDIO_LEN, FREQUENCY_DISCREDITING


class AudioTransform(BasicTransform):
    """Transform for Audio task"""

    @property
    def targets(self):
        return {"data": self.apply, "image": self.apply}

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params


class PadTranc(AudioTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5, max_ms: int = AUDIO_LEN):
        """
        This class results in the total length according to the max_ms parameter.
        :param always_apply: always use or not.
        :param p: the probability of use is from 0 to 1.
        :param max_ms: output length of the audio signal.
        """
        super(PadTranc, self).__init__(always_apply, p)
        self.max_ms = max_ms

    def apply(self, data, **params):
        sig, sr = data
        sig_len = len(sig)
        max_len = sr // 1000 * self.max_ms

        if sig_len > max_len:
            pad_begin_len = random.randint(0, sig_len - max_len)
            return sig[pad_begin_len:pad_begin_len+max_len], sr

        elif sig_len < max_len:
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = np.random.uniform(-0.001, 0.001, pad_begin_len)
            pad_end = np.random.uniform(-0.001, 0.001, pad_end_len)
            sig = np.concatenate((pad_begin, sig, pad_end))
            return sig, sr
        else:
            return sig, sr


class Resample(AudioTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5, new_sr: int = FREQUENCY_DISCREDITING):
        """
        Changing the sampling rate if it does not correspond to the desired one.
        :param always_apply: always use or not.
        :param p: the probability of use is from 0 to 1.
        :param new_sr: new frequency of discrediting.
        """
        super(Resample, self).__init__(always_apply, p)
        self.new_sr = new_sr

    def apply(self, data, **params):
        sig, sr = data
        if sr == self.new_sr:
            return sig, sr
        resig = librosa.resample(y=sig, orig_sr=sr, target_sr=self.new_sr)
        return resig, self.new_sr


class PitchShift(AudioTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5, n_steps: Union[int, None] = None):
        """
        Change the voice.
        :param always_apply: always use or not.
        :param p: the probability of use is from 0 to 1.
        :param n_steps: a specific or random parameter for pitchShift
        """
        super(PitchShift, self).__init__(always_apply, p)
        self.n_steps = n_steps

    def apply(self, data, **params):
        if self.n_steps is None:
            self.n_steps = np.random.randint(-7, 7, dtype=int)
        wav, sr = data

        return librosa.effects.pitch_shift(wav, sr=sr, n_steps=self.n_steps), sr


class StretchAudio(AudioTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5, rate: Union[float, None] = None):
        """
        :param always_apply: always use or not.
        :param p: the probability of use is from 0 to 1.
        :param rate:
        """
        super(StretchAudio, self).__init__(always_apply, p)
        self.rate = rate

    def apply(self, data, **params):
        if self.rate is None:
            self.rate = np.random.uniform(0.8, 1.3)

        wav, sr = data
        wav = librosa.effects.time_stretch(wav, self.rate)
        return wav, sr


class FreqMask(AudioTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5, F: int = 30, num_masks: int = 1,
                 replace_with_zero: bool = False):
        """
        Applying a mask to spectogram along the X axis.
        :param always_apply: always use or not.
        :param p: the probability of use is from 0 to 1.
        :param F:
        :param num_masks: number of generated masks.
        :param replace_with_zero: change the values in the mask to zero.
        """
        super(FreqMask, self).__init__(always_apply, p)
        self.F = F
        self.num_masks = num_masks
        self.replace_with_zero = replace_with_zero

    def apply(self, image, **params):
        cloned = image.copy()
        num_mel_channels = cloned.shape[0]

        for i in range(0, self.num_masks):
            f = random.randrange(0, self.F)
            f_zero = random.randrange(0, num_mel_channels - f)

            if f_zero == f_zero + f:
                return cloned

            mask_end = random.randrange(f_zero, f_zero + f)
            if self.replace_with_zero:
                cloned[f_zero:mask_end, :] = 0
            else:
                cloned[f_zero:mask_end, :] = cloned.mean()

        return cloned


class TimeMask(AudioTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5, T: int = 40, num_masks: int = 1,
                 replace_with_zero: bool = False):
        """
        Applying a mask to spectogram along the Y axis.
        :param always_apply: always use or not.
        :param p: the probability of use is from 0 to 1.
        :param T:
        :param num_masks: number of generated masks.
        :param replace_with_zero: change the values in the mask to zero.
        """
        super(TimeMask, self).__init__(always_apply, p)
        self.T = T
        self.num_masks = num_masks
        self.replace_with_zero = replace_with_zero

    def apply(self, image, **params):
        cloned = image.copy()
        len_spectro = cloned.shape[1]

        for i in range(0, self.num_masks):
            t = random.randrange(0, self.T)
            t_zero = random.randrange(0, len_spectro - t)

            if t_zero == t_zero + t:
                return cloned

            mask_end = random.randrange(t_zero, t_zero + t)

            if self.replace_with_zero:
                cloned[:, t_zero:mask_end] = 0
            else:
                cloned[:, t_zero:mask_end] = cloned.mean()

        return cloned


class AddNoise(AudioTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        """
        Adding noise ti the audio signal.
        :param always_apply: always use or not.
        :param p: the probability of use is from 0 to 1.
        """
        super(AddNoise, self).__init__(always_apply, p)

    def apply(self, data, **params):
        wav, sr = data
        data_wn = wav + 0.005 * np.random.randn(len(wav))
        return data_wn, sr


class PolarityInversion(AudioTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        """
        :param always_apply: always use or not.
        :param p: the probability of use is from 0 to 1.
        """
        super(PolarityInversion, self).__init__(always_apply, p)

    def apply(self, data, **params):
        wav, sr = data
        return -wav, sr


class SpeedTuning(AudioTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5, speed_rate: Union[Tuple, None] = None):
        """
        Audio speed change, if none, then the value changes within (0.7, 1.3)
        :param always_apply: always use or not.
        :param p: the probability of use is from 0 to 1.
        :param speed_rate:
        """
        super(SpeedTuning, self).__init__(always_apply, p)
        self.speed_rate = speed_rate

    def apply(self, data, **params):

        if self.speed_rate is None:
            self.speed_rate = np.random.uniform(0.7, 1.3)

        wav, sr = data
        audio_speed_tune = cv2.resize(wav, (1, int(len(wav) * self.speed_rate))).squeeze()

        return audio_speed_tune, sr


class Gain(AudioTransform):
    def __init__(self, min_gain_in_db: int = -12, max_gain_in_db: int = 12, always_apply: bool = False, p: float = 0.5):
        """
        Change the volume.
        :param min_gain_in_db:
        :param max_gain_in_db:
        :param always_apply: always use or not.
        :param p: the probability of use is from 0 to 1.
        """
        super(Gain, self).__init__(always_apply, p)
        assert min_gain_in_db <= max_gain_in_db
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db

    def apply(self, data, **args):
        wav, sr = data
        amplitude_ratio = 10**(random.uniform(self.min_gain_in_db, self.max_gain_in_db)/20)
        return wav * amplitude_ratio, sr


class OverlayAudio(AudioTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5, intensity_overlay: Tuple = (3, 7)):
        """
        Adding noise to audio.
        :param always_apply: always use or not.
        :param p: the probability of use is from 0 to 1.
        :param intensity_overlay: intensity of noise volume.
        """
        super(OverlayAudio, self).__init__(always_apply, p)
        self.intensity_overlay = intensity_overlay

    def apply(self, data, **params):

        wav_speech, sr_speech = data[0][0], data[0][1]
        wav_noise, sr_noise = data[1][0], data[1][1]

        max_speech = np.max(wav_speech)
        max_noise = np.max(wav_noise)

        min_speech = np.min(wav_speech)
        min_noise = np.min(wav_noise)

        dif_max = max_noise / max_speech
        dif_min = min_noise / min_speech

        coof = ((abs(dif_max) + abs(dif_min)) / 2) * np.random.randint(self.intensity_overlay[0],
                                                                       self.intensity_overlay[1])

        waw = (wav_speech + (wav_noise / coof)) / 2
        sr = (sr_speech + sr_noise) / 2

        return waw, int(sr)


class MakeQuiter(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, quiter: Union[Tuple, None] = None):
        """
        :param always_apply: always use or not.
        :param p: the probability of use is from 0 to 1.
        """
        super(MakeQuiter, self).__init__(always_apply, p)
        self.quiter = quiter

    def apply(self, data, **params):

        if self.quiter is None:
            self.quiter = np.random.randint(2, 12)

        wav_speech, sr_speech = data
        wav_speech = wav_speech / self.quiter

        return wav_speech, sr_speech
