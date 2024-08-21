#-----------------------------------------
# Audio Utils
# Including functions that processing with audio files
#-----------------------------------------

import os
import tensorflow_io as tfio
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import librosa
import params.audio_params as aprs
from tqdm import tqdm

module = hub.KerasLayer('https://www.kaggle.com/models/google/soundstream/TensorFlow2/mel-decoder-music/1')

MEL_BASIS = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=aprs.N_MEL_CHANNELS,
    num_spectrogram_bins=aprs.N_FFT // 2 + 1,
    sample_rate=aprs.SAMPLE_RATE,
    lower_edge_hertz=aprs.MEL_FMIN,
    upper_edge_hertz=aprs.MEL_FMAX)

def calculate_spectrogram(samples, n_mels=aprs.N_MEL_CHANNELS):
  """Calculate mel spectrogram using the parameters the model expects."""
  fft = tf.signal.stft(
      samples,
      frame_length=aprs.WIN_LENGTH,
      frame_step=aprs.HOP_LENGTH,
      fft_length=aprs.N_FFT,
      window_fn=tf.signal.hann_window,
      pad_end=True)


  fft_modulus = tf.abs(fft)

  output = tf.matmul(fft_modulus, MEL_BASIS)

  output = tf.clip_by_value(
      output,
      clip_value_min=aprs.CLIP_VALUE_MIN,
      clip_value_max=aprs.CLIP_VALUE_MAX
      )

  # Add a small constant to avoid log(0)
  output = tf.math.log(output + 1e-6)
  return output

#-------------------------------------------------------------------------
def load_audio(audio_path, resample=True, sample_rate=aprs.SAMPLE_RATE):
    audio, sr = librosa.load(path=audio_path, sr=sample_rate)
    if resample:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=sample_rate)
    return audio

# def read_audio(audio_path,resample=True, sample_rate=aprs.SAMPLE_RATE):
#     audio_bin = tf.io.read_file(audio_path)
#     audio, sr = tf.audio.decode_wav(audio_bin)
#     if resample:
#         audio = tfio.audio.resample(audio, rate_in=tf.cast(sr,tf.int64), rate_out=sample_rate, name=None)
#     return audio

# def normalise_audio(audio):
#     """Normalize the audio to a range [-1, 1]."""
#     audio = audio - audio.min()
#     audio = audio / audio.max()
#     audio = audio * 2 - 1

#     return audio

def norm_audio_tensor(audio):
    """Normalize the audio in the tensor representation
    that match the range of [-1, 1]
    """
    min_val = tf.math.reduce_min(audio)
    audio = audio - min_val
    max_val = tf.math.reduce_max(audio)
    audio_norm = ((audio/max_val)*2)-1
    return audio_norm, max_val, min_val

def denorm_audio_tensor(audio,max_val,min_val):
    """Denormalize the audio in the tensor representation"""
    return (((audio +1)/2)*max_val + min_val)


def normalise_mel(mel_spec):
    """
    Normalize the Mel-spectrogram to a range [0, 1].
    """
    mel_spec = (mel_spec - np.min(mel_spec)) / (np.max(mel_spec) - np.min(mel_spec))
    return mel_spec

def convert_specs_to_tensor(specs):
    specs = tf.convert_to_tensor(specs, dtype=tf.float32)
    specs = tf.expand_dims(specs, axis=-1)
    return specs

def normalise_spec_shape(mel_spec, target_shape=aprs.MEL_SPEC_TARGET_SHAPE):
    """
    Normalize the shape of the mel spectrogram to standardise input shape of all spectrograms
    and match input shape of the model
    TARGET SHAPE: (128,128,1)
    """
    mel_reshaped = tf.expand_dims(tf.squeeze(mel_spec), axis=-1)
    mel_reshaped = tf.reshape(mel_reshaped, target_shape) #
    return mel_reshaped

def reconstruct_audio(mel_spec, sample_rate=aprs.SAMPLE_RATE, n_fft=aprs.N_FFT, hop_length=aprs.HOP_LENGTH):
    """
    Convert mel spectrogram to audio using SoundStream Decoder
    """
    audio = module(mel_spec)
    return audio

    