
# NOTE: The implementation is taken from the SoundStream implementation by Google
# This used to convert the audio waveform to mel spectrogram

import math
import tensorflow as tf
import tensorflow_hub as hub

from params.audio_params import *

module = hub.KerasLayer('https://www.kaggle.com/models/google/soundstream/TensorFlow2/mel-decoder-music/1')

#Simple function to calculate the mel spectrogram
# taken from: https://www.kaggle.com/models/google/soundstream/tensorFlow2/mel-decoder-music/1

#------------------------------------------------------------
# Class-based Implementation
# This code was taken from audio_codecs.py in Music-Spectrogram-Diffusion implementation
# and rewritten in TensorFlow
#------------------------------------------------------------

class Audio2Mel(tf.keras.Model):
  """Audio2Mel frontend."""

  def __init__(self,
               sample_rate=SAMPLE_RATE,
               n_fft=N_FFT,
               hop_length=HOP_LENGTH,
               win_length=WIN_LENGTH,
               n_mel_channels=N_MEL_CHANNELS,
               drop_dc=True,
               mel_fmin=MEL_FMIN,
               mel_fmax=MEL_FMAX,
               clip_value_min=CLIP_VALUE_MIN,
               clip_value_max=CLIP_VALUE_MAX,
               log_amplitude=True,
               **kwargs):
    """Builds the Audio2Mel frontend.

    Args:
      sample_rate: sampling rate. Need to be provided if `n_mel_channels` is not
        `None`.
      n_fft: length of the FFT, in samples.
      hop_length: length of the hop size, in samples.
      win_length: length of the window, in samples.
      n_mel_channels: number of mel channels. If set to None, will return the
        full magnitude STFT.
      drop_dc: if `True`, drop the STFT DC coefficient. Used only when
        n_mel_channels is `None`.
      mel_fmin: lowest frequency in the mel filterbank in Hz.
      mel_fmax: highest frequency in the mel filterbank in Hz.
      clip_value_min: minimal value of the (mel)-spectrogram before log. Used
        only when `log_amplitude` is `True`.
      clip_value_max: maximal value of the (mel)-spectrogram before log. Used
        only when `log_amplitude` is `True`.
      log_amplitude: if `True` apply log amplitude scaling.
      **kwargs: Additional keyword arguments for tf.keras.Model.
    """

    super().__init__(**kwargs)

    self._n_fft = n_fft
    self._hop_length = hop_length
    self._win_length = win_length
    self._sample_rate = sample_rate
    self._clip_value_min = clip_value_min
    self._clip_value_max = clip_value_max
    self._log_amplitude = log_amplitude
    self._n_mel_channels = n_mel_channels
    self._drop_dc = drop_dc

    if n_mel_channels is None:
      self.mel_basis = None
    else:
      if sample_rate is None:
        raise ValueError(
            '`sample_rate` must be provided when `n_mel_channels` is not `None`')
      if mel_fmax is None:
        mel_fmax = sample_rate // 2

      self.mel_basis = tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins=n_mel_channels,
          num_spectrogram_bins=n_fft // 2 + 1,
          sample_rate=sample_rate,
          lower_edge_hertz=mel_fmin,
          upper_edge_hertz=mel_fmax) #configuration for BASIC mel spectrogram

  def call(self, audio, training=False):
    """Computes the mel spectrogram of the input audio samples.

    Coefficients are clipped before log compression to avoid log(0) and large
    coefficients.

    Args:
      audio: input sample of shape (batch_size, num_samples).
      training: flag to distinguish between train and test time behavior.

    Returns:
      Mel spectrogram of shape (batch_size, time_frames, freq_bins).
    """
    fft = tf.signal.stft(
        audio,
        frame_length=self._win_length,
        frame_step=self._hop_length,
        fft_length=self._n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=True)
    fft_modulus = tf.abs(fft)

    # Compute the basic mel spectrogram.
    if self.mel_basis is not None:
      output = tf.matmul(fft_modulus, self.mel_basis)
    else:
      output = fft_modulus
      if self._drop_dc:
        output = output[:, :, 1:]

    # Apply log amplitude scaling if exist
    if self._log_amplitude:
      output = tf.clip_by_value(
          output,
          clip_value_min=self._clip_value_min,
          clip_value_max=self._clip_value_max)
      output = tf.math.log(output)

    return output

#---------------------------------------------------------------------




