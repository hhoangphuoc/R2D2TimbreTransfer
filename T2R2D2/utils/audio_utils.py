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
import params.audio_params as params

# module = hub.KerasLayer('https://www.kaggle.com/models/google/soundstream/TensorFlow2/mel-decoder-music/1')

MEL_BASIS = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=params.N_MEL_CHANNELS,
    num_spectrogram_bins=params.N_FFT // 2 + 1,
    sample_rate=params.SAMPLE_RATE,
    lower_edge_hertz=params.MEL_FMIN,
    upper_edge_hertz=params.MEL_FMAX)

def calculate_spectrogram(samples, n_mels=params.N_MEL_CHANNELS):
  """Calculate mel spectrogram using the parameters the model expects."""
  fft = tf.signal.stft(
      samples,
      frame_length=params.WIN_LENGTH,
      frame_step=params.HOP_LENGTH,
      fft_length=params.N_FFT,
      window_fn=tf.signal.hann_window,
      pad_end=True)
  fft_modulus = tf.abs(fft)

  output = tf.matmul(fft_modulus, MEL_BASIS)

  output = tf.clip_by_value(
      output,
      clip_value_min=params.CLIP_VALUE_MIN,
      clip_value_max=params.CLIP_VALUE_MAX)
  output = tf.math.log(output)
  return output

#-------------------------------------------------------------------------
def load_audio(audio_path, resample=True, sample_rate=params.SAMPLE_RATE):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    if resample:
        audio = librosa.resample(audio, sr, sample_rate)
    return audio

def audio_to_mel(audio, sample_rate=params.SAMPLE_RATE, n_mels=params.N_MEL_CHANNELS, n_fft=params.N_FFT, hop_length=params.HOP_LENGTH):
    """
    Convert audio to mel spectrogram using librosa
    """
    mel = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec = librosa.power_to_db(mel, ref=np.max) # Convert to dB

    return mel_spec

def normalise_mel(mel_spec):
    """Normalize the Mel-spectrogram to a range [0, 1]."""
    mel_spec = (mel_spec - np.min(mel_spec)) / (np.max(mel_spec) - np.min(mel_spec))
    return mel_spec

def mel_to_audio(mel, sample_rate=params.SAMPLE_RATE, n_fft=params.N_FFT, hop_length=params.HOP_LENGTH):
    """
    Convert mel spectrogram to audio using librosa
    """
    mel = librosa.db_to_power(mel)
    audio = librosa.feature.inverse.mel_to_audio(mel, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    return audio

def preprocess_audio(audio_path, sample_rate=params.SAMPLE_RATE, n_mels=params.N_MEL_CHANNELS):
    """
    Convert audio to mel spectrogram
    and normalise it
    """
    audio = load_audio(audio_path=audio_path, resample=True, sample_rate=sample_rate) #load audio
    # mel_spec = audio_to_mel(audio, sample_rate, n_mels, n_fft, hop_length) #convert audio to mel spectrogram 
    #TODO- CAN BE CHANGED TO USE calculate_spectrogram
    mel_spec = calculate_spectrogram(audio, n_mels=n_mels)
    normalised_mel_spec = normalise_mel(mel_spec) #normalise mel spectrogram
    return normalised_mel_spec

def preprocess_dataset(dataset_path=params.DATASET_PATH, sample_rate=params.SAMPLE_RATE, n_mels=params.N_MEL_CHANNELS):
    """
    Generate dataset of audio files to list of mel spectrograms
    """
    dataset = []
    for audio_path in os.listdir(dataset_path):
        audio_path = os.path.join(dataset_path, audio_path)
        mel_spec = preprocess_audio(audio_path, sample_rate, n_mels)
        dataset.append(mel_spec)
    # print(f"Dataset shape: {np.array(dataset).shape}")
    return np.array(dataset) #return the dataset as a numpy array

# def read_audio(audio_path,resample=True, sample_rate=params.SAMPLE_RATE):
#     audio_bin = tf.io.read_file(audio_path)
#     audio, sample_rate = tf.audio.decode_wav(audio_bin)
#     if resample:
#         audio = tfio.audio.resample(audio, rate_in=tf.cast(sample_rate,tf.int64), rate_out=SAMPLE_RATE, name=None)
#     return audio

# def save_audio(audio, path, sample_rate=params.SAMPLE_RATE):
#     audio = tf.cast(audio, tf.float32)
#     audio = tfio.audio.resample(audio, rate_in=sample_rate, rate_out=sample_rate, name=None)
#     audio = tf.cast(audio, tf.int16)
#     audio = tf.audio.encode_wav(audio, sample_rate)
#     tf.io.write_file(path, audio)

def plot_audio(audio):
    """
    Plot the audio in the waveform
    """
    plt.plot(audio)
    plt.show()

def plot_spectrogram(mel_spec):
    plt.imshow(mel_spec, aspect='auto', origin='lower')
    plt.colorbar()
    plt.show()

def plot_spectrogram_from_audio(audio):
    """
    Plot the spectrogram of the audio
    """
    spectrogram = calculate_spectrogram(audio)
    plot_spectrogram(spectrogram)

    