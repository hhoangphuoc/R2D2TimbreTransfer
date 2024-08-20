# TODO: THIS FILE IS USED FOR THE PREPROCESSING PHASE OF THE CHOSEN DATASEt
# So it can input to the model

# Calculate the spectrogram of the audio file
# and Reshape the spectrogram to (128, 128, 1) - 1 channel of image 128x128
# The expected output of this file is two numpy arrays corresponding to two dataset: target timbre and condition timbre, respectively
# The target timbre is the timbre that we want to convert the condition timbre to
# The condition timbre is the timbre that we want to convert to the target timbre
# The two numpy arrays have the shape (number of samples, 128, 128, 1)
import tensorflow as tf
import numpy as np
import os
import tensorflow_io as tfio
from tqdm import tqdm
import params.audio_params as aprs #parameters for audio processing
import params.model_params as mprs #parameters for the model
from utils.audio_utils import calculate_spectrogram, preprocess_dataset
import argparse

def convert_specs_to_tensor(specs):
    specs = tf.convert_to_tensor(specs, dtype=tf.float32)
    specs = tf.expand_dims(specs, axis=-1)
    return specs

def prepare_spectrograms(tgt_dataset="../datasets/r2d2", cond_dataset="../datasets/vn", sr=aprs.SAMPLE_RATE, n_mel_channels=aprs.N_MEL_CHANNELS, specs_config=False):
    """
    Prepare the dataset from the audio files to mel-spectrograms 
    and save them to the dataset folder
    """
    # Get the list of audio files
    tgt_specs = preprocess_dataset(tgt_dataset, sr, n_mel_channels)
    cond_specs = preprocess_dataset(cond_dataset, sr, n_mel_channels)

    # Convert to Tensorflow Tensors
    tgt_specs = convert_specs_to_tensor(tgt_specs)
    cond_specs = convert_specs_to_tensor(cond_specs)

    # tgt_spectrograms = tf.convert_to_tensor(tgt_specs, dtype=tf.float32)
    # tgt_spectrograms = tf.expand_dims(tgt_spectrograms, axis=-1)

    # cond_spectrograms = tf.convert_to_tensor(cond_specs, dtype=tf.float32)
    # cond_spectrograms = tf.expand_dims(cond_spectrograms, axis=-1)

    return tgt_spectrograms, cond_spectrograms #expected output shape (number of samples, 128, 128, 1)


def main():
    #TODO: Add parser for the input data to provide
    # additional configuration for the mel-spectrogram
    """
    THIS FILE ONLY USE FOR PREPROCESSING TASK
    WHICH WILL GENERATE THE DATASET OF MEL-SPECTROGRAMS FROM THE AUDIO FILES
    AND SAVE THEM TO THE DATASET FOLDER
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data_specs/' , help='Path to the datasource')
    parser.add_argument('--tgt_timbre', type=str, default='r2d2' , help='Name of Target timbre')
    parser.add_argument('--cond_timbre', type=str, default='vn' , help='Name of Conditioning timbre')
    parser.add_argument('--sr', type=int, default=aprs.SAMPLE_RATE , help='Sample rate')
    parser.add_argument('--n_mel_channels', type=int, default=aprs.N_MEL_CHANNELS , help='Number of mel channels')

    # parser for the configuration of mel-spectrogram
    # FIXME - to be considered
    # parser.add_argument('--n_fft', type=int, default=aprs.N_FFT , help='Number of FFT')
    # parser.add_argument('--hop_length', type=int, default=aprs.HOP_LENGTH , help='Hop length')
    # parser.add_argument('--win_length', type=int, default=aprs.WIN_LENGTH , help='Window length')
    # parser.add_argument('--fmin', type=float, default=aprs.MEL_FMIN , help='Minimum frequency')
    # parser.add_argument('--fmax', type=float, default=aprs.MEL_FMAX , help='Maximum frequency')

    args = parser.parse_args()

    tgt_dataset = os.path.join(args.data_path, args.tgt_timbre)
    cond_dataset = os.path.join(args.data_path, args.cond_timbre)
    tgt_specs, cond_specs = prepare_spectrograms(tgt_dataset, cond_dataset, args.sr, args.n_mel_channels)

    # Save the dataset to the dataset folder
    np.save(os.path.join(args.data_path, args.tgt_timbre, 'tgt_specs.npy'), tgt_specs)
    np.save(os.path.join(args.data_path, args.cond_timbre, 'cond_specs.npy'), cond_specs)

if __name__ == "__main__":
    main()

 