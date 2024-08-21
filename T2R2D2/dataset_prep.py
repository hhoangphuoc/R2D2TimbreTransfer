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
# import tensorflow_io as tfio
from tqdm import tqdm
import params.audio_params as aprs #parameters for audio processing
import params.model_params as mprs #parameters for the model
from utils.audio_utils import calculate_spectrogram, normalise_mel, read_audio, norm_audio_tensor, denorm_audio_tensor, normalise_spec_shape, convert_specs_to_tensor
import argparse

################################################
# Audio to Mel Spectrogram Processing Functions
################################################

def audio_to_mel(
    audio_path, 
    sample_rate=aprs.SAMPLE_RATE, 
    n_mels=aprs.N_MEL_CHANNELS, 
    do_normalise=True,
    mel_spec_shape=aprs.MEL_SPEC_TARGET_SHAPE
):
    """
    Convert audio to mel spectrogram
    and normalise it
    """
    # audio = load_audio(audio_path=audio_path, resample=True, sample_rate=sample_rate) #load audio

    #change to read_audio because using tensor
    audio = read_audio(audio_path=audio_path, resample=True, sample_rate=sample_rate)

    if do_normalise:
        audio, max_val, min_val = norm_audio_tensor(audio)
    
    #squeeze the audio tensor to 1D
    audio = tf.expand_dims(tf.squeeze(audio), axis=0)

    mel_spec = calculate_spectrogram(audio, n_mels=n_mels)
    mel_spec = normalise_spec_shape(mel_spec, target_shape=mel_spec_shape)

    if mel_spec.shape != mel_spec_shape:
        raise ValueError(f"Mel spectrogram shape is not fit {aprs.MEL_SPEC_TARGET_SHAPE}")

    return mel_spec

def generate_mel_specs(
    audio_files_path, 
    sample_rate=aprs.SAMPLE_RATE, 
    n_mels=aprs.N_MEL_CHANNELS, 
    do_normalise=True,
    mel_spec_shape=aprs.MEL_SPEC_TARGET_SHAPE
):
    """
    Expected to generate a list of mel-spectrograms
    from the list of audio files, given by the path to the audio files (i.e. the audio_files_path)
    Input: ['/path/to/target1.wav', '/path/to/target2.wav',...]
    Output: [tgt_spec1, tgt_spec2, ...]
    """
    specs = []
    for audio_path in tqdm(audio_files_path, desc="Generating mel spectrograms"):
        if audio_path.endswith(".wav"):
            try:
                #convert audio to mel spectrogram
                mel_spec = audio_to_mel(audio_path, sample_rate, n_mels, do_normalise)
            
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

    return np.array(specs) #return the numpy array of mel-spectrograms

# def prepare_spectrograms(data_path="./data_specs/", tgt_timbre="r2d2", cond_timbre="vn", sr=aprs.SAMPLE_RATE, n_mel_channels=aprs.N_MEL_CHANNELS, specs_config=False, do_normalise=True):
#     """
#     Prepare the dataset from the audio files to mel-spectrograms 
#     and save them to the dataset folder
#     This is used to convert audio files from .wav in the dataset folder to mel-spectrograms
#     and return the combined mel-spectrograms of both target and condition timbres
#     """
#     # Get the list of audio files
#     tgt_specs = generate_mel_specs(root_path==data_path, timbre_name=tgt_timbre, sample_rate=sr, n_mels=n_mel_channels, do_normalise=do_normalise)
#     cond_specs = generate_mel_specs(root_path==data_path, timbre_name=cond_timbre, sample_rate=sr, n_mels=n_mel_channels, do_normalise=do_normalise)

#     # Convert to Tensorflow Tensors
#     tgt_specs = convert_specs_to_tensor(tgt_specs)
#     cond_specs = convert_specs_to_tensor(cond_specs)

#     if do_normalise:
#         tgt_specs, _, _ = norm_audio_tensor(tgt_specs)
#         cond_specs, _, _ = norm_audio_tensor(cond_specs)

#     #combine the two spectrograms into one
#     return tf.concat([tgt_specs, cond_specs], axis=-1) #TODO - check if return 2 specs seperately better or not?

    # return tgt_specs, cond_specs
def prepare_pair_specs(
    pair_audio_paths, 
    sr=aprs.SAMPLE_RATE, 
    n_mel_channels=aprs.N_MEL_CHANNELS, 
    do_normalise=True,
    mel_spec_shape=aprs.MEL_SPEC_TARGET_SHAPE
):
    """
    Prepare the mel-spectrograms from the pairs of target and condition audio files
    Input: ['/path/to/target1.wav', '/path/to/condition1.wav']
    Output: [tgt_spec, cond_spec]
    """
    tgt_audio_path = pair_audio_paths[0]
    cond_audio_path = pair_audio_paths[1]

    print("Target audio files:", tgt_audio_path)
    print("Conditioning audio files:", cond_audio_path)
    
    try:
        tgt_spec = audio_to_mel(
            tgt_audio_path, 
            sample_rate=sr, 
            n_mels=n_mel_channels, 
            do_normalise=do_normalise,
            mel_spec_shape=mel_spec_shape
        )
    except Exception as e:
        print(f"Error processing target audio file {tgt_audio_path}: {str(e)}")
        return None

    try:
        cond_spec = audio_to_mel(
            cond_audio_path, 
            sample_rate=sr, 
            n_mels=n_mel_channels, 
            do_normalise=do_normalise, 
            mel_spec_shape=mel_spec_shape
        )
    except Exception as e:
        print(f"Error processing condition audio file {cond_audio_path}: {str(e)}")
        return None

    #convert to tensor
    tgt_spec = tf.convert_to_tensor(tgt_spec)
    cond_spec = tf.convert_to_tensor(cond_spec)

    if do_normalise:
        tgt_spec, _, _ = norm_audio_tensor(tgt_spec)
        cond_spec, _, _ = norm_audio_tensor(cond_spec)
    
    return tf.concat([tgt_spec, cond_spec], axis=-1)

#------------------------------------------------

################################
# Dataset Preparation Functions
################################

def load_all_audio_files(data_path="./data_specs/", tgt_timbre="r2d2", cond_timbre="vn"):
    """
    Return: list of paths to all audio files of pairs: [target files, condition files]
    """
    tgt_data_path = os.path.join(data_path, tgt_timbre) # ./data_specs/r2d2
    cond_data_path = os.path.join(data_path, cond_timbre) # ./data_specs/vn

    tgt_audio_files = os.listdir(tgt_data_path) # all r2d2 audio files
    cond_audio_files = os.listdir(cond_data_path) # all vn audio files 

    tgt_audio_files = [f for f in tgt_audio_files if f.endswith('.wav')]
    cond_audio_files = [f for f in cond_audio_files if f.endswith('.wav')]

    print(f"Found {len(tgt_audio_files)} target audio files and {len(cond_audio_files)} conditioning audio files.")

    tgt_audio_files.sort()
    cond_audio_files.sort()

    all_tracks = [[os.path.join(tgt_data_path, t), os.path.join(cond_data_path, f)] for t, f in zip(tgt_audio_files, cond_audio_files)]
    print(f"All tracks: {all_tracks}")
    return all_tracks

def split_train_val(all_audio_paths, val_perc=0.2):
    n_train = len(all_audio_paths) - int(np.floor(val_perc * len(all_audio_paths)))
    
    rng = np.random.default_rng(12345)
    idxs = rng.choice(len(all_audio_paths), len(all_audio_paths), False)
    train_idxs = idxs[:n_train]
    val_idxs = idxs[n_train:]

    train_paths = np.array(all_audio_paths)[train_idxs].tolist()
    val_paths = np.array(all_audio_paths)[val_idxs].tolist()
    return train_paths, val_paths

def create_tf_dataset(audio_paths, batch_size=mprs.BATCH_SIZE, dataset_reps=mprs.DATASET_REPETITIONS, training=True):
    # slice the audio_paths into seperate arrays of [target_audio_path, condition_audio_path]
    dataset = tf.data.Dataset.from_tensor_slices(audio_paths)
    print(dataset.as_numpy_iterator())
    # if training:
    #     dataset = dataset.shuffle(buffer_size=1000)

    features_dataset = dataset.map(prepare_pair_specs) #map the prepare_pair_specs function to the dataset

    if training:
        features_dataset = features_dataset.repeat(dataset_reps)
    else:
        features_dataset = features_dataset.cache().repeat(int(2*dataset_reps))
    
    features_dataset = features_dataset.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # if training:
    #     # dataset = dataset.shuffle(buffer_size=1000)
    # #    features_dataset = dataset.map(prepare_spectrograms).repeat(dataset_repetitions).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
    #     features_dataset = dataset.map(prepare_specs).repeat(dataset_reps).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # else:
    #     # features_dataset = dataset.map(prepare_spectrograms).cache().repeat(int(2*dataset_repetitions)).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    #     features_dataset = dataset.map(prepare_specs).cache().repeat(int(2*dataset_reps)).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return features_dataset

#------------------------------------------------

# def create_tf_dataset(tgt_specs, cond_specs, batch_size=mprs.BATCH_SIZE, training=True):
#     dataset = tf.data.Dataset.from_tensor_slices((tgt_specs, cond_specs))
#     if training:
#         dataset = dataset.shuffle(buffer_size=1000)
#     dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
#     return dataset

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

    # tgt_specs, cond_specs = prepare_spectrograms(
    #     data_path=args.data_path, 
    #     tgt_timbre=args.tgt_timbre, 
    #     cond_timbre=args.cond_timbre, 
    #     sr=args.sr, 
    #     n_mel_channels=args.n_mel_channels
    # )

    specs = prepare_spectrograms(
        data_path=args.data_path, 
        tgt_timbre=args.tgt_timbre, 
        cond_timbre=args.cond_timbre, 
        sr=args.sr, 
        n_mel_channels=args.n_mel_channels
    )

    # # Save the dataset to the dataset folder
    # np.save(os.path.join(args.data_path, args.tgt_timbre, 'tgt_specs.npy'), tgt_specs)
    # np.save(os.path.join(args.data_path, args.cond_timbre, 'cond_specs.npy'), cond_specs)

if __name__ == "__main__":
    main()

 