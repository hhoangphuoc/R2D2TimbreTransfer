#TODO: The main file to execute the model
import os
import tqdm
import numpy as np
import tensorflow as tf
import datetime

from tensorflow import keras
import argparse

import model
import params.audio_params as aprs
import params.model_params as mprs
from dataset_prep import load_all_audio_files, split_train_val, create_tf_dataset

def prepare_data(data_path, tgt_timbre, cond_timbre):
    """
    Prepares the data for the model
    """
    # load the spectrograms from the dataset
    tgt_data_path = os.path.join(data_path, tgt_timbre) #path to the target timbre .wav files
    cond_data_path = os.path.join(data_path, cond_timbre) #path to the conditioning timbre .wav files

    # tgt_specs_path = os.path.join(tgt_data_path, 'tgt_specs.npy') #path to the target timbre mel-spectrograms
    # cond_specs_path = os.path.join(cond_data_path, 'cond_specs.npy') #path to the conditioning timbre mel-spectrograms
    # tgt_specs = None
    # cond_specs = None

    all_audio_paths = load_all_audio_files(data_path=data_path, tgt_timbre=tgt_timbre, cond_timbre=cond_timbre)
    
    # #check if the dataset is already preprocessed
    # if not os.path.exists(tgt_specs_path) and not os.path.exists(cond_specs_path):
    #     print("Dataset not preprocessed, preprocessing...")
    #     tgt_specs, cond_specs = prepare_spectrograms(tgt_data_path, cond_data_path)
    # else:
    #     print("Loading spectrograms...")
    #     tgt_specs = np.load(tgt_specs_path)
    #     cond_specs = np.load(cond_specs_path)

    # split the train and val path before converting to spectrograms
    train_paths, val_paths = split_train_val(all_audio_paths)
    print("Train paths:", train_paths)
    print("Val paths:", val_paths)
    # convert the train and val path to spectrograms
    train_specs = create_tf_dataset(train_paths)
    val_specs = create_tf_dataset(val_paths)

    # train_tgt_specs = tgt_specs[train_idxs]
    # train_cond_specs = cond_specs[train_idxs]
    # val_tgt_specs = tgt_specs[val_idxs]
    # val_cond_specs = cond_specs[val_idxs]
    
    return train_specs, val_specs


def train_model(
    train_specs,
    val_specs,
    model_path,
    log_path, 
    checkpoint_path, 
    lr, 
    weight_decay, 
    batch_size, 
    epochs, 
    train=True
):
    # Create and compile the model
    first = True
    for s in val_specs:
        if first:
            val_data = s
            first = False
        else:
            val_data = tf.concat([val_data, s], axis=0)
    val_data = val_data[:mprs.BATCH_SIZE]
    print(val_data.shape)

    diff_model = model.DiffusionModel(
        aprs.MEL_SPEC_NORM_SIZE, 
        mprs.WIDTHS, 
        mprs.BLOCK_DEPTH, 
        val_data, 
        mprs.HAS_ATTENTION, 
        log_path, 
        mprs.BATCH_SIZE
    )
    diff_model.network.summary()

    if train:
        # train the model
        diff_model.compile(
                optimizer=keras.optimizers.experimental.AdamW(
                    learning_rate= lr, weight_decay=weight_decay
                ),
                loss=keras.losses.mean_absolute_error,
            )

        # save the best model based on the validation KID metric
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                monitor="val_n_loss",
                mode="min",
                save_best_only=True,
            )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/T2R2D2_"+args.cond_timbre+'_to_'+args.tgt_timbre+datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S"))
        
        # calculate mean and variance of training dataset for normalization
        diff_model.fit(
            train_dataset,
            epochs=epochs, #default 5000
            validation_data=val_dataset,
            callbacks=[
                keras.callbacks.LambdaCallback(on_epoch_end=diff_model.plot_images),
                checkpoint_callback,
                tensorboard_callback,
            ],
        )
    else:
        diff_model.load_weights(os.path.join(checkpoint_path, 'T2R2D2_diff_model.h5'))
    
    return diff_model

#----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    #parser for input data
    parser.add_argument('--data_path', type=str, default='./data_specs/' , help='Path to the .wav datasource')
    parser.add_argument('--tgt_timbre', type=str, default='r2d2' , help='Name of Target timbre')
    parser.add_argument('--cond_timbre', type=str, default='vn' , help='Name of Conditioning timbre')

    parser.add_argument('--model_path', type=str, default='./save_models/' , help='Path to the models')
    parser.add_argument('--GPU', type=str, default='0' , help='Number of GPU')
    parser.add_argument('--train', type=bool, default=True , help='Train the model')

    #model config
    parser.add_argument('--batch_size', type=int, default=mprs.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=mprs.LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=mprs.WEIGHT_DECAY)
    parser.add_argument('--epochs', type=int, default=mprs.NUM_EPOCHS)
    
    parser.add_argument('--ema', type=float, default=mprs.EMA)

    args = parser.parse_args()
    print(args)

    #prepare data
    train_specs, val_specs = prepare_data(args.data_path, args.tgt_timbre, args.cond_timbre) #spectrograms for training and validation

    #create checkpoint and log paths
    checkpoint_path = "checkpoints/T2R2D2_"+args.cond_timbre+'_to_'+args.tgt_timbre+'_'+ datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S")
    log_dir = 'logs/'
    log_path = log_dir + 'T2R2D2_' + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + str(args.epochs) + '_'+args.cond_timbre+'_to_'+args.tgt_timbre

    trained_model = train_model(
        train_specs=train_specs,
        val_specs=val_specs,
        model_path=args.model_path,
        log_path=log_path, 
        checkpoint_path=checkpoint_path, 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        batch_size=args.batch_size, 
        epochs=args.epochs,
        train=args.train
    ) #either train or load the model

    trained_model.save_weights(os.path.join(args.model_path, 'T2R2D2_diff_model.h5'))

if __name__ == "__main__":
    main()
