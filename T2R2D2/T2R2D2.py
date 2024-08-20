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
from preprocess.dataset_prep import prepare_spectrograms

def prepare_data(data_path, tgt_timbre, cond_timbre):
    # load the spectrograms from the dataset
    tgt_data_path = os.path.join(args.data_path, tgt_timbre)
    cond_data_path = os.path.join(args.data_path, cond_timbre)
    tgt_specs_path = os.path.join(tgt_data_path, 'tgt_specs.npy')
    cond_specs_path = os.path.join(cond_data_path, 'cond_specs.npy')

    tgt_specs = None
    cond_specs = None
    
    #check if the dataset is already preprocessed
    if not os.path.exists(tgt_specs_path) and not os.path.exists(cond_specs_path):
        print("Dataset not preprocessed, preprocessing...")
        tgt_specs, cond_specs = prepare_spectrograms(tgt_data_path, cond_data_path)
    else:
        print("Loading spectrograms...")
        tgt_specs = np.load(tgt_specs_path)
        cond_specs = np.load(cond_specs_path)

    val_perc = 0.2
    n_samples = len(tgt_specs)
    n_train = int(n_samples * (1 - val_perc))
    
    rng = np.random.default_rng(12345)
    idxs = rng.permutation(n_samples)
    train_idxs = idxs[:n_train]
    val_idxs = idxs[n_train:]

    train_tgt_specs = tgt_specs[train_idxs]
    train_cond_specs = cond_specs[train_idxs]
    val_tgt_specs = tgt_specs[val_idxs]
    val_cond_specs = cond_specs[val_idxs]
    
    return train_tgt_specs, train_cond_specs, val_tgt_specs, val_cond_specs
    
    # return tgt_specs, cond_specs

def create_tf_dataset(tgt_specs, cond_specs, batch_size, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((tgt_specs, cond_specs))
    if training:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def train_model(train_tgt_specs, train_cond_specs, val_tgt_specs, val_cond_specs, log_path, checkpoint_path, lr, weight_decay, batch_size, epochs):

    # Prepare datasets
    train_dataset = create_tf_dataset(train_tgt_specs, train_cond_specs, batch_size, training=True)
    val_dataset = create_tf_dataset(val_tgt_specs, val_cond_specs, batch_size, training=False)

    # Prepare validation subset for visualization
    val_data = next(iter(val_dataset.take(1)))[:18]  # Adjust the number as need
    print(val_data.shape)

    diff_model = model.DiffusionModel(aprs.MEL_SPEC_NORM_SIZE, mprs.WIDTHS, mprs.BLOCK_DEPTH, val_data, mprs.HAS_ATTENTION, log_path, mprs.BATCH_SIZE)

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
    return diff_model

#----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    #parser for input data
    parser.add_argument('--data_path', type=str, default='./data_specs/' , help='Path to the datasource')
    parser.add_argument('--model_path', type=str, default='./save_models/' , help='Path to the models')
    parser.add_argument('--tgt_timbre', type=str, default='r2d2' , help='Name of Target timbre')
    parser.add_argument('--cond_timbre', type=str, default='vn' , help='Name of Conditioning timbre')

    parser.add_argument('--GPU', type=str, default='0' , help='Number of GPU')
    parser.add_argument('--train', type=bool, default=False , help='Train the model')

    #parser for model parameters
    parser.add_argument('--batch_size', type=int, default=mprs.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=mprs.LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=mprs.WEIGHT_DECAY)
    parser.add_argument('--epochs', type=int, default=mprs.NUM_EPOCHS)
    
    parser.add_argument('--ema', type=float, default=mprs.EMA)

    args = parser.parse_args()
    print(args)

    # tgt_specs, cond_specs = prepare_data(args.data_path, args.tgt_timbre, args.cond_timbre)
    train_tgt_specs, train_cond_specs, val_tgt_specs, val_cond_specs = prepare_data(args.data_path, args.tgt_timbre, args.cond_timbre)

    checkpoint_path = "checkpoints/T2R2D2_"+args.cond_timbre+'_to_'+args.tgt_timbre+'_'+ datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S")

    log_dir = 'logs/'
    log_path = log_dir + 'T2R2D2_' + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + '5000__'+args.cond_timbre+'_to_'+args.tgt_timbre

    
    if args.train:
        # trained_model = train_model(tgt_specs, cond_specs, log_path, checkpoint_path, args.lr, args.weight_decay, args.batch_size, args.epochs)
        # trained_model.save_weights(os.path.join(args.model_path, 'T2R2D2_diff_model.h5'))
        trained_model = train_model(train_tgt_specs, train_cond_specs, val_tgt_specs, val_cond_specs, 
                                    log_path, checkpoint_path, args.lr, args.weight_decay, args.batch_size, args.epochs)
        trained_model.save_weights(os.path.join(args.model_path, 'T2R2D2_diff_model.h5'))
    else:
        diff_model.load_weights(os.path.join(args.data_path, 'T2R2D2_diff_model.h5'))


if __name__ == "__main__":
    main()
