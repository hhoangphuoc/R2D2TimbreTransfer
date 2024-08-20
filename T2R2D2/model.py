"""Partially taken the DDIM model implementation from
https://github.com/keras-team/keras-io/blob/master/examples/generative/ddim.py

sinusoidal embedding and U-NET model architecture from
https://github.com/lucacoma/DiffTransfer
"""

import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from keras import layers

from params.model_params import *
from utils.model_utils import plot_to_image, denorm_tensor, kernel_init
from utils.audio_utils import calculate_spectrogram
from preprocess.dataset_prep import convert_specs_to_tensor
from modules.embeddings import sinusoidal_embedding
from modules.blocks import ResidualBlock, DownBlock, UpBlock, AttentionBlock


import io
tf.config.list_physical_devices('GPU')
# # SoundStream Spectrogram Inverter (Stuff stolen from https://storage.googleapis.com/music-synthesis-with-spectrogram-diffusion/index.html) and https://tfhub.dev/google/soundstream/mel/decoder/music/1
# module = hub.KerasLayer('https://tfhub.dev/google/soundstream/mel/decoder/music/1')

do_norm_specs = True
do_normalization = False
# data

# architecture
#-----------------------------------------------------------#
# CONSTRUCTION OF U-NET MODEL
#-----------------------------------------------------------#
def get_network(mel_spec_size, widths, block_depth,has_attention):
    norm_groups=8

    noisy_images = keras.Input(shape=(mel_spec_size[0], mel_spec_size[1], COND_IMG_CHANNELS)) #The conditioning difussion noise image channels is 2
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=mel_spec_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = [] # skip connections for the U-Net (connecting the corresponding downsampling and upsampling block)
    idx = 0

    # Downsampling paths of U-NET
    # except the 512-channel layer
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])
        if has_attention[idx]:
            x = AttentionBlock(width, groups=norm_groups)(x)
        idx = idx +1

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x) #added residual block for each block downsampled
    
    # Add attention block to the 512-channel layer
    x = AttentionBlock(widths[-1], groups=norm_groups)(x) #widths[-1] = 512

    # Upsampling paths of U-NET (except the 512-channel layer)
    idx = len(widths[:-1])-1
    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])
        if has_attention[idx]:
            x = AttentionBlock(width, groups=norm_groups)(x)
        idx = idx -1

    x = layers.Conv2D(N_IMG_CHANNELS, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")

#-----------------------------------------------------------#

##############################
# Diffusion Model Structure  #
##############################

class DiffusionModel(keras.Model):
    def __init__(self, mel_spec_size, widths, block_depth, 
    val_data, 
    has_attention, logdir='logs',batch_size=64): # val_data is BAD and should be removed
        super().__init__()
        if do_normalization:
            self.normalizer = layers.Normalization()
        self.network = get_network(mel_spec_size, widths, block_depth,has_attention)
        self.ema_network = keras.models.clone_model(self.network)
        self.val_data = val_data
        self.mel_spec_size = mel_spec_size
        self.logdir=logdir
        self.summary_writer = tf.summary.create_file_writer(logdir)
        self.batch_size=batch_size

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]#, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        if do_normalization:
            images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return images #tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(MAX_SIGNAL_RATE)
        end_angle = tf.acos(MIN_SIGNAL_RATE)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (tf.expand_dims(noisy_images[:,:,:,0],axis=-1) - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, cond_images, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                tf.concat([noisy_images, cond_images],axis=-1), noise_rates, signal_rates, training=False
            )
            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, cond_images, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, self.mel_spec_size[0], self.mel_spec_size[1], N_IMG_CHANNELS)) #initially generate random noise

        generated_images = self.reverse_diffusion(initial_noise, cond_images, diffusion_steps) # generate image from the random noise

        # change normalisation if it goes wrong
        if do_normalization:
            generated_images = self.denormalize(tf.concat([generated_images,cond_images],axis=-1)) # WRONG NORM
        generated_images = tf.expand_dims(generated_images[:,:,:,0],axis=-1)
        return generated_images

    def generate_fixed_noise(self, cond_images, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(1, self.mel_spec_size[0], self.mel_spec_size[1], N_IMG_CHANNELS))
        initial_noise = tf.tile(initial_noise,tf.constant([num_images,1,1,1]))
        generated_images = self.reverse_diffusion(initial_noise, cond_images, diffusion_steps)

        # change normalisation if it goes wrong
        if do_normalization:
            generated_images = self.denormalize(tf.concat([generated_images,cond_images],axis=-1)) # WRONG NORM
        generated_images = tf.expand_dims(generated_images[:,:,:,0],axis=-1)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        if do_normalization:
            images = self.normalizer(images, training=True)

        des_images = tf.expand_dims(images[:,:,:,0],axis=-1)
        cond_images = tf.expand_dims(images[:,:,:,1],axis=-1)

        noises = tf.random.normal(shape=(self.batch_size, self.mel_spec_size[0], self.mel_spec_size[1], N_IMG_CHANNELS))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * des_images + noise_rates * noises

        # Concatenatioon to condition
        noisy_images = tf.concat([noisy_images, cond_images],axis=-1)

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(des_images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(EMA * ema_weight + (1 - EMA) * weight)

        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        if do_normalization:
            images = self.normalizer(images, training=False)

        des_images = tf.expand_dims(images[:,:,:,0],axis=-1)
        cond_images = tf.expand_dims(images[:,:,:,1],axis=-1)

        noises = tf.random.normal(shape=(self.batch_size, self.mel_spec_size[0], self.mel_spec_size[1], N_IMG_CHANNELS))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * des_images + noise_rates * noises

        noisy_images = tf.concat([noisy_images, cond_images],axis=-1)

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(des_images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=4, num_cols=4):
        if epoch % 10 == 0:
            images = self.val_data
            if do_normalization:
                images = self.normalizer(images, training=False)
            des_images = tf.expand_dims(images[:,:,:,0],axis=-1)
            cond_images = tf.expand_dims(images[:,:,:,1],axis=-1)
            cond_images = cond_images[:18]
            # plot random generated images for visual evaluation of generation quality
            generated_images = self.generate(cond_images,
                num_images=num_rows * num_cols,
                diffusion_steps=PLOT_DIFFUSSION_STEPS,
            )

            spec_figures= plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    plt.subplot(num_rows, num_cols, index + 1)
                    plt.imshow(tf.transpose(generated_images[index],(1,0,2)))
                    plt.gca().invert_yaxis()
                    plt.axis("off")
            plt.tight_layout()

            # Audio
            idx = tf.random.uniform((),0,(num_rows * num_cols),dtype=tf.int32)
            tgt_spec =  tf.expand_dims(generated_images[idx,:,:,0],0)
            cond_spec = tf.expand_dims(cond_images[idx,:,:,0],0)
            gt_spec = tf.expand_dims(des_images[idx,:,:,0],0)
            comp_SPEC_FIG = plt.figure()
            plt.subplot(311),
            plt.title('Cond ')
            plt.imshow(tf.transpose(cond_spec[0]), aspect='auto'),plt.colorbar()
            plt.gca().invert_yaxis()
            plt.subplot(312),
            plt.title('Est ')
            plt.imshow(tf.transpose(est_spec[0]), aspect='auto'),plt.colorbar()
            plt.gca().invert_yaxis()
            plt.subplot(313),
            plt.title('Gt ')
            plt.imshow(tf.transpose(gt_spec[0]), aspect='auto'), plt.colorbar()
            plt.gca().invert_yaxis()
            plt.tight_layout()


            if do_norm_specs:
                tgt_spec = denorm_tensor(est_spec) # target timbre spectrogram
                cond_spec = denorm_tensor(cond_spec) # conditioning timbre spectrogram
                gt_spec = denorm_tensor(gt_spec) # ground truth timbre spectrogram

            # audio
            # audio_est = tf.cast(tf.expand_dims(module(est_spec),axis=-1),dtype=tf.float32)
            # cond_audio = tf.cast(tf.expand_dims(module(cond_spec),axis=-1),dtype=tf.float32)
            # gt_audio = tf.cast(tf.expand_dims(module(gt_spec),axis=-1),dtype=tf.float32)
            tgt_audio = tf.cast(tf.expand_dims(calculate_spectrogram(tgt_spec),axis=-1),dtype=tf.float32)
            cond_audio = tf.cast(tf.expand_dims(calculate_spectrogram(cond_spec),axis=-1),dtype=tf.float32)
            gt_audio = tf.cast(tf.expand_dims(calculate_spectrogram(gt_spec),axis=-1),dtype=tf.float32)

            with self.summary_writer.as_default():
                # Write images to tensorboard
                tf.summary.image("Examples", plot_to_image(spec_figures), step=epoch)
                tf.summary.image("TT", plot_to_image(comp_SPEC_FIG), step=epoch)
                tf.summary.audio('Estimated Audio ', audio_est, sample_rate=params.SAMPLE_RATE, step=epoch)
                tf.summary.audio('Conditioning Audio ', cond_audio, sample_rate=params.SAMPLE_RATE, step=epoch)
                tf.summary.audio('GT audio violin', gt_audio, sample_rate=params.SAMPLE_RATE, step=epoch)

