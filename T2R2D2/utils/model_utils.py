######################
# MODEL HELPER FUNCTION   #
######################

# This file contains functions that help keep track of the diffusion process, and find the diffusion model's output over steps

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image
# def denorm_tensor(audio, max_val=6,min_val=-12):
#     max_val = 6
#     min_val = -12
#     return (audio * (max_val-min_val))+min_val

def norm_tensor(audio):
    min_val = tf.math.reduce_min(audio)
    audio = audio - min_val
    max_val = tf.math.reduce_max(audio)
    audio_norm = ((audio/max_val)*2)-1
    return audio_norm, max_val, min_val

def denorm_tensor(audio,max_val,min_val):
    return (((audio +1)/2)*max_val + min_val)

    # return (audio * (max_val-min_val))+min_val

def get_audio_track_diff(cond_track_path, diff_steps=20):
    cond_track = read_audio(cond_track_path)
    # Remove audio Channel and add Batch Dimension
    cond_track = tf.expand_dims(cond_track[:,0],axis=0)
    do_norm = True
    if do_norm:
        cond_track,_,_ = norm_tensor(cond_track)

    # Compute mel spectrogram
    cond_track_spec = calculate_spectrogram(cond_track)

    # Frames given as input to diff model
    N_frames_diff = 128

    # Number of full frames contained in the considered track
    N_frames_full = cond_track_spec.shape[1]//N_frames_diff
    N_frames_gt = cond_track_spec.shape[1]
    # Split the cond track in frame sizes suitable to diff model
    if N_frames_full*N_frames_diff < cond_track_spec.shape[1]:
        cond_track_input_diff = np.zeros((N_frames_full+1,N_frames_diff,params.N_MEL_CHANNELS),dtype=np.float32)
        for i in range(N_frames_full):
            cond_track_input_diff[i] = cond_track_spec[0,(i*N_frames_diff):(i*N_frames_diff)+N_frames_diff]
        N_remaining_frames = len(cond_track_spec[0, (i * N_frames_diff) + N_frames_diff:])
        cond_track_input_diff[i+1,:N_remaining_frames] = cond_track_spec[0, (i * N_frames_diff) + N_frames_diff:]
    else:
        cond_track_input_diff = np.zeros(N_frames_full,N_frames_diff,N_MEL_CHANNELS,1)
        for i in range(N_frames_full):
            cond_track_input_diff[i] = cond_track_spec[0,(i*N_frames_diff):(i*N_frames_diff)+N_frames_diff]


    # Now let's apply the diffusion model
    model = network_lib.DiffusionModel(image_size, widths, block_depth,val_data=None,batch_size=batch_size)
    model.load_weights(checkpoint_path)
    N = cond_track_input_diff.shape[0]

    est_spec = model.generate(
        cond_images=tf.expand_dims(cond_track_input_diff, axis=-1),
        num_images=N,
        diffusion_steps=diff_steps,
    )

    est_spec_shift = tf.expand_dims(tf.zeros_like(cond_track_input_diff), axis=-1).numpy()
    N_slices = est_spec_shift.shape[0]
    for i in range(N_slices - 1):
        # Curr + shifted slice
        cond_shift = tf.expand_dims(
            tf.expand_dims(tf.concat([cond_track_input_diff[i][64:], cond_track_input_diff[i + 1][:64]], axis=0),
                           axis=0), axis=-1)
        est_spec_shift[i] = model.generate(cond_images=cond_shift, num_images=1, diffusion_steps=diff_steps)

    est_spec_smooth = est_spec.numpy()
    for i in range(est_spec.numpy().shape[0] - 1):
        est_spec_smooth[i, 96:] = est_spec_shift[i, 32:64]
        est_spec_smooth[i + 1, :32] = est_spec_shift[i, 64:96]

    est_spec = tf.reshape(est_spec_smooth,(N*128,128)).numpy()[:N_frames_gt]
    cond_track_input_diff = tf.reshape(cond_track_input_diff,(N*128,128)).numpy()[:N_frames_gt]

    est_audio = module(tf.expand_dims(est_spec,axis=0)).numpy()
    cond_audio =  module(tf.expand_dims(cond_track_input_diff,axis=0)).numpy()

    return est_audio, cond_audio


def get_audio_track_diff_norm(cond_track_path, checkpoint_path, model, diff_steps=20):
    cond_track = read_audio(cond_track_path)
    # Remove audio Channel and add Batch Dimension
    cond_track = tf.expand_dims(cond_track[:, 0], axis=0)
    do_norm = True
    if do_norm:
        cond_track, _, _ = norm_tensor(cond_track)

    # Compute mel spectrogram
    cond_track_spec = calculate_spectrogram(cond_track)

    # Frames given as input to diff model
    N_frames_diff = 128

    # Number of full frames contained in the considered track
    N_frames_full = cond_track_spec.shape[1] // N_frames_diff
    N_frames_gt = cond_track_spec.shape[1]
    # Split the cond track in frame sizes suitable to diff model
    if N_frames_full * N_frames_diff < cond_track_spec.shape[1]:
        cond_track_input_diff = np.zeros((N_frames_full + 1, N_frames_diff, params.N_MEL_CHANNELS), dtype=np.float32)
        for i in range(N_frames_full):
            cond_track_input_diff[i] = cond_track_spec[0, (i * N_frames_diff):(i * N_frames_diff) + N_frames_diff]
        N_remaining_frames = len(cond_track_spec[0, (i * N_frames_diff) + N_frames_diff:])
        cond_track_input_diff[i + 1, :N_remaining_frames] = cond_track_spec[0, (i * N_frames_diff) + N_frames_diff:]
    else:
        cond_track_input_diff = np.zeros(N_frames_full, N_frames_diff, params.N_MEL_CHANNELS, 1)
        for i in range(N_frames_full):
            cond_track_input_diff[i] = cond_track_spec[0, (i * N_frames_diff):(i * N_frames_diff) + N_frames_diff]

    # Now let's apply the diffusion model
    # model = network_lib.DiffusionModel(image_size, widths, block_depth,val_data=None,batch_size=batch_size)
    model.load_weights(checkpoint_path)
    N = cond_track_input_diff.shape[0]

    # Compute norm of cond ttrack
    cond_track_spec_reshaped_norm = np.zeros_like(cond_track_input_diff)
    max_val, min_val = np.zeros(N), np.zeros(N)
    for i in range(cond_track_input_diff.shape[0]):
        cond_track_spec_reshaped_norm[i], max_val[i], min_val[i] = norm_tensor(cond_track_input_diff[i])
        # print(str(max_val[i]) + ' ' + str(min_val[i]))

    cond_track_spec_reshaped_norm, max_val, min_val = norm_tensor(cond_track_input_diff)

    est_spec_norm = model.generate_fixed_noise(
        cond_images=tf.expand_dims(cond_track_spec_reshaped_norm, axis=-1),
        num_images=N,
        diffusion_steps=diff_steps,
    )
    # est_spec =  denorm_tensor(est_spec_norm,np.mean(max_val),np.mean(min_val))
    est_spec = denorm_tensor(est_spec_norm, max_val, min_val)

    est_spec = est_spec.numpy()

    est_spec = tf.reshape(est_spec, (N * 128, 128)).numpy()[:N_frames_gt]
    cond_track_input_diff = tf.reshape(cond_track_input_diff, (N * 128, 128)).numpy()[:N_frames_gt]

    est_audio = module(tf.expand_dims(est_spec, axis=0)).numpy()
    cond_audio = module(tf.expand_dims(cond_track_input_diff, axis=0)).numpy()

    return est_audio, cond_audio, cond_track_input_diff, est_spec
