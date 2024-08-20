import tensorflow as tf

TGT_IMG_CHANNELS = 1 #3 Numbers of channels in the target timbre image
COND_IMG_CHANNELS=2 # Number of channels in the condition timbre image

EMBEDDINGS_DIMS = 32
EMBEDDINGS_MAX_FREQ = 1000.0

# U-NET Model Architecture --------------------
# with the Mel-spectrogram size of (128, 512)

CONV_KERNEL_SIZE = 3

BLOCK_DEPTH = 4 # number of up/downsampling blocks
WIDTHS = [64, 128, 256, 512] # number of filters in up/downsampling blocks
HAS_ATTENTION = [False, False, True, True] # specified which block in U-NET added with the attention layer

# BATCH_SIZE = 16 #starting with 16, can be increased to 32 or 64 depending on the GPU memory
BATCH_SIZE = 64 #starting with 16, can be increased to 32 or 64 depending on the GPU memory

DURATION_SAMPLE = 40960 #*2 if 256


# DIFFUSION Config --------------------
# KID = Kernel Inception Distance, see related section
KID_IMG_SIZE = 75
KID_DIFFUSION_STEPS = 10
PLOT_DIFFUSSION_STEPS = 20
# sampling
MIN_SIGNAL_RATE = 0.02
MAX_SIGNAL_RATE = 0.95
# --------------------------------------

# Training Config --------------------

DATASET_REPETITIONS = 5
# NUM_EPOCHS = 5000 # train for at least 50 epochs for good results
NUM_EPOCHS = 500
NUM_EPOCHS_PER_CHECKPOINT = 100

EMA = 0.999
LEARNING_RATE =  2e-5
WEIGHT_DECAY = 1e-4