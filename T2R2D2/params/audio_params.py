# PARAMETERS RELATED TO THE AUDIO PROCESSING 
# AND FOR THE CONVERSION FROM AUDIO TO MEL_SPECTROGRAM

SAMPLE_RATE = 16000
N_FFT = 1024 # (can be adjusted based on frequency resolution requirements)
HOP_LENGTH = 320
WIN_LENGTH = 640 # (20 ms)
N_MEL_CHANNELS = 128
MEL_FMIN = 0.0
MEL_FMAX = int(SAMPLE_RATE // 2)
CLIP_VALUE_MIN = 1e-5
CLIP_VALUE_MAX = 1e8

#------------------------------------------------

# Dataset Parameters
DATASET_PATH = "./data_specs/"

# Mel-Spectrogram Parameters

MEL_SPEC_HEIGHT = 128 #spec height
MEL_SPEC_WIDTH = 128 #spec width 
# MEL_SPEC_WIDTH = 512 #TODO - try width = 512


MEL_SPEC_TARGET_SHAPE = (MEL_SPEC_HEIGHT, MEL_SPEC_WIDTH, 1) # size of the original Mel-Spectrogram
MEL_SPEC_NORM_SIZE = (128, 128) #size of Mel-Spectrogram that will be used as input to the model

DURATION_SAMPLE = 40960 #*2 if 256
DURATION_TRACK = 480000