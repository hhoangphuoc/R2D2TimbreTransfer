from tqdm import tqdm
import argparse
import shutil
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--datasource", type=str, default='../../data/urmp/', help="data source of URMP dataset")
parser.add_argument("--outdir", type=str, default='../T2R2D2/data_specs/', help="output directory storing the processed data")
args = parser.parse_args()
print(args)

# Gets the audio seperated wavs of a specific instrument from the URMP dataset
def get_audiosep_ins(ins):
    return glob.glob(args.datasource + '**/AuSep*'+ins+'*.wav', recursive = True)

# Saves wavs belonging to speaker from list of speaker files
def urmp_prep_wavs(outdir, instrument_files, instrument):
    os.makedirs(outdir, exist_ok=True)
    for f in tqdm(instrument_files, desc="extracting audio for instrument %s"%instrument):
        shutil.copy(f, outdir)
        
# trumpet_files = get_audiosep_ins('tpt')
violin_files = get_audiosep_ins('vn')

# Data preparation
urmp_prep_wavs(args.outdir+'vn', violin_files, 'vn')