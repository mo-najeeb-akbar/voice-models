import os
import sys
sys.path.append('/code/')
import json
import argparse
import glob
from tqdm import tqdm
from infer.modules.vc.modules import VC
from configs.config import Config
from scipy.io import wavfile


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run batched inference on all audio files given some voice model.')
    parser.add_argument('--options',
                        type=str,
                        default='',
                        required=True,
                        help='Path to your options JSON file.')

    return parser.parse_args()


def load_options(config_path):
    data = None
    with open(config_path, 'r') as file:
        data = json.load(file)
    return data


def make_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f'Created new directory: {output_path}')
    else:
        print('Directory already exists -- not creating.')


if __name__ == "__main__":

    # TODO: make this less shitty
    options = load_options('/code/headless/config.json')

    # Data Options
    data_options = options['data']
    input_data_path = data_options['input_data_path']
    output_data_path = data_options['output_data_path']
    make_dir(output_data_path)

    # Model and Inference Options
    inference_options = options['inference']
    model_path = inference_options['model_path']
    index_path = inference_options['index_path']
    f0_method = inference_options['f0_method']  # any of ["pm", "harvest", "crepe", "rmvpe"]
    f0_up_key = inference_options['f0_up_key']

    # TODO: determine the importance of these
    index_rate = inference_options['index_rate']
    filter_radius = inference_options['filter_radius']
    resample_sr = inference_options['resample_sr']
    rms_mix_rate = inference_options['rms_mix_rate']
    protect = inference_options['protect']

    # Setup Inference Object -- TODO: confusing, remove this wrapper
    config = Config()
    vc = VC(config)
    vc.get_vc(model_path, protect, 0)

    # Run inference in a loop for every audio file
    audio_files = glob.glob(os.path.join(input_data_path, '*.mp3'))
    print(f'Beginning processing ...')
    print(f'Model: {model_path} -- will be applied to {len(audio_files)} files.')
    for audio_file in tqdm(audio_files):

        res = vc.vc_single(
            0,
            audio_file,
            f0_up_key,
            None,
            f0_method,
            index_path,
            index_path,
            index_rate,
            filter_radius,  # smooths out pitch contour
            resample_sr,  # how many times to send the audio file through the model -- sound more like model voice
            rms_mix_rate,  # 0 - close to volume of input audio -- 1 consistent volume
            protect  # preserve consonant sounds
        )
        filename = os.path.basename(audio_file)
        filename_without_wav = filename.removesuffix('.wav')
        out_name = os.path.join(output_data_path, filename_without_wav + '_processed.wav')
        wavfile.write(out_name, res[1][0], res[1][1])
        print(f'Finished processing ... {filename_without_wav}.')
