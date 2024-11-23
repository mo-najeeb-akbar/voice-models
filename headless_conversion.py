from infer.modules.vc.modules import VC
from configs.config import Config
from scipy.io import wavfile

if __name__ == "__main__":
    config = Config()
    vc = VC(config)
    sid = '/code/assets/weights/pac.pth'
    vc.get_vc(sid, .33, 0)

    input_audio0 = '/code/assets/test.wav'
    vc_transform0 = 0
    f0_file = None
    f0method0 = ["pm", "harvest", "crepe", "rmvpe"]
    file_index1 = '/code/assets/indices/pac.index'

    index_rate1 = 0.75
    filter_radius0 = 3
    resample_sr0 = 0
    rms_mix_rate0 = 0.25
    protect0 = 0.33

    res = vc.vc_single(
        0, input_audio0, 0, None, "pm",
        file_index1, file_index1, index_rate1 ,filter_radius0, resample_sr0, rms_mix_rate0, protect0
    )

    # Save as WAV file
    wavfile.write('test_output.wav', res[1][0], res[1][1])
