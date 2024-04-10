import os
import torch
import torchaudio
import numpy as np
import random

from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
from models import voicecraft
from inference_tts_scale import inference_one_sample

MFA_MODEL_ACOUTIC_DICT = {
    "english": "english_us_arpa",
    "mandarin": "mandarin_mfa",
}

MFA_MODEL_DICTIONARY_DICT = {
    "english": "english_us_arpa",
    "mandarin": "mandarin_taiwan_mfa",
}


class VoiceTask:
    def __init__(self, voice_name: str = "", voice_file: str = "",
                 voice_language_dictionary: str = "", voice_language_acoustic: str = "",
                 transcript: str = "", temp_folder_name: str = "", cut_off_sec: float = 0.0,
                 target_transcript_prefix: str = "", voice_source_url: str = "", target_transcript: str = "",
                 model_name: str = "", model_path: str = "",
                 encodec_fn: str = "", output_folder_path: str = ""):
        # Voice role (Input)
        self.voice_name: str = voice_name
        self.voice_file: str = voice_file
        self.voice_language_dictionary: str = voice_language_dictionary
        self.voice_language_acoustic: str = voice_language_acoustic
        self.transcript: str = transcript
        self.temp_folder_name: str = temp_folder_name
        self.cut_off_sec: float = cut_off_sec
        self.target_transcript_prefix: str = target_transcript_prefix
        self.voice_source_url: str = voice_source_url
        # Target Transcript (Input)
        self.target_transcript: str = target_transcript
        # Model selection
        self.model_name: str = model_name
        self.model_path: str = model_path
        self.encodec_fn: str = encodec_fn
        # Output
        self.output_folder_path: str = output_folder_path


def main(voice_task: VoiceTask):
    # Set environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["USER"] = "WINSTON"  # Change this to your username

    voice_language_dictionary = voice_task.voice_language_dictionary
    voice_language_acoustic = voice_task.voice_language_acoustic

    # Install MFA models and dictionaries
    os.system("source ~/.zshrc")
    os.system("conda activate voicecraft")
    os.system(f"mfa model download dictionary {voice_language_dictionary}")
    os.system(f"mfa model download acoustic {voice_language_acoustic}")

    # Load model, tokenizer, and other necessary files
    device = "cuda" if torch.cuda.is_available() else "cpu"

    voicecraft_name = voice_task.model_path
    encodec_fn = voice_task.encodec_fn

    if not os.path.exists(voicecraft_name):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/{voicecraft_name}\?download\=true")
        os.system(f"mv {voicecraft_name}\?download\=true ./pretrained_models/{voicecraft_name}")

    if not os.path.exists(encodec_fn):
        os.system("wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
        os.system("mv encodec_4cb2048_giga.th ./pretrained_models/encodec_4cb2048_giga.th")

    ckpt = torch.load(voicecraft_name, map_location="cpu")
    model = voicecraft.VoiceCraft(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    phn2num = ckpt['phn2num']

    text_tokenizer = TextTokenizer(backend="espeak")
    audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=device)

    # Prepare your audio
    orig_audio = voice_task.voice_file
    orig_transcript = voice_task.transcript

    temp_folder = voice_task.temp_folder_name
    os.makedirs(temp_folder, exist_ok=True)
    os.system(f"cp {orig_audio} {temp_folder}")
    filename = os.path.splitext(orig_audio.split("/")[-1])[0]
    with open(f"{temp_folder}/{filename}.txt", "w") as f:
        f.write(orig_transcript)

    align_temp = f"{temp_folder}/mfa_alignments"
    os.system(f"conda activate voicecraft")
    os.system(
        f"mfa align -v --clean -j 1 --output_format csv {temp_folder} {voice_language_dictionary} {voice_language_acoustic} {align_temp}"
    )

    cut_off_sec = voice_task.cut_off_sec
    target_transcript = voice_task.target_transcript_prefix + voice_task.target_transcript

    audio_fn = f"{temp_folder}/{filename}.wav"
    info = torchaudio.info(audio_fn)
    audio_dur = info.num_frames / info.sample_rate

    assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
    prompt_end_frame = int(cut_off_sec * info.sample_rate)
    # prompt_end_frame = int(audio_dur)

    codec_audio_sr = 16000
    codec_sr = 50
    top_k = 0
    top_p = 0.8
    temperature = 1.0
    silence_tokens = [1388, 1898, 131]
    kvcache = 1

    stop_repetition = 3
    sample_batch_size = 4
    seed = 1

    def seed_everything(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_everything(seed)

    decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition,
                     'kvcache': kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr,
                     "silence_tokens": silence_tokens, "sample_batch_size": sample_batch_size}

    concated_audio, gen_audio = inference_one_sample(model, ckpt["config"], phn2num, text_tokenizer, audio_tokenizer,
                                                     audio_fn, target_transcript, device, decode_config,
                                                     prompt_end_frame)

    # save segments for comparison
    concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()

    output_dir = voice_task.output_folder_path
    os.makedirs(output_dir, exist_ok=True)
    seg_save_fn_gen = f"{output_dir}/{os.path.basename(audio_fn)[:-4]}_gen_seed{seed}.wav"
    seg_save_fn_concat = f"{output_dir}/{os.path.basename(audio_fn)[:-4]}_concat_seed{seed}.wav"

    torchaudio.save(seg_save_fn_gen, gen_audio, codec_audio_sr)
    torchaudio.save(seg_save_fn_concat, concated_audio, codec_audio_sr)


if __name__ == "__main__":
    # Set voice task (Trump)
    trump_voice_role = VoiceTask(voice_name="Trump", voice_file="./demo/trump_voice.wav",
                                 voice_language_dictionary="english_us_arpa",
                                 voice_language_acoustic="english_us_arpa",
                                 transcript="As long as I am President of the United States, Iran will never be allowed to have a nuclear weapon. Good morning. I'm pleased to inform you",
                                 temp_folder_name="./demo/trump", cut_off_sec=3.89,
                                 target_transcript_prefix="As long as I am President of the United States.",
                                 voice_source_url="https://www.youtube.com/watch?v=LLTDTVEqOXo",
                                 target_transcript="Learning a new language can be challenging, but with practice, it becomes easier.",
                                 model_name="VoiceCraft",
                                 model_path="/Volumes/Seagate Bac/models/TTS/VoiceCraft/giga830M.pth",
                                 encodec_fn="/Volumes/Seagate Bac/models/TTS/VoiceCraft/encodec_4cb2048_giga.th",
                                 output_folder_path="/Volumes/Seagate Bac/AIGC/TTS/voicecraft")

    # Set voice task (Modi)
    modi_voice_role = VoiceTask(voice_name="Modi", voice_file="./demo/modi_voice.wav",
                                voice_language_dictionary="english_us_arpa",
                                voice_language_acoustic="english_us_arpa",
                                transcript="India's development, demography and demand provide a unique long term opportunity for Australia, and all in the familiar. I stand here as one of you",
                                temp_folder_name="./demo/modi", cut_off_sec=11.17,
                                target_transcript_prefix="India's development, demography and demand provide a unique long term opportunity for Australia, ",
                                voice_source_url="https://www.youtube.com/watch?v=i2oBEmmNdFY",
                                target_transcript="Learning a new language can be challenging, but with practice, it becomes easier.",
                                model_name="VoiceCraft",
                                model_path="/Volumes/Seagate Bac/models/TTS/VoiceCraft/giga830M.pth",
                                encodec_fn="/Volumes/Seagate Bac/models/TTS/VoiceCraft/encodec_4cb2048_giga.th",
                                output_folder_path="/Volumes/Seagate Bac/AIGC/TTS/voicecraft")
    # modi_voice_role.target_transcript = "Chipi chipi chapa chapa, Dubee dubee, daba daba, Magicomi dubee dubee, Boom! Boom! Boom! Boom! Boom!"

    # Set voice task (Elder)
    elder_voice_role = VoiceTask(voice_name="Elder", voice_file="./demo/elder_voice.wav",
                                 voice_language_dictionary="mandarin_taiwan_mfa",
                                 voice_language_acoustic="mandarin_mfa",
                                 transcript="我告诉你们我是身经百战了，见得多了，西方的哪一个国家我没去过？媒体他们，你们要知道，美国的华莱士，那比你们高到不知道哪里去了，诶，我给他谈笑风生。",
                                 temp_folder_name="./demo/elder", cut_off_sec=9.02,
                                 target_transcript_prefix="我告诉你们我是身经百战了，见得多了，西方的哪一个国家我没去过？媒体他们，你们要知道，",
                                 voice_source_url="https://www.youtube.com/watch?v=kMCSMrnVsek",
                                 target_transcript="当然，官方网站专栏常常绽放万丈光芒，让网站上满满旁观呆丸郎按赞逛逛！",
                                 # target_transcript="南港展覽館館長晚上趕場幫忙南山廣場場方把鋼彈跟單槓搬到南港展覽館辦場鋼彈觀光展看鋼彈吊單槓往上。當然，官方網站專欄常常綻放萬丈光芒，讓網站上滿滿旁觀呆丸郎按讚逛逛，難波萬，棒！",
                                 model_name="VoiceCraft",
                                 model_path="/Volumes/Seagate Bac/models/TTS/VoiceCraft/giga830M.pth",
                                 encodec_fn="/Volumes/Seagate Bac/models/TTS/VoiceCraft/encodec_4cb2048_giga.th",
                                 output_folder_path="/Volumes/Seagate Bac/AIGC/TTS/voicecraft")

    # main(trump_voice_role)
    # main(modi_voice_role)
    main(elder_voice_role)

# TODO
# [X] 1. demo內資料夾自動建立，包含temp_folder_name及內部的mfa_alignments/
# [ ] 2. 透過mfa進行alignment後，自動從csv內找截斷點 (cut_off_sec)
# [X] 3. 欲生成文字應以變數傳入 (Learning a new language can be challenging...)
# [ ] 4. 生成的音檔應以原始檔案加上流水存放於該聲線資料夾內
# [X] 5. 自建repo存放 (已 fork)
