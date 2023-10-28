#%cd SoniTranslate
import numpy as np
import gradio as gr
import whisperx
from whisperx.utils import LANGUAGES as LANG_TRANSCRIPT
from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH as DAMT, DEFAULT_ALIGN_MODELS_HF as DAMHF
from IPython.utils import capture
import torch
from gtts import gTTS
import librosa
import edge_tts
import asyncio
import gc
from pydub import AudioSegment
from tqdm import tqdm
from deep_translator import GoogleTranslator
import os
from soni_translate.audio_segments import create_translated_audio
from soni_translate.text_to_speech import make_voice_gradio
from soni_translate.translate_segments import translate_text
import time
import shutil
from urllib.parse import unquote
import zipfile
import rarfile
import logging
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)


title = "<center><strong><font size='7'>📽️ SoniTranslate 🈷️</font></strong></center>"

news = """ ## 📖 Новости
        🔥 2023/07/26: Новый пользовательский интерфейс и добавление параметров смешивания.

        🔥 2023/07/27: Фикс большого количества багов.

        🔥 2023/08/01: Добавление разных опций.

        🔥 2023/08/02: Добавление новых языков. 🌐

        🔥 2023/08/03: Изменены параметры по умолчанию и добавлен просмотр каталога загрузок..
        """

description = """
### 🎥 **Легко переводите видео с помощью SoniTranslate!** 📽️

Загрузите видео или дайте ссылку на видео. 📽️ **Получите обновленный блокнот из официального репозитория: [SoniTranslate](https://github.com/R3gm/SoniTranslate)!**

См. вкладку с надписью `Помощь` для получения инструкций о том, как его использовать. Начнем развлекаться с переводом видео! 🚀🎉
"""



tutorial = """

# 🔰 **Инструкции по использованию:**

1. 📤 **Загрузить видео** на первой вкладке или 🌐 **используйте ссылку на видео** на второй вкладке.

2. 🌍 Выберите язык, на котором вы хотите **перевести видео**.

3. 🗣️ Укажите **номер спикера** в видео и **назначьте каждому голос для преобразования текста в речь** подходит для языка перевода.

4. 🚀 Нажми на кнопку'**Translate**' и жди результаты.


# 🎤 Как использовать RVC и RVC2 модели 🎶

Цель состоит в том, чтобы применить RVC (Retrieval-based Voice Conversion) к созданному TTS (Text-to-Speech) 🎙️

1. Во вкладке `Кастомная модель RVC`, скачайте нужные вам модели. 📥 Вы можете использовать ссылки с Hugging Face и Google Drive в таких форматах, как zip, pth или index. Также можно скачать полные space репозитории HF, но этот вариант не очень стабилен 😕

2. Теперь перейдите к `Заменить войс: TTS в RVC` и проверьте что она `включена` ✅ После этого вы можете выбрать модели, которые хотите применить к каждому TTS. 👩‍🦰👨‍🦱👩‍🦳👨‍🦲

3. Настройте метод F0, который будет применяться ко всем RVC. 🎛️

4. Нажмите `ПРИМЕНИТЬ КОНФИГУРАЦИЮ` чтобы применить внесенные вами изменения 🔄

5. Вернитесь на вкладку перевода видео и нажмите 'Translate' ▶️ Теперь перевод будет выполняться с использованием RVC. 🗣️

Совет: вы можете использовать «Тестирование RVC», чтобы поэкспериментировать и найти лучшие TTS или конфигурации для применения к RVC. 🧪🔍

"""



# проверка GPU
if torch.cuda.is_available():
    device = "cuda"
    list_compute_type = ['float16', 'float32']
    compute_type_default = 'float16'
    whisper_model_default = 'large-v2'
else:
    device = "cpu"
    list_compute_type = ['float32']
    compute_type_default = 'float32'
    whisper_model_default = 'medium'
print('Работаю на: ', device)

list_tts = ['af-ZA-AdriNeural-Female', 'af-ZA-WillemNeural-Male', 'am-ET-AmehaNeural-Male', 'am-ET-MekdesNeural-Female', 'ar-AE-FatimaNeural-Female', 'ar-AE-HamdanNeural-Male', 'ar-BH-AliNeural-Male', 'ar-BH-LailaNeural-Female', 'ar-DZ-AminaNeural-Female', 'ar-DZ-IsmaelNeural-Male', 'ar-EG-SalmaNeural-Female', 'ar-EG-ShakirNeural-Male', 'ar-IQ-BasselNeural-Male', 'ar-IQ-RanaNeural-Female', 'ar-JO-SanaNeural-Female', 'ar-JO-TaimNeural-Male', 'ar-KW-FahedNeural-Male', 'ar-KW-NouraNeural-Female', 'ar-LB-LaylaNeural-Female', 'ar-LB-RamiNeural-Male', 'ar-LY-ImanNeural-Female', 'ar-LY-OmarNeural-Male', 'ar-MA-JamalNeural-Male', 'ar-MA-MounaNeural-Female', 'ar-OM-AbdullahNeural-Male', 'ar-OM-AyshaNeural-Female', 'ar-QA-AmalNeural-Female', 'ar-QA-MoazNeural-Male', 'ar-SA-HamedNeural-Male', 'ar-SA-ZariyahNeural-Female', 'ar-SY-AmanyNeural-Female', 'ar-SY-LaithNeural-Male', 'ar-TN-HediNeural-Male', 'ar-TN-ReemNeural-Female', 'ar-YE-MaryamNeural-Female', 'ar-YE-SalehNeural-Male', 'az-AZ-BabekNeural-Male', 'az-AZ-BanuNeural-Female', 'bg-BG-BorislavNeural-Male', 'bg-BG-KalinaNeural-Female', 'bn-BD-NabanitaNeural-Female', 'bn-BD-PradeepNeural-Male', 'bn-IN-BashkarNeural-Male', 'bn-IN-TanishaaNeural-Female', 'bs-BA-GoranNeural-Male', 'bs-BA-VesnaNeural-Female', 'ca-ES-EnricNeural-Male', 'ca-ES-JoanaNeural-Female', 'cs-CZ-AntoninNeural-Male', 'cs-CZ-VlastaNeural-Female', 'cy-GB-AledNeural-Male', 'cy-GB-NiaNeural-Female', 'da-DK-ChristelNeural-Female', 'da-DK-JeppeNeural-Male', 'de-AT-IngridNeural-Female', 'de-AT-JonasNeural-Male', 'de-CH-JanNeural-Male', 'de-CH-LeniNeural-Female', 'de-DE-AmalaNeural-Female', 'de-DE-ConradNeural-Male', 'de-DE-KatjaNeural-Female', 'de-DE-KillianNeural-Male', 'el-GR-AthinaNeural-Female', 'el-GR-NestorasNeural-Male', 'en-AU-NatashaNeural-Female', 'en-AU-WilliamNeural-Male', 'en-CA-ClaraNeural-Female', 'en-CA-LiamNeural-Male', 'en-GB-LibbyNeural-Female', 'en-GB-MaisieNeural-Female', 'en-GB-RyanNeural-Male', 'en-GB-SoniaNeural-Female', 'en-GB-ThomasNeural-Male', 'en-HK-SamNeural-Male', 'en-HK-YanNeural-Female', 'en-IE-ConnorNeural-Male', 'en-IE-EmilyNeural-Female', 'en-IN-NeerjaExpressiveNeural-Female', 'en-IN-NeerjaNeural-Female', 'en-IN-PrabhatNeural-Male', 'en-KE-AsiliaNeural-Female', 'en-KE-ChilembaNeural-Male', 'en-NG-AbeoNeural-Male', 'en-NG-EzinneNeural-Female', 'en-NZ-MitchellNeural-Male', 'en-NZ-MollyNeural-Female', 'en-PH-JamesNeural-Male', 'en-PH-RosaNeural-Female', 'en-SG-LunaNeural-Female', 'en-SG-WayneNeural-Male', 'en-TZ-ElimuNeural-Male', 'en-TZ-ImaniNeural-Female', 'en-US-AnaNeural-Female', 'en-US-AriaNeural-Female', 'en-US-ChristopherNeural-Male', 'en-US-EricNeural-Male', 'en-US-GuyNeural-Male', 'en-US-JennyNeural-Female', 'en-US-MichelleNeural-Female', 'en-US-RogerNeural-Male', 'en-US-SteffanNeural-Male', 'en-ZA-LeahNeural-Female', 'en-ZA-LukeNeural-Male', 'es-AR-ElenaNeural-Female', 'es-AR-TomasNeural-Male', 'es-BO-MarceloNeural-Male', 'es-BO-SofiaNeural-Female', 'es-CL-CatalinaNeural-Female', 'es-CL-LorenzoNeural-Male', 'es-CO-GonzaloNeural-Male', 'es-CO-SalomeNeural-Female', 'es-CR-JuanNeural-Male', 'es-CR-MariaNeural-Female', 'es-CU-BelkysNeural-Female', 'es-CU-ManuelNeural-Male', 'es-DO-EmilioNeural-Male', 'es-DO-RamonaNeural-Female', 'es-EC-AndreaNeural-Female', 'es-EC-LuisNeural-Male', 'es-ES-AlvaroNeural-Male', 'es-ES-ElviraNeural-Female', 'es-GQ-JavierNeural-Male', 'es-GQ-TeresaNeural-Female', 'es-GT-AndresNeural-Male', 'es-GT-MartaNeural-Female', 'es-HN-CarlosNeural-Male', 'es-HN-KarlaNeural-Female', 'es-MX-DaliaNeural-Female', 'es-MX-JorgeNeural-Male', 'es-NI-FedericoNeural-Male', 'es-NI-YolandaNeural-Female', 'es-PA-MargaritaNeural-Female', 'es-PA-RobertoNeural-Male', 'es-PE-AlexNeural-Male', 'es-PE-CamilaNeural-Female', 'es-PR-KarinaNeural-Female', 'es-PR-VictorNeural-Male', 'es-PY-MarioNeural-Male', 'es-PY-TaniaNeural-Female', 'es-SV-LorenaNeural-Female', 'es-SV-RodrigoNeural-Male', 'es-US-AlonsoNeural-Male', 'es-US-PalomaNeural-Female', 'es-UY-MateoNeural-Male', 'es-UY-ValentinaNeural-Female', 'es-VE-PaolaNeural-Female', 'es-VE-SebastianNeural-Male', 'et-EE-AnuNeural-Female', 'et-EE-KertNeural-Male', 'fa-IR-DilaraNeural-Female', 'fa-IR-FaridNeural-Male', 'fi-FI-HarriNeural-Male', 'fi-FI-NooraNeural-Female', 'fil-PH-AngeloNeural-Male', 'fil-PH-BlessicaNeural-Female', 'fr-BE-CharlineNeural-Female', 'fr-BE-GerardNeural-Male', 'fr-CA-AntoineNeural-Male', 'fr-CA-JeanNeural-Male', 'fr-CA-SylvieNeural-Female', 'fr-CH-ArianeNeural-Female', 'fr-CH-FabriceNeural-Male', 'fr-FR-DeniseNeural-Female', 'fr-FR-EloiseNeural-Female', 'fr-FR-HenriNeural-Male', 'ga-IE-ColmNeural-Male', 'ga-IE-OrlaNeural-Female', 'gl-ES-RoiNeural-Male', 'gl-ES-SabelaNeural-Female', 'gu-IN-DhwaniNeural-Female', 'gu-IN-NiranjanNeural-Male', 'he-IL-AvriNeural-Male', 'he-IL-HilaNeural-Female', 'hi-IN-MadhurNeural-Male', 'hi-IN-SwaraNeural-Female', 'hr-HR-GabrijelaNeural-Female', 'hr-HR-SreckoNeural-Male', 'hu-HU-NoemiNeural-Female', 'hu-HU-TamasNeural-Male', 'id-ID-ArdiNeural-Male', 'id-ID-GadisNeural-Female', 'is-IS-GudrunNeural-Female', 'is-IS-GunnarNeural-Male', 'it-IT-DiegoNeural-Male', 'it-IT-ElsaNeural-Female', 'it-IT-IsabellaNeural-Female', 'ja-JP-KeitaNeural-Male', 'ja-JP-NanamiNeural-Female', 'jv-ID-DimasNeural-Male', 'jv-ID-SitiNeural-Female', 'ka-GE-EkaNeural-Female', 'ka-GE-GiorgiNeural-Male', 'kk-KZ-AigulNeural-Female', 'kk-KZ-DauletNeural-Male', 'km-KH-PisethNeural-Male', 'km-KH-SreymomNeural-Female', 'kn-IN-GaganNeural-Male', 'kn-IN-SapnaNeural-Female', 'ko-KR-InJoonNeural-Male', 'ko-KR-SunHiNeural-Female', 'lo-LA-ChanthavongNeural-Male', 'lo-LA-KeomanyNeural-Female', 'lt-LT-LeonasNeural-Male', 'lt-LT-OnaNeural-Female', 'lv-LV-EveritaNeural-Female', 'lv-LV-NilsNeural-Male', 'mk-MK-AleksandarNeural-Male', 'mk-MK-MarijaNeural-Female', 'ml-IN-MidhunNeural-Male', 'ml-IN-SobhanaNeural-Female', 'mn-MN-BataaNeural-Male', 'mn-MN-YesuiNeural-Female', 'mr-IN-AarohiNeural-Female', 'mr-IN-ManoharNeural-Male', 'ms-MY-OsmanNeural-Male', 'ms-MY-YasminNeural-Female', 'mt-MT-GraceNeural-Female', 'mt-MT-JosephNeural-Male', 'my-MM-NilarNeural-Female', 'my-MM-ThihaNeural-Male', 'nb-NO-FinnNeural-Male', 'nb-NO-PernilleNeural-Female', 'ne-NP-HemkalaNeural-Female', 'ne-NP-SagarNeural-Male', 'nl-BE-ArnaudNeural-Male', 'nl-BE-DenaNeural-Female', 'nl-NL-ColetteNeural-Female', 'nl-NL-FennaNeural-Female', 'nl-NL-MaartenNeural-Male', 'pl-PL-MarekNeural-Male', 'pl-PL-ZofiaNeural-Female', 'ps-AF-GulNawazNeural-Male', 'ps-AF-LatifaNeural-Female', 'pt-BR-AntonioNeural-Male', 'pt-BR-FranciscaNeural-Female', 'pt-PT-DuarteNeural-Male', 'pt-PT-RaquelNeural-Female', 'ro-RO-AlinaNeural-Female', 'ro-RO-EmilNeural-Male', 'ru-RU-DmitryNeural-Male', 'ru-RU-SvetlanaNeural-Female', 'si-LK-SameeraNeural-Male', 'si-LK-ThiliniNeural-Female', 'sk-SK-LukasNeural-Male', 'sk-SK-ViktoriaNeural-Female', 'sl-SI-PetraNeural-Female', 'sl-SI-RokNeural-Male', 'so-SO-MuuseNeural-Male', 'so-SO-UbaxNeural-Female', 'sq-AL-AnilaNeural-Female', 'sq-AL-IlirNeural-Male', 'sr-RS-NicholasNeural-Male', 'sr-RS-SophieNeural-Female', 'su-ID-JajangNeural-Male', 'su-ID-TutiNeural-Female', 'sv-SE-MattiasNeural-Male', 'sv-SE-SofieNeural-Female', 'sw-KE-RafikiNeural-Male', 'sw-KE-ZuriNeural-Female', 'sw-TZ-DaudiNeural-Male', 'sw-TZ-RehemaNeural-Female', 'ta-IN-PallaviNeural-Female', 'ta-IN-ValluvarNeural-Male', 'ta-LK-KumarNeural-Male', 'ta-LK-SaranyaNeural-Female', 'ta-MY-KaniNeural-Female', 'ta-MY-SuryaNeural-Male', 'ta-SG-AnbuNeural-Male', 'ta-SG-VenbaNeural-Female', 'te-IN-MohanNeural-Male', 'te-IN-ShrutiNeural-Female', 'th-TH-NiwatNeural-Male', 'th-TH-PremwadeeNeural-Female', 'tr-TR-AhmetNeural-Male', 'tr-TR-EmelNeural-Female', 'uk-UA-OstapNeural-Male', 'uk-UA-PolinaNeural-Female', 'ur-IN-GulNeural-Female', 'ur-IN-SalmanNeural-Male', 'ur-PK-AsadNeural-Male', 'ur-PK-UzmaNeural-Female', 'uz-UZ-MadinaNeural-Female', 'uz-UZ-SardorNeural-Male', 'vi-VN-HoaiMyNeural-Female', 'vi-VN-NamMinhNeural-Male', 'zh-CN-XiaoxiaoNeural-Female', 'zh-CN-XiaoyiNeural-Female', 'zh-CN-YunjianNeural-Male', 'zh-CN-YunxiNeural-Male', 'zh-CN-YunxiaNeural-Male', 'zh-CN-YunyangNeural-Male', 'zh-CN-liaoning-XiaobeiNeural-Female', 'zh-CN-shaanxi-XiaoniNeural-Female']

### голоса
with capture.capture_output() as cap:
    os.system('mkdir downloads')
    os.system('mkdir logs')
    os.system('mkdir weights')
    os.system('mkdir downloads')
    del cap


def print_tree_directory(root_dir, indent=''):
    if not os.path.exists(root_dir):
        print(f"{indent}Нерабочая директория или файл: {root_dir}")
        return

    items = os.listdir(root_dir)

    for index, item in enumerate(sorted(items)):
        item_path = os.path.join(root_dir, item)
        is_last_item = index == len(items) - 1

        if os.path.isfile(item_path) and item_path.endswith('.zip'):
            with zipfile.ZipFile(item_path, 'r') as zip_file:
                print(f"{indent}{'└──' if is_last_item else '├──'} {item} (zip file)")
                zip_contents = zip_file.namelist()
                for zip_item in sorted(zip_contents):
                    print(f"{indent}{'    ' if is_last_item else '│   '}{zip_item}")
        else:
            print(f"{indent}{'└──' if is_last_item else '├──'} {item}")

            if os.path.isdir(item_path):
                new_indent = indent + ('    ' if is_last_item else '│   ')
                print_tree_directory(item_path, new_indent)


def upload_model_list():
    weight_root = "weights"
    models = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            models.append(name)

    index_root = "logs"
    index_paths = []
    for name in os.listdir(index_root):
        if name.endswith(".index"):
            index_paths.append("logs/"+name)

    print(models, index_paths)
    return models, index_paths

def manual_download(url, dst):
    token = os.getenv("YOUR_HF_TOKEN")
    user_header = f"\"Авторизация: с помощью {token}\""

    if 'drive.google' in url:
        print("Ссылка на диск")
        if 'folders' in url:
            print("folder")
            os.system(f'gdown --folder "{url}" -O {dst} --fuzzy -c')
        else:
            print("single")
            os.system(f'gdown "{url}" -O {dst} --fuzzy -c')
    elif 'huggingface' in url:
        print("HuggingFace link")
        if '/blob/' in url or '/resolve/' in url:
          if '/blob/' in url:
              url = url.replace('/blob/', '/resolve/')
          #parsed_link = '\n{}\n\tout={}'.format(url, unquote(url.split('/')[-1]))
          #os.system(f'echo -e "{parsed_link}" | aria2c --header={user_header} --console-log-level=error --summary-interval=10 -i- -j5 -x16 -s16 -k1M -c -d "{dst}"')
          os.system(f"wget -P {dst} {url}")
        else:
          os.system(f"git clone {url} {dst+'repo/'}")
    elif 'http' in url or 'magnet' in url:
        parsed_link = '"{}"'.format(url)
        os.system(f'aria2c --optimize-concurrent-downloads --console-log-level=error --summary-interval=10 -j5 -x16 -s16 -k1M -c -d {dst} -Z {parsed_link}')


def download_list(text_downloads):
    try:
      urls = [elem.strip() for elem in text_downloads.split(',')]
    except:
      return 'Ссылка нерабочая!'

    os.system('mkdir downloads')
    os.system('mkdir logs')
    os.system('mkdir weights')
    path_download = "downloads/"
    for url in urls:
      manual_download(url, path_download)
    
    # Tree
    print('####################################')
    print_tree_directory("downloads", indent='')
    print('####################################')

    # Place files
    select_zip_and_rar_files("downloads/")

    models, _ = upload_model_list()
    os.system("rm -rf downloads/repo")

    return f"Скачано = {models}"


def select_zip_and_rar_files(directory_path="downloads/"):
    #filter
    zip_files = []
    rar_files = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".zip"):
            zip_files.append(file_name)
        elif file_name.endswith(".rar"):
            rar_files.append(file_name)

    # extract
    for file_name in zip_files:
        file_path = os.path.join(directory_path, file_name)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(directory_path)

    for file_name in rar_files:
        file_path = os.path.join(directory_path, file_name)
        with rarfile.RarFile(file_path, 'r') as rar_ref:
            rar_ref.extractall(directory_path)

    # set in path
    def move_files_with_extension(src_dir, extension, destination_dir):
        for root, _, files in os.walk(src_dir):
            for file_name in files:
                if file_name.endswith(extension):
                    source_file = os.path.join(root, file_name)
                    destination = os.path.join(destination_dir, file_name)
                    shutil.move(source_file, destination)

    move_files_with_extension(directory_path, ".index", "logs/")
    move_files_with_extension(directory_path, ".pth", "weights/")

    return 'Загрузка завершена'

def custom_model_voice_enable(enable_custom_voice):
    if enable_custom_voice:
      os.environ["VOICES_MODELS"] = 'ENABLE'
    else:
      os.environ["VOICES_MODELS"] = 'DISABLE'


models, index_paths = upload_model_list()

f0_methods_voice = ["pm", "harvest", "crepe", "rmvpe"]


from voice_main import ClassVoices
voices = ClassVoices()

'''
def translate_from_video(video, WHISPER_MODEL_SIZE, batch_size, compute_type,
                         TRANSLATE_AUDIO_TO, min_speakers, max_speakers,
                         tts_voice00, tts_voice01,tts_voice02,tts_voice03,tts_voice04,tts_voice05):

    YOUR_HF_TOKEN = os.getenv("My_hf_token")

    create_translated_audio(result_diarize, audio_files, Output_name_file)

    os.system("rm audio_dub_stereo.wav")
    os.system("ffmpeg -i audio_dub_solo.wav -ac 1 audio_dub_stereo.wav")

    os.system(f"rm {mix_audio}")
    os.system(f'ffmpeg -y -i audio.wav -i audio_dub_stereo.wav -filter_complex "[0:0]volume=0.15[a];[1:0]volume=1.90[b];[a][b]amix=inputs=2:duration=longest" -c:a libmp3lame {mix_audio}')

    os.system(f"rm {video_output}")
    os.system(f"ffmpeg -i {OutputFile} -i {mix_audio} -c:v copy -c:a copy -map 0:v -map 1:a -shortest {video_output}")

    return video_output
'''

def translate_from_video(
    video,
    YOUR_HF_TOKEN,
    preview=False,
    WHISPER_MODEL_SIZE="large-v1",
    batch_size=16,
    compute_type="float16",
    SOURCE_LANGUAGE= "Automatic detection",
    TRANSLATE_AUDIO_TO="English (en)",
    min_speakers=1,
    max_speakers=2,
    tts_voice00="en-AU-WilliamNeural-Male",
    tts_voice01="en-CA-ClaraNeural-Female",
    tts_voice02="en-GB-ThomasNeural-Male",
    tts_voice03="en-GB-SoniaNeural-Female",
    tts_voice04="en-NZ-MitchellNeural-Male",
    tts_voice05="en-GB-MaisieNeural-Female",
    video_output="video_dub.mp4",
    AUDIO_MIX_METHOD='Adjusting volumes and mixing audio',
    progress=gr.Progress(),
    ):

    if YOUR_HF_TOKEN == "" or YOUR_HF_TOKEN == None:
      YOUR_HF_TOKEN = os.getenv("YOUR_HF_TOKEN")
      if YOUR_HF_TOKEN == None:
        print('Токен невалидный')
        return "Токен невалидный"
      else:
        os.environ["YOUR_HF_TOKEN"] = YOUR_HF_TOKEN

    video = video if isinstance(video, str) else video.name
    print(video)

    if "SET_LIMIT" == os.getenv("DEMO"):
      preview=True
      print("Демо-версия; поставь preview=True; Генерация **ограничена 10 секундами**, чтобы предотвратить ошибки ЦП. Если вы используете графический процессор, у вас не будет ни одного из этих ограничений.")
      AUDIO_MIX_METHOD='Adjusting volumes and mixing audio'
      print("Демо-версия; поставь Adjusting volumes и mixing audio")
      WHISPER_MODEL_SIZE="medium"
      print("Демо-версия; поставь whisper модель на medium")

    LANGUAGES = {
        'Automatic detection': 'Automatic detection',
        'Arabic (ar)': 'ar',
        'Chinese (zh)': 'zh',
        'Czech (cs)': 'cs',
        'Danish (da)': 'da',
        'Dutch (nl)': 'nl',
        'English (en)': 'en',
        'Finnish (fi)': 'fi',
        'French (fr)': 'fr',
        'German (de)': 'de',
        'Greek (el)': 'el',
        'Hebrew (he)': 'he',
        'Hungarian (hu)': 'hu',
        'Italian (it)': 'it',
        'Japanese (ja)': 'ja',
        'Korean (ko)': 'ko',
        'Persian (fa)': 'fa',
        'Polish (pl)': 'pl',
        'Portuguese (pt)': 'pt',
        'Russian (ru)': 'ru',
        'Spanish (es)': 'es',
        'Turkish (tr)': 'tr',
        'Ukrainian (uk)': 'uk',
        'Urdu (ur)': 'ur',
        'Vietnamese (vi)': 'vi',
        'Hindi (hi)': 'hi',
    }

    TRANSLATE_AUDIO_TO = LANGUAGES[TRANSLATE_AUDIO_TO]
    SOURCE_LANGUAGE = LANGUAGES[SOURCE_LANGUAGE]


    if not os.path.exists('audio'):
        os.makedirs('audio')

    if not os.path.exists('audio2/audio'):
        os.makedirs('audio2/audio')

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float32" if device == "cpu" else compute_type

    OutputFile = 'Video.mp4'
    audio_wav = "audio.wav"
    Output_name_file = "audio_dub_solo.ogg"
    mix_audio = "audio_mix.mp3"

    os.system("rm Video.mp4")
    os.system("rm audio.webm")
    os.system("rm audio.wav")

    progress(0.15, desc="Processing video...")
    if os.path.exists(video):
        if preview:
            print('Создание превью видео продолжительностью 10 секунд, чтобы отключить эту опцию, зайдите в дополнительные настройки и отключите предпросмотр.')
            os.system(f'ffmpeg -y -i "{video}" -ss 00:00:20 -t 00:00:10 -c:v libx264 -c:a aac -strict experimental Video.mp4')
        else:
            # Check if the file ends with ".mp4" extension
            if video.endswith(".mp4"):
                destination_path = os.path.join(os.getcwd(), "Video.mp4")
                shutil.copy(video, destination_path)
            else:
                print("Файл не имеет расширения «.mp4». Конвертирование видео.")
                os.system(f'ffmpeg -y -i "{video}" -c:v libx264 -c:a aac -strict experimental Video.mp4')

        for i in range (120):
            time.sleep(1)
            print('Процесс с видео...')
            if os.path.exists(OutputFile):
                time.sleep(1)
                os.system("ffmpeg -y -i Video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav")
                time.sleep(1)
                break
            if i == 119:
              print('Ошибка процесса видео')
              return

        for i in range (120):
            time.sleep(1)
            print('Процесс с аудио...')
            if os.path.exists(audio_wav):
                break
            if i == 119:
              print("Ошибка: невозможно создать аудио")
              return

    else:
        if preview:
            print('Создание превью по ссылке, 10 секунд, чтобы отключить эту опцию, зайдите в дополнительные настройки и отключите предпросмотр.')
            #https://github.com/yt-dlp/yt-dlp/issues/2220
            mp4_ = f'yt-dlp -f "mp4" --downloader ffmpeg --downloader-args "ffmpeg_i: -ss 00:00:20 -t 00:00:10" --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {video}'
            wav_ = "ffmpeg -y -i Video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav"
            os.system(mp4_)
            os.system(wav_)
        else:
            mp4_ = f'yt-dlp -f "mp4" --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {OutputFile} {video}'
            wav_ = f'python -m yt_dlp --output {audio_wav} --force-overwrites --max-downloads 1 --no-warnings --no-abort-on-error --ignore-no-formats-error --extract-audio --audio-format wav {video}'

            os.system(wav_)

            for i in range (120):
                time.sleep(1)
                print('Процесс с аудио...')
                if os.path.exists(audio_wav) and not os.path.exists('audio.webm'):
                    time.sleep(1)
                    os.system(mp4_)
                    break
                if i == 119:
                  print('Ошибка скачивания аудио')
                  return

    print("Установка файл завершена.")
    progress(0.30, desc="Транскрипция...")

    SOURCE_LANGUAGE = None if SOURCE_LANGUAGE == 'Automatic detection' else SOURCE_LANGUAGE

    # 1. Transcribe with original whisper (batched)
    with capture.capture_output() as cap:
      model = whisperx.load_model(
          WHISPER_MODEL_SIZE,
          device,
          compute_type=compute_type,
          language= SOURCE_LANGUAGE,
          )
      del cap
    audio = whisperx.load_audio(audio_wav)
    result = model.transcribe(audio, batch_size=batch_size)
    gc.collect(); torch.cuda.empty_cache(); del model
    print("Транскрипция завершена")

    

    # 2. Align whisper output
    progress(0.45, desc="Aligning...")
    DAMHF.update(DAMT) #lang align
    EXTRA_ALIGN = {
        "hi": "theainerd/Wav2Vec2-large-xlsr-hindi"
    } # add new align models here
    #print(result['language'], DAM.keys(), EXTRA_ALIGN.keys())
    if not result['language'] in DAMHF.keys() and not result['language'] in EXTRA_ALIGN.keys():
        audio = result = None
        print("Автоматическое определение: исходный язык несовместим.")
        print(f"Найденный язык {result['language']} несовместим, вы можете выбрать исходный язык, чтобы избежать этой ошибки.")
        return

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
        model_name = None if result["language"] in DAMHF.keys() else EXTRA_ALIGN[result["language"]]
        )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=True,
        )
    gc.collect(); torch.cuda.empty_cache(); del model_a
    print("Выравнивание завершено")

    if result['segments'] == []:
        print('В аудио не обнаружено активной речи')
        return

    # 3. Assign speaker labels
    progress(0.60, desc="Обработка...")
    with capture.capture_output() as cap:
      diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)
      del cap
    diarize_segments = diarize_model(
        audio_wav,
        min_speakers=min_speakers,
        max_speakers=max_speakers)
    result_diarize = whisperx.assign_word_speakers(diarize_segments, result)
    gc.collect(); torch.cuda.empty_cache(); del diarize_model
    print("Обработка завершена")

    progress(0.75, desc="Перевод...")
    if TRANSLATE_AUDIO_TO == "zh":
        TRANSLATE_AUDIO_TO = "zh-CN"
    if TRANSLATE_AUDIO_TO == "he":
        TRANSLATE_AUDIO_TO = "iw"
    result_diarize['segments'] = translate_text(result_diarize['segments'], TRANSLATE_AUDIO_TO)
    print("Перевод завершен")

    progress(0.85, desc="Text_to_speech...")
    audio_files = []
    speakers_list = []

    # Mapping speakers to voice variables
    speaker_to_voice = {
        'SPEAKER_00': tts_voice00,
        'SPEAKER_01': tts_voice01,
        'SPEAKER_02': tts_voice02,
        'SPEAKER_03': tts_voice03,
        'SPEAKER_04': tts_voice04,
        'SPEAKER_05': tts_voice05
    }

    for segment in tqdm(result_diarize['segments']):

        text = segment['text']
        start = segment['start']
        end = segment['end']

        try:
            speaker = segment['speaker']
        except KeyError:
            segment['speaker'] = "SPEAKER_99"
            speaker = segment['speaker']
            print(f"Спикер не найден на этом элементе: вспомогательный TTS будет использоваться во время сегмента {segment['start'], segment['text']}")

        # make the tts audio
        filename = f"audio/{start}.ogg"

        if speaker in speaker_to_voice and speaker_to_voice[speaker] != 'None':
            make_voice_gradio(text, speaker_to_voice[speaker], filename, TRANSLATE_AUDIO_TO)
        elif speaker == "SPEAKER_99":
            try:
                tts = gTTS(text, lang=TRANSLATE_AUDIO_TO)
                tts.save(filename)
                print('Использую GTTS')
            except:
                tts = gTTS('a', lang=TRANSLATE_AUDIO_TO)
                tts.save(filename)
                print('Ошибка: Аудио будет заменено.')

        # duration
        duration_true = end - start
        duration_tts = librosa.get_duration(filename=filename)

        # porcentaje
        porcentaje = duration_tts / duration_true

        if porcentaje > 2.1:
            porcentaje = 2.1
        elif porcentaje <= 1.2 and porcentaje >= 0.8:
            porcentaje = 1.0
        elif porcentaje <= 0.79:
            porcentaje = 0.8

        # Smoth and round
        porcentaje = round(porcentaje+0.0, 1)

        # apply aceleration or opposite to the audio file in audio2 folder
        os.system(f"ffmpeg -y -loglevel panic -i {filename} -filter:a atempo={porcentaje} audio2/{filename}")

        duration_create = librosa.get_duration(filename=f"audio2/{filename}")
        audio_files.append(filename)
        speakers_list.append(speaker)

    # custom voice
    if os.getenv('VOICES_MODELS') == 'ENABLE':
        progress(0.90, desc="Применение настроенных голосов...")
        voices(speakers_list, audio_files)

    # replace files with the accelerates
    os.system("mv -f audio2/audio/*.ogg audio/")

    os.system(f"rm {Output_name_file}")

    progress(0.95, desc="Создание окончательно переведенного видео...")

    create_translated_audio(result_diarize, audio_files, Output_name_file)

    os.system(f"rm {mix_audio}")

    # TYPE MIX AUDIO
    if AUDIO_MIX_METHOD == 'Adjusting volumes and mixing audio':
        # volume mix
        os.system(f'ffmpeg -y -i {audio_wav} -i {Output_name_file} -filter_complex "[0:0]volume=0.15[a];[1:0]volume=1.90[b];[a][b]amix=inputs=2:duration=longest" -c:a libmp3lame {mix_audio}')
    else:
        try:
            # background mix
            os.system(f'ffmpeg -i {audio_wav} -i {Output_name_file} -filter_complex "[1:a]asplit=2[sc][mix];[0:a][sc]sidechaincompress=threshold=0.003:ratio=20[bg]; [bg][mix]amerge[final]" -map [final] {mix_audio}')
        except:
            # volume mix except
            os.system(f'ffmpeg -y -i {audio_wav} -i {Output_name_file} -filter_complex "[0:0]volume=0.25[a];[1:0]volume=1.80[b];[a][b]amix=inputs=2:duration=longest" -c:a libmp3lame {mix_audio}')

    os.system(f"rm {video_output}")
    os.system(f"ffmpeg -i {OutputFile} -i {mix_audio} -c:v copy -c:a copy -map 0:v -map 1:a -shortest {video_output}")

    return video_output

import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

sys.stdout = Logger("output.log")

def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()

def submit_file_func(file):
    print(file.name)
    return file.name, file.name

# max tts
MAX_TTS = 6

theme='Taithrah/Minimal'

with gr.Blocks(theme=theme) as demo:
    gr.Markdown(title)
    gr.Markdown(description)

#### video
    with gr.Tab("Аудиоперевод для видео"):
        with gr.Row():
            with gr.Column():
                #video_input = gr.UploadButton("Click to Upload a video", file_types=["video"], file_count="single") #gr.Video() # height=300,width=300
                video_input = gr.File(label="VIDEO")
                #link = gr.HTML()
                #video_input.change(submit_file_func, video_input, [video_input, link], show_progress='full')

                SOURCE_LANGUAGE = gr.Dropdown(['Automatic detection', 'Arabic (ar)', 'Chinese (zh)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Russian (ru)', 'Spanish (es)', 'Turkish (tr)', 'Ukrainian (uk)', 'Urdu (ur)', 'Vietnamese (vi)'], value='Automatic detection',label = 'Source language', info="This is the original language of the video")
                TRANSLATE_AUDIO_TO = gr.Dropdown(['Arabic (ar)', 'Chinese (zh)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Russian (ru)', 'Spanish (es)', 'Turkish (tr)', 'Ukrainian (uk)', 'Urdu (ur)', 'Vietnamese (vi)'], value='English (en)',label = 'Translate audio to', info="Select the target language, and make sure to select the language corresponding to the speakers of the target language to avoid errors in the process.")

                line_ = gr.HTML("<hr></h2>")
                gr.Markdown("Выберите, сколько человек говорят в видео.")
                min_speakers = gr.Slider(1, MAX_TTS, default=1, label="Минимально спикеров", step=1, visible=False)
                max_speakers = gr.Slider(1, MAX_TTS, value=2, step=1, label="Максимально спикеров", interative=True)
                gr.Markdown("Выберите нужный голос для каждого спикера.")
                def submit(value):
                    visibility_dict = {
                        f'tts_voice{i:02d}': gr.update(visible=i < value) for i in range(6)
                    }
                    return [value for value in visibility_dict.values()]
                tts_voice00 = gr.Dropdown(list_tts, value='en-AU-WilliamNeural-Male', label = 'TTS Спикер 1', visible=True, interactive= True)
                tts_voice01 = gr.Dropdown(list_tts, value='en-CA-ClaraNeural-Female', label = 'TTS Спикер 2', visible=True, interactive= True)
                tts_voice02 = gr.Dropdown(list_tts, value='en-GB-ThomasNeural-Male', label = 'TTS Спикер 3', visible=False, interactive= True)
                tts_voice03 = gr.Dropdown(list_tts, value='en-GB-SoniaNeural-Female', label = 'TTS Спикер 4', visible=False, interactive= True)
                tts_voice04 = gr.Dropdown(list_tts, value='en-NZ-MitchellNeural-Male', label = 'TTS Спикер 5', visible=False, interactive= True)
                tts_voice05 = gr.Dropdown(list_tts, value='en-GB-MaisieNeural-Female', label = 'TTS Спикер 6', visible=False, interactive= True)
                max_speakers.change(submit, max_speakers, [tts_voice00, tts_voice01, tts_voice02, tts_voice03, tts_voice04, tts_voice05])

                with gr.Column():
                      with gr.Accordion("Расширенные настройки", open=False):

                          AUDIO_MIX = gr.Dropdown(['Микширование звука с помощью сжатия боковой цепи', 'Регулировка громкости и микширование звука».'], value='Adjusting volumes and mixing audio', label = 'Audio Mixing Method', info="Mix original and translated audio files to create a customized, balanced output with two available mixing modes.")

                          gr.HTML("<hr></h2>")
                          gr.Markdown("Конфигурация Whisper по умолчанию.")
                          WHISPER_MODEL_SIZE = gr.inputs.Dropdown(['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2'], default=whisper_model_default, label="Whisper model")
                          batch_size = gr.inputs.Slider(1, 32, default=16, label="Batch size", step=1)
                          compute_type = gr.inputs.Dropdown(list_compute_type, default=compute_type_default, label="Compute type")

                          gr.HTML("<hr></h2>")
                          VIDEO_OUTPUT_NAME = gr.Textbox(label="Переведенное имя файла" ,value="video_output.mp4", info="Имя выходного файла")
                          PREVIEW = gr.Checkbox(label="Превью", info="Предварительный просмотр сокращает видео до 10 секунд в целях тестирования. Пожалуйста, отключите его, чтобы получить полную продолжительность видео.")

            with gr.Column(variant='compact'):
                with gr.Row():
                    video_button = gr.Button("ПЕРЕВЕСТИ", )
                with gr.Row():
                    video_output = gr.outputs.File(label="СКАЧАТЬ ПЕРЕВЕДЕННОЕ ВИДЕО") #gr.Video()

                line_ = gr.HTML("<hr></h2>")
                if os.getenv("YOUR_HF_TOKEN") == None or os.getenv("YOUR_HF_TOKEN") == "":
                  HFKEY = gr.Textbox(visible= True, label="HF Token", info="Важным шагом является принятие лицензионного соглашения на использование Pyannote. Вам необходимо иметь учетную запись на Hugging Face и принять лицензию на использование моделей: https://huggingface.co/pyannote/speaker-diarization и https://huggingface.co/pyannote/segmentation. Получите свой ТОКЕН здесь: https://hf.co/settings/tokens", placeholder="Токен идет сюда...")
                else:
                  HFKEY = gr.Textbox(visible= False, label="HF Token", info="Важным шагом является принятие лицензионного соглашения на использование Pyannote. Вам необходимо иметь учетную запись на Hugging Face и принять лицензию на использование моделей: https://huggingface.co/pyannote/speaker-diarization и https://huggingface.co/pyannote/segmentation. Получите свой ТОКЕН здесь: https://hf.co/settings/tokens", placeholder="Токен идет сюда...")

                gr.Examples(
                    examples=[
                        [
                            "./assets/Video_main.mp4",
                            "",
                            False,
                            "large-v2",
                            16,
                            "float16",
                            "Spanish (es)",
                            "English (en)",
                            1,
                            2,
                            'en-AU-WilliamNeural-Male',
                            'en-CA-ClaraNeural-Female',
                            'en-GB-ThomasNeural-Male',
                            'en-GB-SoniaNeural-Female',
                            'en-NZ-MitchellNeural-Male',
                            'en-GB-MaisieNeural-Female',
                            "video_output.mp4",
                            'Adjusting volumes and mixing audio',
                        ],
                    ],
                    fn=translate_from_video,
                    inputs=[
                    video_input,
                    HFKEY,
                    PREVIEW,
                    WHISPER_MODEL_SIZE,
                    batch_size,
                    compute_type,
                    SOURCE_LANGUAGE,
                    TRANSLATE_AUDIO_TO,
                    min_speakers,
                    max_speakers,
                    tts_voice00,
                    tts_voice01,
                    tts_voice02,
                    tts_voice03,
                    tts_voice04,
                    tts_voice05,
                    VIDEO_OUTPUT_NAME,
                    AUDIO_MIX,
                    ],
                    outputs=[video_output],
                    cache_examples=False,
                )

### link

    with gr.Tab("Аудиоперевод через ссылку на видео"):
        with gr.Row():
            with gr.Column():

                blink_input = gr.Textbox(label="Media link.", info="Дефолтная: www.youtube.com/watch?v=dQw4w9WgXcQ", placeholder="Ссылка должна быть тут...")

                bSOURCE_LANGUAGE = gr.Dropdown(['Автоматическое нахождение', 'Arabic (ar)', 'Chinese (zh)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Russian (ru)', 'Spanish (es)', 'Turkish (tr)', 'Ukrainian (uk)', 'Urdu (ur)', 'Vietnamese (vi)'], value='Automatic detection',label = 'Source language', info="This is the original language of the video")
                bTRANSLATE_AUDIO_TO = gr.Dropdown(['Arabic (ar)', 'Chinese (zh)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Russian (ru)', 'Spanish (es)', 'Turkish (tr)', 'Ukrainian (uk)', 'Urdu (ur)', 'Vietnamese (vi)'], value='English (en)',label = 'Translate audio to', info="Select the target language, and make sure to select the language corresponding to the speakers of the target language to avoid errors in the process.")

                bline_ = gr.HTML("<hr></h2>")
                gr.Markdown("Выбери, сколько людей говорят в видео.")
                bmin_speakers = gr.Slider(1, MAX_TTS, default=1, label="min_speakers", step=1, visible=False)
                bmax_speakers = gr.Slider(1, MAX_TTS, value=2, step=1, label="Max speakers", interative=True)
                gr.Markdown("Выберите нужный голос для каждого спикера.")
                def bsubmit(value):
                    visibility_dict = {
                        f'btts_voice{i:02d}': gr.update(visible=i < value) for i in range(6)
                    }
                    return [value for value in visibility_dict.values()]
                btts_voice00 = gr.Dropdown(list_tts, value='en-AU-WilliamNeural-Male', label = 'TTS Speaker 1', visible=True, interactive= True)
                btts_voice01 = gr.Dropdown(list_tts, value='en-CA-ClaraNeural-Female', label = 'TTS Speaker 2', visible=True, interactive= True)
                btts_voice02 = gr.Dropdown(list_tts, value='en-GB-ThomasNeural-Male', label = 'TTS Speaker 3', visible=False, interactive= True)
                btts_voice03 = gr.Dropdown(list_tts, value='en-GB-SoniaNeural-Female', label = 'TTS Speaker 4', visible=False, interactive= True)
                btts_voice04 = gr.Dropdown(list_tts, value='en-NZ-MitchellNeural-Male', label = 'TTS Speaker 5', visible=False, interactive= True)
                btts_voice05 = gr.Dropdown(list_tts, value='en-GB-MaisieNeural-Female', label = 'TTS Speaker 6', visible=False, interactive= True)
                bmax_speakers.change(bsubmit, bmax_speakers, [btts_voice00, btts_voice01, btts_voice02, btts_voice03, btts_voice04, btts_voice05])


                with gr.Column():
                      with gr.Accordion("Advanced Settings", open=False):

                          bAUDIO_MIX = gr.Dropdown(['Mixing audio with sidechain compression', 'Adjusting volumes and mixing audio'], value='Adjusting volumes and mixing audio', label = 'Audio Mixing Method', info="Смешайте оригинальные и переведенные аудиофайлы для создания индивидуального сбалансированного результата с двумя доступными режимами микширования.")

                          gr.HTML("<hr></h2>")
                          gr.Markdown("Конфигурация Whisper по умолчанию.")
                          bWHISPER_MODEL_SIZE = gr.inputs.Dropdown(['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2'], default=whisper_model_default, label="Whisper model")
                          bbatch_size = gr.inputs.Slider(1, 32, default=16, label="Batch size", step=1)
                          bcompute_type = gr.inputs.Dropdown(list_compute_type, default=compute_type_default, label="Compute type")

                          gr.HTML("<hr></h2>")
                          bVIDEO_OUTPUT_NAME = gr.Textbox(label="Переведенное имя файла" ,value="video_output.mp4", info="The name of the output file")
                          bPREVIEW = gr.Checkbox(label="Превью", info="Предварительный просмотр сокращает видео до 10 секунд в целях тестирования. Пожалуйста, отключите его, чтобы получить полную продолжительность видео.")

            with gr.Column(variant='compact'):
                with gr.Row():
                    text_button = gr.Button("ПЕРЕВОД")
                with gr.Row():
                    blink_output = gr.outputs.File(label="СКАЧАТЬ ПЕРЕВЕДЕННОЕ ВИДЕО") # gr.Video()


                bline_ = gr.HTML("<hr></h2>")
                if os.getenv("YOUR_HF_TOKEN") == None or os.getenv("YOUR_HF_TOKEN") == "":
                  bHFKEY = gr.Textbox(visible= True, label="HF Token", info="Важным шагом является принятие лицензионного соглашения на использование Pyannote. Вам необходимо иметь учетную запись на Hugging Face и принять лицензию на использование моделей: https://huggingface.co/pyannote/speaker-diarization и https://huggingface.co/pyannote/segmentation. Получите свой ТОКЕН здесь: https://hf.co/settings/tokens", placeholder="Токен идет сюда...")
                else:
                  bHFKEY = gr.Textbox(visible= False, label="HF Token", info="Важным шагом является принятие лицензионного соглашения на использование Pyannote. Вам необходимо иметь учетную запись на Hugging Face и принять лицензию на использование моделей: https://huggingface.co/pyannote/speaker-diarization и https://huggingface.co/pyannote/segmentation. Получите свой ТОКЕН здесь: https://hf.co/settings/tokens", placeholder="Токен идет сюда...")

                gr.Examples(
                    examples=[
                        [
                            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                            "",
                            False,
                            "large-v2",
                            16,
                            "float16",
                            "Japanese (ja)",
                            "English (en)",
                            1,
                            2,
                            'en-CA-ClaraNeural-Female',
                            'en-AU-WilliamNeural-Male',
                            'en-GB-ThomasNeural-Male',
                            'en-GB-SoniaNeural-Female',
                            'en-NZ-MitchellNeural-Male',
                            'en-GB-MaisieNeural-Female',
                            "video_output.mp4",
                            'Adjusting volumes and mixing audio',
                        ],
                    ],
                    fn=translate_from_video,
                    inputs=[
                    blink_input,
                    bHFKEY,
                    bPREVIEW,
                    bWHISPER_MODEL_SIZE,
                    bbatch_size,
                    bcompute_type,
                    bSOURCE_LANGUAGE,
                    bTRANSLATE_AUDIO_TO,
                    bmin_speakers,
                    bmax_speakers,
                    btts_voice00,
                    btts_voice01,
                    btts_voice02,
                    btts_voice03,
                    btts_voice04,
                    btts_voice05,
                    bVIDEO_OUTPUT_NAME,
                    bAUDIO_MIX
                    ],
                    outputs=[blink_output],
                    cache_examples=False,
                )


    with gr.Tab("Кастомный голос при помощи RVC"):
        with gr.Column():
          with gr.Accordion("Получить RVC модели", open=True):
            url_links = gr.Textbox(label="Ссылки", value="",info="Автоматически загружайте модели RVC по URL-адресу. Вы можете использовать ссылки с HuggingFace или Drive, а также включить несколько ссылок, каждая из которых разделена запятой. Пример: https://huggingface.co/sail-rvc/yoimiya-jp/blob/main/model.pth, https://huggingface.co/sail-rvc/yoimiya-jp/blob/main/model.index", placeholder="Ссылочки в сибирь тут...", lines=1)
            download_finish = gr.HTML()
            download_button = gr.Button("СКАЧАТЬ МОДЕЛЬ")

            def update_models():
              models, index_paths = upload_model_list()
              for i in range(8):                      
                dict_models = {
                    f'model_voice_path{i:02d}': gr.update(choices=models) for i in range(8)
                }
                dict_index = {
                    f'file_index2_{i:02d}': gr.update(choices=index_paths) for i in range(8)
                }
                dict_changes = {**dict_models, **dict_index}
                return [value for value in dict_changes.values()]

        with gr.Column():
          with gr.Accordion("Замена войса: TTS в RVC", open=False):
            with gr.Column(variant='compact'):
              with gr.Column():
                gr.Markdown("### 1. Чтобы разрешить его использование, отметьте его как включенный.")
                enable_custom_voice = gr.Checkbox(label="ВКЛЮЧЕННЫЙ", info="Установите этот флажок, чтобы разрешить использование моделей.")
                enable_custom_voice.change(custom_model_voice_enable, [enable_custom_voice], [])

                gr.Markdown("### 2. Выберите голос, который будет применяться к каждому TTS каждого соответствующего спикера, и примените настройки.")
                gr.Markdown('В зависимости от того, сколько спикеров TTS вы будете использовать, каждому из них потребуется своя модель. Дополнительно есть вспомогательный, если по каким-то причинам спикер не определяется корректно.')
                gr.Markdown("Голос для применения к первому говорящему.")
                with gr.Row():
                  model_voice_path00 = gr.Dropdown(models, label = 'Модель-1', visible=True, interactive= True)
                  file_index2_00 = gr.Dropdown(index_paths, label = 'Индекс-1', visible=True, interactive= True)
                  name_transpose00 = gr.Number(label = 'Транспонирование-1', value=0, visible=True, interactive= True)
                gr.HTML("<hr></h2>")
                gr.Markdown("Голос для применения ко второму спикеру.")
                with gr.Row():
                  model_voice_path01 = gr.Dropdown(models, label='Модель-2', visible=True, interactive=True)
                  file_index2_01 = gr.Dropdown(index_paths, label='Индекс-2', visible=True, interactive=True)
                  name_transpose01 = gr.Number(label='Транспонирование-2', value=0, visible=True, interactive=True)
                gr.HTML("<hr></h2>")
                gr.Markdown("Голос для применения к третьему спикеру")
                with gr.Row():
                  model_voice_path02 = gr.Dropdown(models, label='Модель-3', visible=True, interactive=True)
                  file_index2_02 = gr.Dropdown(index_paths, label='Индекс-3', visible=True, interactive=True)
                  name_transpose02 = gr.Number(label='Транспонирование-3', value=0, visible=True, interactive=True)
                gr.HTML("<hr></h2>")
                gr.Markdown("Голос для применения к четвертому спикеру.")
                with gr.Row():
                  model_voice_path03 = gr.Dropdown(models, label='Модель-4', visible=True, interactive=True)
                  file_index2_03 = gr.Dropdown(index_paths, label='Индекс-4', visible=True, interactive=True)
                  name_transpose03 = gr.Number(label='Транспонирование-4', value=0, visible=True, interactive=True)
                gr.HTML("<hr></h2>")
                gr.Markdown("Голос для применения к пятому спикеру.")
                with gr.Row():
                  model_voice_path04 = gr.Dropdown(models, label='Модель-5', visible=True, interactive=True)
                  file_index2_04 = gr.Dropdown(index_paths, label='Индекс-5', visible=True, interactive=True)
                  name_transpose04 = gr.Number(label='Транспонирование-5', value=0, visible=True, interactive=True)
                gr.HTML("<hr></h2>")
                gr.Markdown("VГолос для применения к шестому спикеру.")
                with gr.Row():
                  model_voice_path05 = gr.Dropdown(models, label='Модель-6', visible=True, interactive=True)
                  file_index2_05 = gr.Dropdown(index_paths, label='Индекс-6', visible=True, interactive=True)
                  name_transpose05 = gr.Number(label='Транспонирование-6', value=0, visible=True, interactive=True)
                gr.HTML("<hr></h2>")
                gr.Markdown("- Голос, применяемый в случае, если говорящий не обнаружен.")
                with gr.Row():
                  model_voice_path06 = gr.Dropdown(models, label='Модель-Aux', visible=True, interactive=True)
                  file_index2_06 = gr.Dropdown(index_paths, label='Индекс-Aux', visible=True, interactive=True)
                  name_transpose06 = gr.Number(label='Транспонирование-Aux', value=0, visible=True, interactive=True)
                gr.HTML("<hr></h2>")
                with gr.Row():
                  f0_method_global = gr.Dropdown(f0_methods_voice, value='pm', label = 'Глобальный метод F0', visible=True, interactive= True)

            with gr.Row(variant='compact'):
              button_config = gr.Button("Принять конфигурацию")

              confirm_conf = gr.HTML()

            button_config.click(voices.apply_conf, inputs=[
                f0_method_global,
                model_voice_path00, name_transpose00, file_index2_00,
                model_voice_path01, name_transpose01, file_index2_01,
                model_voice_path02, name_transpose02, file_index2_02,
                model_voice_path03, name_transpose03, file_index2_03,
                model_voice_path04, name_transpose04, file_index2_04,
                model_voice_path05, name_transpose05, file_index2_05,
                model_voice_path06, name_transpose06, file_index2_06,
                ], outputs=[confirm_conf])


          with gr.Column():
                with gr.Accordion("Тест RVC", open=False):

                  with gr.Row(variant='compact'):
                    text_test = gr.Textbox(label="Текст", value="This is an example",info="напиши свойт екст", placeholder="...", lines=5)
                    with gr.Column(): 
                      tts_test = gr.Dropdown(list_tts, value='en-GB-ThomasNeural-Male', label = 'TTS', visible=True, interactive= True)
                      model_voice_path07 = gr.Dropdown(models, label = 'Модель', visible=True, interactive= True) #value=''
                      file_index2_07 = gr.Dropdown(index_paths, label = 'Индекс', visible=True, interactive= True) #value=''
                      transpose_test = gr.Number(label = 'Транспонирование', value=0, visible=True, interactive= True, info="целое число, количество полутонов, повышение на октаву: 12, понижение на октаву: -12")
                      f0method_test = gr.Dropdown(f0_methods_voice, value='pm', label = 'F0 метод', visible=True, interactive= True) 
                  with gr.Row(variant='compact'):
                    button_test = gr.Button("Тест аудио")

                  with gr.Column():
                    with gr.Row():
                      original_ttsvoice = gr.Audio()
                      ttsvoice = gr.Audio()

                    button_test.click(voices.make_test, inputs=[
                        text_test,
                        tts_test,
                        model_voice_path07,
                        file_index2_07,
                        transpose_test,
                        f0method_test,
                        ], outputs=[ttsvoice, original_ttsvoice])

                download_button.click(download_list, [url_links], [download_finish]).then(update_models, [], 
                                  [
                                    model_voice_path00, model_voice_path01, model_voice_path02, model_voice_path03, model_voice_path04, model_voice_path05, model_voice_path06, model_voice_path07,
                                    file_index2_00, file_index2_01, file_index2_02, file_index2_03, file_index2_04, file_index2_05, file_index2_06, file_index2_07
                                  ])


    with gr.Tab("Помощь"):
        gr.Markdown(tutorial)
        gr.Markdown(news)

    with gr.Accordion("Логи", open = False):
        logs = gr.Textbox()
        demo.load(read_logs, None, logs, every=1)

    # run
    video_button.click(translate_from_video, inputs=[
        video_input,
        HFKEY,
        PREVIEW,
        WHISPER_MODEL_SIZE,
        batch_size,
        compute_type,
        SOURCE_LANGUAGE,
        TRANSLATE_AUDIO_TO,
        min_speakers,
        max_speakers,
        tts_voice00,
        tts_voice01,
        tts_voice02,
        tts_voice03,
        tts_voice04,
        tts_voice05,
        VIDEO_OUTPUT_NAME,
        AUDIO_MIX,
        ], outputs=video_output)
    text_button.click(translate_from_video, inputs=[
        blink_input,
        bHFKEY,
        bPREVIEW,
        bWHISPER_MODEL_SIZE,
        bbatch_size,
        bcompute_type,
        bSOURCE_LANGUAGE,
        bTRANSLATE_AUDIO_TO,
        bmin_speakers,
        bmax_speakers,
        btts_voice00,
        btts_voice01,
        btts_voice02,
        btts_voice03,
        btts_voice04,
        btts_voice05,
        bVIDEO_OUTPUT_NAME,
        bAUDIO_MIX,
        ], outputs=blink_output)

#demo.launch(debug=True, enable_queue=True)
demo.launch(share=True, enable_queue=True, quiet=True, debug=False)
