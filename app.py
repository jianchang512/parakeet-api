host = '127.0.0.1'
port = 5092
threads = 4
MIN_DURATION=5#小于等于5s的输入音视频，将强制仅仅返回一条纯文本
import signal

# Windows 兼容补丁：解决 NeMo 缺少 SIGKILL 的问题
if not hasattr(signal, "SIGKILL"):
    signal.SIGKILL = signal.SIGTERM

import os,sys,json,math,re,threading

from pathlib import Path
ROOT_DIR=Path(os.getcwd()).as_posix()
MODEL_DIR=ROOT_DIR + "/models"
os.environ['HF_HOME'] = MODEL_DIR
os.environ['HF_HUB_CACHE'] = MODEL_DIR
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = "3600"
if sys.platform == 'win32':
    os.environ['PATH'] = ROOT_DIR + f';{ROOT_DIR}/ffmpeg;' + os.environ['PATH']
import shutil
import uuid
import subprocess
import datetime
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, Response
from waitress import serve
import nemo.collections.asr as nemo_asr
from ten_vad import TenVad
import scipy.io.wavfile as Wavfile
import torch


TEMP_DIR=f'{ROOT_DIR}/temp_uploads'
# 确保临时上传目录存在
if not os.path.exists('temp_uploads'):
    os.makedirs('temp_uploads')


try:
    # 这一步会下载并加载模型，需要较长时间和网络连接
    print("\n开始检测模型 parakeet-tdt-0.6b-v3 是否存在，若不存在将下载")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="nvidia/parakeet-tdt-0.6b-v3")
    print("\n开始检测模型 parakeet-tdt_ctc-0.6b-ja 是否存在，若不存在将下载")
    snapshot_download( repo_id="nvidia/parakeet-tdt_ctc-0.6b-ja")
    print("\n开始检测模型 parakeet-ctc-0.6b-Vietnamese 是否存在，若不存在将下载")
    snapshot_download( repo_id="nvidia/parakeet-ctc-0.6b-Vietnamese",revision="5be0ba9c9d4528b6c3a17c56b0b38c15fea9c3d6")
except Exception as e:
    print(e)
    sys.exit()
    
print("="*50)


# --- Flask 应用初始化 ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 20000 * 1024 * 1024  

# --- 辅助函数 ---
def get_audio_duration(file_path: str) -> float:
    """使用 ffprobe 获取音频文件的时长（秒）"""
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"无法获取文件 '{file_path}' 的时长: {e}")
        return 0.0

def format_srt_time(seconds: float) -> str:
    """将秒数格式化为 SRT 时间戳格式 HH:MM:SS,ms"""
    delta = datetime.timedelta(seconds=seconds)
    # 格式化为 0:00:05.123000
    s = str(delta)
    # 分割秒和微秒
    if '.' in s:
        parts = s.split('.')
        integer_part = parts[0]
        fractional_part = parts[1][:3] # 取前三位毫秒
    else:
        integer_part = s
        fractional_part = "000"

    # 填充小时位
    if len(integer_part.split(':')) == 2:
        integer_part = "0:" + integer_part
    
    return f"{integer_part},{fractional_part}"


def segments_to_srt(segments: list) -> str:
    """将 NeMo 的分段时间戳转换为 SRT 格式字符串"""
    srt_content = []
    for i, segment in enumerate(segments):
        start_time = format_srt_time(segment['start'])
        end_time = format_srt_time(segment['end'])
        text = segment['segment'].strip()
        
        if text: # 仅添加有内容的字幕
            srt_content.append(str(i + 1))
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("") # 空行分隔
            
    return "\n".join(srt_content)



def cut_audio(audio_file):
    from pydub import AudioSegment
    import time
    dir_name = f"{TEMP_DIR}/clip_{time.time()}"
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    data = []
    speech_chunks = get_speech_timestamp(audio_file)
    speech_len = len(speech_chunks)

    audio = AudioSegment.from_wav(audio_file)
    # 对大于30s的强制拆分，小于1s的强制合并，防止报错
    check_1 = []
    # 裁切出的最小语音时长需符合 min_speech_duration_ms 要求，合并过短的
    min_speech_duration_ms = 1000
    for i, it in enumerate(speech_chunks):
        diff = it[1] - it[0]
        if diff < min_speech_duration_ms:
            # 距离前面空隙
            prev_diff = it[0] - check_1[-1][1] if len(check_1) > 0 else None
            # 距离下个空隙
            next_diff = speech_chunks[i + 1][0] - it[1] if i < speech_len - 1 else None
            if prev_diff is None and next_diff is not None:
                # 插入后边
                speech_chunks[i + 1][0] = it[0]
            elif prev_diff is not None and next_diff is None:
                # 前面延长
                check_1[-1][1] = it[1]
            elif prev_diff is not None and next_diff is not None:
                if prev_diff < next_diff:
                    check_1[-1][1] = it[1]
                else:
                    speech_chunks[i + 1][0] = it[0]
            else:
                check_1.append(it)
        elif diff < 30000:
            check_1.append(it)
        else:
            # 超过30s，一分为二
            off = diff // 2
            check_1.append([it[0], it[0] + off])
            check_1.append([it[0] + off, it[1]])
    speech_chunks = check_1

    for i, it in enumerate(speech_chunks):
        start_ms, end_ms = it[0], it[1]

        chunk = audio[start_ms:end_ms]
        file_name = f"{dir_name}/audio_{i}.wav"
        chunk.export(file_name, format="wav")
        tmp={"line": i + 1, "text": "", "start_time": start_ms, "end_time": end_ms, "file": file_name}
        tmp['time']=f'{format_srt_time(start_ms/1000.0)} --> {format_srt_time(end_ms/1000.0)}'
        data.append(tmp)

    return data


def _detect_raw_segments(data, threshold, min_silent_frames, max_speech_frames=None):
    """
    内部辅助函数：根据给定的静音阈值和最大长度检测语音片段。
    """
    hop_size = 256
    ten_vad_instance = TenVad(hop_size, threshold)
    num_frames = data.shape[0] // hop_size

    segments = []
    triggered = False
    speech_start_frame = 0
    silence_frame_count = 0

    for i in range(num_frames):
        audio_frame = data[i * hop_size: (i + 1) * hop_size]
        _, is_speech = ten_vad_instance.process(audio_frame)

        if triggered:
            current_speech_len = i - speech_start_frame
            if is_speech == 1:
                silence_frame_count = 0
            else:
                silence_frame_count += 1

            # 结束条件：1. 静音满足长度 2. (可选) 达到最大长度强制切断
            is_silence_timeout = silence_frame_count >= min_silent_frames
            is_max_timeout = max_speech_frames is not None and current_speech_len >= max_speech_frames

            if is_silence_timeout or is_max_timeout:
                if is_max_timeout:
                    end_frame = i
                else:
                    end_frame = i - silence_frame_count

                segments.append([speech_start_frame, end_frame])
                triggered = False
                silence_frame_count = 0
        else:
            if is_speech == 1:
                triggered = True
                speech_start_frame = i
                silence_frame_count = 0

    if triggered:
        end_frame = num_frames - silence_frame_count
        segments.append([speech_start_frame, end_frame])

    return segments


def get_speech_timestamp(input_wav):
    """
    优化后的 VAD 策略：
    1. 使用长静音阈值进行初步分割。
    2. 对过长的片段，降低静音阈值进行二次细分。
    3. 对仍超长的片段进行硬截断。
    """
    # --- 参数初始化 ---
    threshold = 0.5
    min_speech_duration_ms = 1000
    max_speech_duration_ms = 8000
    min_silent_duration_ms = 500
    frame_duration_ms = 16
    hop_size = 256

    try:
        sr, data = Wavfile.read(input_wav)
    except Exception as e:
        print(f"Error reading wav file: {e}")
        return []

    # --- 第一阶段：使用初始长静音阈值进行初步切分 (不设 max_speech 限制) ---
    min_sil_frames = min_silent_duration_ms / frame_duration_ms
    initial_segments = _detect_raw_segments(data, threshold, min_sil_frames, max_speech_frames=None)

    # --- 第二阶段：细化超长片段 ---
    refined_segments = []
    half_max_frames = (max_speech_duration_ms / 2) / frame_duration_ms
    max_frames_limit = max_speech_duration_ms / frame_duration_ms
    tighter_min_sil_frames = (min_silent_duration_ms / 2) / frame_duration_ms

    for s, e in initial_segments:
        duration = e - s
        if duration > half_max_frames:
            # 提取该段音频数据
            sub_data = data[s * hop_size: e * hop_size]
            # 使用减半的静音阈值重新检测，同时带上最大时长限制
            sub_segs = _detect_raw_segments(sub_data, threshold, tighter_min_sil_frames,
                                                 max_speech_frames=max_frames_limit)

            for ss, se in sub_segs:
                refined_segments.append([s + ss, s + se])
        else:
            refined_segments.append([s, e])

    if not refined_segments:
        return []

    # --- 第三阶段：毫秒转换 & 强制硬截断保护 ---
    # 即使二次细分，如果有人一口气说了30秒没停顿，仍需硬截断
    segments_ms = []
    for s, e in refined_segments:
        start_ms = int(s * frame_duration_ms)
        end_ms = int(e * frame_duration_ms)

        # 循环确保不超 max_speech_duration_ms
        curr_s = start_ms
        while (end_ms - curr_s) > max_speech_duration_ms:
            segments_ms.append([curr_s, curr_s + int(max_speech_duration_ms)])
            curr_s += int(max_speech_duration_ms)

        if end_ms - curr_s > 0:
            segments_ms.append([curr_s, end_ms])
    speech_len = len(segments_ms)
    if speech_len <= 1:
        return segments_ms
    

    check_1 = []

    # 不允许最小语音片段低于500ms，可能无法有效识别而报错
    min_speech_duration_ms = max(min_speech_duration_ms or 1000, 500)
    for i, it in enumerate(segments_ms):
        diff = it[1] - it[0]

        if diff >= min_speech_duration_ms:
            check_1.append(it)
        elif diff < 500:
            # 低于500ms的视为噪音，直接丢弃
            continue
        else:
            # 500-min_speech_duration_ms 之间的语音片段合并到邻近
            # 距离前面空隙
            prev_diff = it[0] - check_1[-1][1] if len(check_1) > 0 else None
            # 距离下个空隙
            next_diff = segments_ms[i + 1][0] - it[1] if i < speech_len - 1 else None
            if prev_diff is None and next_diff is not None:
                # 插入后边
                segments_ms[i + 1][0] = it[0]
            elif prev_diff is not None and next_diff is None:
                # 前面延长
                check_1[-1][1] = it[1]
            elif prev_diff is not None and next_diff is not None:
                if prev_diff < next_diff:
                    check_1[-1][1] = it[1]
                else:
                    segments_ms[i + 1][0] = it[0]
            else:
                check_1.append(it)
    return check_1

# --- Flask 路由 ---

@app.route('/')
def index():
    """提供前端上传页面"""
    return render_template('index.html')


@app.route('/v1/audio/transcriptions', methods=['POST'])
def transcribe_audio():
    """
    兼容 OpenAI 的语音识别接口，支持长音频分片处理。
    """
    # --- 1. 基本校验 ---
    if 'file' not in request.files:
        return jsonify({"error": "请求中未找到文件部分"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    if not shutil.which('ffmpeg'):
        return jsonify({"error": "FFmpeg 未安装或未在系统 PATH 中"}), 500
    if not shutil.which('ffprobe'):
        return jsonify({"error": "ffprobe 未安装或未在系统 PATH 中"}), 500
    # 用 model 参数传递特殊要求，例如 ----*---- 分隔字符串和json
    return_type = request.form.get('model', '')
    # prompt 用于获取语言
    language = request.form.get('prompt', 'default')
    model_list={
        "default":"parakeet-tdt-0.6b-v3",
        "ja":"parakeet-tdt_ctc-0.6b-ja",
        "vi":"parakeet-ctc-0.6b-Vietnamese"
    }
    if language not in model_list:
        language='default'

    original_filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    temp_original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{original_filename}")
    target_wav_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}.wav")
    
    # 用于清理所有临时文件的列表
    temp_files_to_clean = []

    try:
        # --- 2. 保存并统一转换为 16k 单声道 WAV ---
        file.save(temp_original_path)
        temp_files_to_clean.append(temp_original_path)
        
        print(f"[{unique_id}] 正在将 '{original_filename}' 转换为标准 WAV 格式...")
        ffmpeg_command = [
            'ffmpeg', '-y', '-i', temp_original_path,
            '-ac', '1', '-ar', '16000', target_wav_path
        ]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg 错误: {result.stderr}")
            return jsonify({"error": "文件转换失败", "details": result.stderr}), 500
        temp_files_to_clean.append(target_wav_path)


        total_duration = get_audio_duration(target_wav_path)
        if total_duration == 0:
            return jsonify({"error": "无法处理时长为0的音频"}), 400
        print(f'加载模型：{model_list[language]}')
        
        
        
        
        if total_duration<MIN_DURATION:
            chunk_paths=[{"line":  1, "text": "", "start_time": 0, "end_time": int(total_duration*1000), "file": target_wav_path}]
        else:
            chunk_paths=cut_audio(target_wav_path)
        
        if language=='vi':
            asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=f'{MODEL_DIR}/models--nvidia--parakeet-ctc-0.6b-Vietnamese/snapshots/5be0ba9c9d4528b6c3a17c56b0b38c15fea9c3d6/parakeet-ctc-0.6b-vi.nemo')
        else:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=f'nvidia/{model_list[language]}')
        # --- 4. 循环转录并合并结果 ---       
        
        for i, chunk_path in enumerate(chunk_paths):            
            # 对当前切片进行转录
            output = asr_model.transcribe([chunk_path['file']])
            print(f'{output=}')
            chunk_path['text']=output[0].text if hasattr(output[0],'text') else output[0][0]
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del asr_model
            import gc
            gc.collect()
        except:
            pass
        print(chunk_paths)
        if total_duration<MIN_DURATION:      
            return Response(chunk_paths[0]['text'], mimetype='text/plain')
        srt_list=[f'{it["line"]}\n{it["time"]}\n{it["text"]}' for it in chunk_paths]
        return Response("\n\n".join(srt_list), mimetype='text/plain')

    except Exception as e:
        print(f"处理过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "服务器内部错误", "details": str(e)}), 500



def openweb():
    import webbrowser,time
    time.sleep(5)
    webbrowser.open_new_tab(f'http://127.0.0.1:{port}')

# --- Waitress 服务器启动 ---
if __name__ == '__main__':

    print(f"服务器启动中...")
    print(f"访问前端页面: http://127.0.0.1:{port}")
    print(f"API 端点: POST http://{host}:{port}/v1/audio/transcriptions")
    print(f"服务将使用 {threads} 个线程运行。")
    threading.Thread(target=openweb).start()
    serve(app, host=host, port=port, threads=threads)