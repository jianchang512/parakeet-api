host = '127.0.0.1'
port = 5092
threads = 4
# 默认按照10分钟将音视频裁切为多段
CHUNK_MINITE=10


import os,sys,json,math
import shutil
import uuid
import subprocess
import datetime
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, render_template, Response
from waitress import serve
from pathlib import Path
ROOT_DIR=Path(os.getcwd()).as_posix()
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HOME'] = ROOT_DIR + "/models"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'
if sys.platform == 'win32':
    os.environ['PATH'] = ROOT_DIR + f';{ROOT_DIR}/ffmpeg;' + os.environ['PATH']


# --- 全局设置与模型预加载 ---

# 确保临时上传目录存在
if not os.path.exists('temp_uploads'):
    os.makedirs('temp_uploads')

print("="*50)
print("正在加载 NVIDIA NeMo ASR 模型...若不存在将下载")
print("模型名称: nvidia/parakeet-tdt-0.6b-v2")
print("这可能需要几分钟时间，请耐心等待...")
print("加载完毕后将启动 API 服务...")

try:
    # 这一步会下载并加载模型，需要较长时间和网络连接
    import nemo.collections.asr as nemo_asr
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    print("✅ NeMo ASR 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("请确保已正确安装 'nemo_toolkit[asr]' 及其依赖，并有可用的网络连接。")
    exit(1)

print("="*50)


# --- Flask 应用初始化 ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  

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

    model_name = request.form.get('model', '')
    print(f"接收到请求，指定模型参数: '{model_name}'")

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

        # --- 3. 音频切片 (Chunking) ---
        CHUNK_DURATION_SECONDS = CHUNK_MINITE * 60  
        total_duration = get_audio_duration(target_wav_path)
        if total_duration == 0:
            return jsonify({"error": "无法处理时长为0的音频"}), 400

        num_chunks = math.ceil(total_duration / CHUNK_DURATION_SECONDS)
        chunk_paths = []
        print(f"[{unique_id}] 文件总时长: {total_duration:.2f}s. 将切分为 {num_chunks} 个片段。")
        
        if num_chunks>1:
            for i in range(num_chunks):
                start_time = i * CHUNK_DURATION_SECONDS
                chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_chunk_{i}.wav")
                chunk_paths.append(chunk_path)
                temp_files_to_clean.append(chunk_path)
                
                print(f"[{unique_id}] 正在创建切片 {i+1}/{num_chunks}...")
                chunk_command = [
                    'ffmpeg', '-y', '-i', target_wav_path,
                    '-ss', str(start_time),
                    '-t', str(CHUNK_DURATION_SECONDS),
                    '-c', 'copy',
                    chunk_path
                ]
                subprocess.run(chunk_command, capture_output=True, text=True)
        else:
            chunk_paths.append(target_wav_path)
            
        # --- 4. 循环转录并合并结果 ---
        all_segments = []
        all_words = []
        cumulative_time_offset = 0.0

        for i, chunk_path in enumerate(chunk_paths):
            print(f"[{unique_id}] 正在转录切片 {i+1}/{num_chunks}...")
            
            # 对当前切片进行转录
            output = asr_model.transcribe([chunk_path], timestamps=True)
            
            if output and output[0].timestamp:
                # 修正并收集 segment 时间戳
                if 'segment' in output[0].timestamp:
                    for seg in output[0].timestamp['segment']:
                        seg['start'] += cumulative_time_offset
                        seg['end'] += cumulative_time_offset
                        all_segments.append(seg)
                
                # 修正并收集 word 时间戳
                if 'word' in output[0].timestamp:
                     for word in output[0].timestamp['word']:
                        word['start'] += cumulative_time_offset
                        word['end'] += cumulative_time_offset
                        all_words.append(word)

            # 更新下一个切片的时间偏移量
            # 使用实际切片时长来更新，更精确
            chunk_actual_duration = get_audio_duration(chunk_path)
            cumulative_time_offset += chunk_actual_duration

        print(f"[{unique_id}] 所有切片转录完成，正在合并结果。")

        # --- 5. 格式化最终输出 ---
        if not all_segments:
            return jsonify({"error": "转录失败，模型未返回任何有效内容"}), 500

        srt_result = segments_to_srt(all_segments)
        
        if model_name == 'parakeet_srt_words':
            json_str_list = [
                {"start": it['start'], "end": it['end'], "word": it['word']} 
                for it in all_words
            ]
            srt_result += "----..----" + json.dumps(json_str_list)
        
        return Response(srt_result, mimetype='text/plain')

    except Exception as e:
        print(f"处理过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "服务器内部错误", "details": str(e)}), 500
    finally:
        # --- 6. 清理所有临时文件 ---
        print(f"[{unique_id}] 清理临时文件...")
        for f_path in temp_files_to_clean:
            if os.path.exists(f_path):
                os.remove(f_path)
        print(f"[{unique_id}] 临时文件已清理。")

# --- Waitress 服务器启动 ---
if __name__ == '__main__':

    print(f"🚀 服务器启动中...")
    print(f"访问前端页面: http://127.0.0.1:{port}")
    print(f"API 端点: POST http://{host}:{port}/v1/audio/transcriptions")
    print(f"服务将使用 {threads} 个线程运行。")
    serve(app, host=host, port=port, threads=threads)