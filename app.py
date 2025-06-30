import os,sys
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

host = '127.0.0.1'
port = 5092
threads = 4

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
    兼容 OpenAI 的语音识别接口。
    接收 'file' 字段中的音视频文件。
    固定返回 SRT 格式的转录结果。
    """
    if 'file' not in request.files:
        return jsonify({"error": "请求中未找到文件部分"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    if not shutil.which('ffmpeg'):
        return jsonify({"error": "请先安装ffmpeg"}), 400
    if file:
        original_filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        # 保存原始上传文件
        temp_original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{original_filename}")
        file.save(temp_original_path)

        # 准备转换后的 WAV 文件路径
        target_wav_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}.wav")
        
        # 使用 try...finally 确保临时文件被清理
        try:
            # --- 使用 FFmpeg 将任何音视频文件转换为 16kHz, 1通道的 WAV ---
            print(f"[{unique_id}] 正在转换文件: {original_filename}")
            # -y: 覆盖输出文件 -i: 输入文件 -ac: 音频通道数 -ar: 采样率
            ffmpeg_command = [
                'ffmpeg',
                '-y',
                '-i', temp_original_path,
                '-ac', '1',
                '-ar', '16000',
                target_wav_path
            ]
            
            # 执行 ffmpeg 命令
            result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg 错误: {result.stderr}")
                return jsonify({"error": "文件转换失败", "details": result.stderr}), 500
            
            print(f"[{unique_id}] 文件转换成功，开始转录...")
            
            # --- 使用 NeMo 模型进行转录 ---
            # timestamps=True 会返回分词和分段的时间戳
            output = asr_model.transcribe([target_wav_path], timestamps=True)

            if not output or not output[0].timestamp or 'segment' not in output[0].timestamp:
                return jsonify({"error": "转录失败，模型未返回有效时间戳"}), 500

            segment_timestamps = output[0].timestamp['segment']
            print(f"[{unique_id}] 转录完成。")
            
            # --- 将结果转换为 SRT 格式 ---
            srt_result = segments_to_srt(segment_timestamps)
            
            # 返回 SRT 格式的响应，MIME 类型为 text/plain 方便前端处理
            return Response(srt_result, mimetype='text/plain')

        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            return jsonify({"error": "服务器内部错误", "details": str(e)}), 500
        finally:
            # --- 清理临时文件 ---
            if os.path.exists(temp_original_path):
                os.remove(temp_original_path)
            if os.path.exists(target_wav_path):
                os.remove(target_wav_path)
            print(f"[{unique_id}] 临时文件已清理。")

# --- Waitress 服务器启动 ---
if __name__ == '__main__':

    print(f"🚀 服务器启动中...")
    print(f"访问前端页面: http://127.0.0.1:{port}")
    print(f"API 端点: POST http://{host}:{port}/v1/audio/transcriptions")
    print(f"服务将使用 {threads} 个线程运行。")
    serve(app, host=host, port=port, threads=threads)