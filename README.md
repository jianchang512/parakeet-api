# Parakeet-API: 高性能本地语音转录服务

**parakeet-api** 项目  是一个基于 **NVIDIA Parakeet-tdt-0.6b** 模型的本地语音转录服务。它提供了一个与 `OpenAI API` 兼容的接口和一个简洁的 Web 用户界面，让您能够轻松、快速地将任何音视频文件转换为高精度的 SRT 字幕,同时可适配`pyVideoTrans v3.72+`。



## ✨ Parakeet-API 核心优势

*   🚀 **极致的速度与性能**: Parakeet 模型经过高度优化，尤其在配备 NVIDIA GPU 的环境下，其转录速度非常快，特别适合处理大量或长时长的音视频文件。
*   🎯 **精准的时间戳**: 采用先进的 Transducer (TDT) 技术，生成的 SRT 字幕时间戳非常精确，能与音频流完美对齐，非常适合视频字幕制作。
*   💰 **完全免费，无限使用**: 在您自己的硬件上运行，无需支付任何 API 调用费用，也没有使用时长的限制。
*   🌐 **灵活的访问方式**: 提供直观的 Web 界面和标准化的 API 接口，可轻松集成到 `pyVideoTrans` 等现有工作流中。

## 🛠️ 安装与配置指南

本项目支持 Windows, macOS 和 Linux。请按照以下步骤进行安装和配置。

### 步骤 0: 配置 python3.10 环境

如果你本机无python3，请照此教程安装:  [https://pvt9.com/_posts/pythoninstall](https://pvt9.com/_posts/pythoninstall)

### 步骤 1: 准备 FFmpeg

本项目使用 `ffmpeg` 进行音视频格式预处理。

*   **Windows (推荐)**:
    1.  从 [FFmpeg github 仓库下载](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip)  解压后得到`ffmpeg.exe`。
    2.  将下载的 **`ffmpeg.exe` 文件直接放置在本项目根目录** (与 `app.py` 文件在同一级)，程序会自动检测并使用它，无需配置环境变量。

*   **macOS (使用 Homebrew)**:
    ```bash
    brew install ffmpeg
    ```
*   **Linux (Debian/Ubuntu)**:
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```

### 步骤 2: 创建 Python 虚拟环境并安装依赖

1.  **下载或克隆本项目代码**到您的本地计算机(建议放在非系统盘的英文或数字文件夹内)。
2.  **打开终端或命令行工具**，并进入项目根目录(windows上直接在文件夹地址栏里输入`cmd`回车即可)。
![](https://pvtr2.pyvideotrans.com/1751277781831_image.png)

3.  **创建虚拟环境**: `python -m venv venv`
4.  **激活虚拟环境**:
    *   **Windows (CMD/PowerShell)**: `.\venv\Scripts\activate`
    *   **macOS / Linux (Bash/Zsh)**: `source venv/bin/activate`

5.  **安装依赖库**:
    *   **如果您没有 NVIDIA 显卡 (仅使用 CPU)**:
        ```bash
        pip install -r requirements.txt
        ```

    *   **如果您有 NVIDIA 显卡 (使用 GPU 加速)**:
        a. 确保您已安装最新的 [NVIDIA 驱动](https://www.nvidia.com/Download/index.aspx) 和相应的 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)。
        b. 卸载可能存在的旧版 PyTorch: `pip uninstall -y torch`
        c. 安装与您的 CUDA 版本匹配的 PyTorch (以 CUDA 12.6 为例):
        ```bash
        pip install torch --index-url https://download.pytorch.org/whl/cu126
        ```

### 步骤 3: 启动服务

在**已激活虚拟环境**的终端中，运行以下命令：

```bash
python app.py
```

您将看到服务启动的提示。**首次运行**会下载模型（约1.2GB），请耐心等待。
![](https://pvtr2.pyvideotrans.com/1751277964995_image.png)

如果出现一堆提示，无需介意，
![](https://pvtr2.pyvideotrans.com/1751278084962_image.png)

**启动成功界面**

![](https://pvtr2.pyvideotrans.com/1751278233994_image.png)


## 🚀 使用方法

### 方法 1: 使用 Web 界面

1.  在浏览器中打开：[http://127.0.0.1:5092](http://127.0.0.1:5092)
2.  拖拽或点击上传您的音视频文件。
3.  点击 **"开始转录"**，等待处理完成即可在下方看到并下载 SRT 字幕。

![](https://pvtr2.pyvideotrans.com/1751278256778_image.png)



### 方法 2: API 调用 (Python 示例)

使用 `openai` 库可以轻松调用本服务。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5092/v1",
    api_key="any-key",
)

with open("your_audio.mp3", "rb") as audio_file:
    srt_result = client.audio.transcriptions.create(
        model="parakeet",
        file=audio_file,
        response_format="srt"
    )
print(srt_result)
```

### 方法 3: 集成到 pyVideoTrans (推荐)

Parakeet-API 可与视频翻译工具 `pyVideoTrans` (v3.72及以上版本) 无缝集成。

![](https://pvtr2.pyvideotrans.com/1751278281473_image.png)

1.  确保您的 `parakeet-api` 服务正在本地运行。
2.  打开 `pyVideoTrans` 软件。
3.  在菜单栏中，选择 **语音识别(R) -> Nvidia parakeet-tdt**。
4.  在弹出的配置窗口中，将 **"http地址"** 设置为：`http://127.0.0.1:5092/v1`
5.  点击 **"保存"**，即可开始使用。

