# PaddleOCR 智能文档处理工具包 / PaddleOCR Intelligent Document Processing Toolkit

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-3.0+-green.svg)](https://www.paddlepaddle.org.cn/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[English](#english) | [中文](#中文)

---

## 中文

基于 PaddleOCR 3.3 和 PP-StructureV3 的智能文档处理工具包，提供完整的文档分析、OCR识别、表格处理、公式识别等功能。支持 PDF 转图片、文档布局检测、文字识别、表格结构化、印章检测等多种文档处理任务。

### 📖 快速导航

#### 🎯 核心文档

- **[📚 PaddleOCR 模块详细介绍](docs/Learn_PaddleOCR_From_Scratch.md)** - PP-StructureV3 的7个核心模块详解
- **[⚙️ 项目安装部署全流程](docs/note.md)** - 从环境搭建到云端部署的完整指南

#### 🚀 主要特性

- 🔍 **智能文档分析**: 基于 PP-StructureV3 的端到端文档处理
- 📄 **PDF 处理**: 高质量 PDF 转图片，支持批量处理
- 🎯 **布局检测**: 精确识别文档中的文字、表格、图片、公式等区域
- 📝 **OCR 识别**: 高精度文字识别，支持多语言
- 📊 **表格处理**: 表格结构识别和内容提取，支持复杂表格
- 🧮 **公式识别**: 数学公式检测和识别，输出LaTeX格式
- 🔐 **印章检测**: 印章区域检测和提取
- 🚀 **GPU 加速**: 支持 CUDA、TensorRT 加速推理
- 📊 **性能监控**: 内置性能监控和资源使用统计

### 🛠️ 环境要求

- Python 3.8+
- PaddlePaddle 3.0+
- CUDA 11.8+ (GPU 版本，TensorRT 加速必需)
- 8GB+ RAM (推荐 16GB+)

### 📦 快速安装

#### 方式一：使用 uv (推荐)

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境
uv venv .venv --python 3.10
source .venv/bin/activate

# 设置环境变量
export PYTHONPATH="$PWD:$PYTHONPATH"

# 安装 PaddlePaddle GPU 版本
uv pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 安装 PaddleOCR
uv pip install "paddleocr[all]"

# 克隆项目并安装
git clone -b release/3.3 https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
uv pip install -r requirements.txt
uv pip install -e .
```

#### 方式二：使用 conda

```bash
# 创建环境
conda create -n paddleocr3 python=3.10 -y
conda activate paddleocr3

# 设置环境变量
export PYTHONPATH="$PWD:$PYTHONPATH"

# 安装依赖
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install paddleocr==3.0.1.0 paddlex==3.0.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 克隆项目
git clone -b release/3.3 https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip install -r requirements.txt
pip install -e .
```

### 🚀 快速开始

#### 基础 OCR 识别

```python
from paddleocr import PaddleOCR

# 初始化 OCR
ocr = PaddleOCR(
    device="gpu:0",              # 使用 GPU
    use_tensorrt=True,           # 启用 TensorRT 加速
    precision="fp16"             # 使用半精度
)

# 识别图片
result = ocr.predict("your_image.jpg")
for res in result:
    res.print()
    res.save_to_json("output/")
```

#### 完整文档处理 (PP-StructureV3)

```python
from paddleocr import PPStructureV3

# 初始化完整文档处理管道
pipeline = PPStructureV3(
    device="gpu:0",
    enable_hpi=True,             # 启用高性能推理
    use_tensorrt=True,           # 使用 TensorRT 加速
    precision="fp16"             # 使用半精度
)

# 处理文档
results = pipeline.predict("document.pdf")

# 提取 Markdown 内容
markdown_list = []
for result in results:
    result.print()
    result.save_to_json("output/")
    result.save_to_markdown("output/")
    markdown_list.append(result.markdown)

# 合并所有页面的 Markdown
markdown_text = pipeline.concatenate_markdown_pages(markdown_list)
with open("output/document.md", "w", encoding="utf-8") as f:
    f.write(markdown_text)
```

### 📁 项目结构

```
PaddleOCR-Toolkit/
├── 📖 docs/                           # 📚 核心文档目录
│   ├── Learn_PaddleOCR_From_Scratch.md  # 🎯 PP-StructureV3 模块详解
│   ├── note.md                          # ⚙️ 安装部署完整指南
│   └── imgs/                            # 📷 文档配置图片
├── 🧪 tests/                          # 测试用例和示例
│   ├── OCR/                            # OCR 测试
│   ├── Layout_Detection/               # 布局检测测试
│   ├── Chart/                          # 表格处理测试
│   ├── Formula/                        # 公式识别测试
│   ├── Seal/                           # 印章检测测试
│   └── PreProcess/                     # 预处理测试
├── 🔧 main/                           # 主要功能模块
│   ├── Complete/                       # 完整文档处理
│   └── monitor/                        # 性能监控
├── 🛠️ utils/                          # 工具函数
│   ├── pdf2image.py                    # PDF 转图片工具
│   └── performance_monitor.py          # 性能监控
├── 📊 data/                           # 数据目录
├── 🏗️ PaddleOCR/                      # PaddleOCR 源码
└── 📋 requirements.txt                 # Python 依赖
```

### 🎯 核心功能模块

1. **版面区域检测** - 识别文档中的不同元素（文字、表格、图片、公式等）
2. **通用 OCR 子产线** - 文字检测和识别，支持多语言
3. **表格识别 v2** - 表格结构识别和内容提取，输出HTML格式
4. **公式识别** - 数学公式检测和识别，输出LaTeX格式
5. **印章文本识别** - 印章区域检测和文字识别
6. **文档图像预处理** - 文档方向分类和形变矫正

### 🧪 运行测试

```bash
# OCR 文字识别测试
python tests/OCR/test.py

# 表格识别测试
python tests/Chart/test.py

# 布局检测测试
python tests/Layout_Detection/test.py

# 公式识别测试
python tests/Formula/test.py

# 印章检测测试
python tests/Seal/test.py

# 图像预处理测试
python tests/PreProcess/test.py
```

### 🌐 服务化部署

```bash
# 启动 OCR 服务
paddleocr_mcp --pipeline OCR --ppocr_source local --port 8090 --http

# 启动完整文档处理服务
paddleocr_mcp --pipeline PP-StructureV3 --ppocr_source local --host 0.0.0.0 --port 8090 --http --pipeline_config PaddleOCR.yaml --device gpu
```

---

## English

An intelligent document processing toolkit based on PaddleOCR 3.3 and PP-StructureV3, providing comprehensive document analysis, OCR recognition, table processing, formula recognition, and more. Supports PDF to image conversion, document layout detection, text recognition, table structuring, seal detection, and various document processing tasks.

### 📖 Quick Navigation

#### 🎯 Core Documentation

- **[📚 PaddleOCR Module Detailed Guide](docs/Learn_PaddleOCR_From_Scratch.md)** - Comprehensive guide to PP-StructureV3's 7 core modules
- **[⚙️ Complete Installation & Deployment Guide](docs/note.md)** - Full guide from environment setup to cloud deployment

#### 🚀 Key Features

- 🔍 **Intelligent Document Analysis**: End-to-end document processing based on PP-StructureV3
- 📄 **PDF Processing**: High-quality PDF to image conversion with batch processing
- 🎯 **Layout Detection**: Precise identification of text, tables, images, formulas in documents
- 📝 **OCR Recognition**: High-accuracy text recognition with multi-language support
- 📊 **Table Processing**: Table structure recognition and content extraction for complex tables
- 🧮 **Formula Recognition**: Mathematical formula detection and recognition with LaTeX output
- 🔐 **Seal Detection**: Seal area detection and extraction
- 🚀 **GPU Acceleration**: Support for CUDA and TensorRT accelerated inference
- 📊 **Performance Monitoring**: Built-in performance monitoring and resource usage statistics

### 🛠️ Requirements

- Python 3.8+
- PaddlePaddle 3.0+
- CUDA 11.8+ (for GPU version, required for TensorRT acceleration)
- 8GB+ RAM (16GB+ recommended)

### 📦 Quick Installation

#### Method 1: Using uv (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv .venv --python 3.10
source .venv/bin/activate

# Set environment variables
export PYTHONPATH="$PWD:$PYTHONPATH"

# Install PaddlePaddle GPU version
uv pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Install PaddleOCR
uv pip install "paddleocr[all]"

# Clone project and install
git clone -b release/3.3 https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
uv pip install -r requirements.txt
uv pip install -e .
```

#### Method 2: Using conda

```bash
# Create environment
conda create -n paddleocr3 python=3.10 -y
conda activate paddleocr3

# Set environment variables
export PYTHONPATH="$PWD:$PYTHONPATH"

# Install dependencies
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install paddleocr==3.0.1.0 paddlex==3.0.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Clone project
git clone -b release/3.3 https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip install -r requirements.txt
pip install -e .
```

### 🚀 Quick Start

#### Basic OCR Recognition

```python
from paddleocr import PaddleOCR

# Initialize OCR
ocr = PaddleOCR(
    device="gpu:0",              # Use GPU
    use_tensorrt=True,           # Enable TensorRT acceleration
    precision="fp16"             # Use half precision
)

# Recognize image
result = ocr.predict("your_image.jpg")
for res in result:
    res.print()
    res.save_to_json("output/")
```

#### Complete Document Processing (PP-StructureV3)

```python
from paddleocr import PPStructureV3

# Initialize complete document processing pipeline
pipeline = PPStructureV3(
    device="gpu:0",
    enable_hpi=True,             # Enable high-performance inference
    use_tensorrt=True,           # Use TensorRT acceleration
    precision="fp16"             # Use half precision
)

# Process document
results = pipeline.predict("document.pdf")

# Extract Markdown content
markdown_list = []
for result in results:
    result.print()
    result.save_to_json("output/")
    result.save_to_markdown("output/")
    markdown_list.append(result.markdown)

# Merge all pages' Markdown
markdown_text = pipeline.concatenate_markdown_pages(markdown_list)
with open("output/document.md", "w", encoding="utf-8") as f:
    f.write(markdown_text)
```

### 📁 Project Structure

```
PaddleOCR-Toolkit/
├── 📖 docs/                           # 📚 Core documentation
│   ├── Learn_PaddleOCR_From_Scratch.md  # 🎯 PP-StructureV3 module guide
│   ├── note.md                          # ⚙️ Complete installation guide
│   └── imgs/                            # 📷 Documentation images
├── 🧪 tests/                          # Test cases and examples
│   ├── OCR/                            # OCR tests
│   ├── Layout_Detection/               # Layout detection tests
│   ├── Chart/                          # Table processing tests
│   ├── Formula/                        # Formula recognition tests
│   ├── Seal/                           # Seal detection tests
│   └── PreProcess/                     # Preprocessing tests
├── 🔧 main/                           # Main functional modules
│   ├── Complete/                       # Complete document processing
│   └── monitor/                        # Performance monitoring
├── 🛠️ utils/                          # Utility functions
│   ├── pdf2image.py                    # PDF to image converter
│   └── performance_monitor.py          # Performance monitoring
├── 📊 data/                           # Data directory
├── 🏗️ PaddleOCR/                      # PaddleOCR source code
└── 📋 requirements.txt                 # Python dependencies
```

### 🎯 Core Functional Modules

1. **Layout Detection** - Identify different elements in documents (text, tables, images, formulas, etc.)
2. **General OCR Pipeline** - Text detection and recognition with multi-language support
3. **Table Recognition v2** - Table structure recognition and content extraction with HTML output
4. **Formula Recognition** - Mathematical formula detection and recognition with LaTeX output
5. **Seal Text Recognition** - Seal area detection and text recognition
6. **Document Image Preprocessing** - Document orientation classification and distortion correction

### ⚡ Performance Optimization

#### TensorRT Acceleration (Recommended)

```bash
# Download TensorRT 8.6.1
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz

# Extract and install
tar xvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
cd TensorRT-8.6.1.6/python
uv pip install tensorrt-8.6.1-cp310-none-linux_x86_64.whl

# Set environment variables
export TENSORRT_ROOT=/path/to/TensorRT-8.6.1.6
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH
```

#### High Performance Inference (HPI)

```bash
# Install HPI dependencies
paddlex --install hpi-gpu

# Enable in code
pipeline = PPStructureV3(
    enable_hpi=True,
    use_tensorrt=True,
    precision="fp16"
)
```

### 🧪 Running Tests

The project provides comprehensive test cases covering all functional modules:

```bash
# OCR text recognition test
python tests/OCR/test.py

# Table recognition test
python tests/Chart/test.py

# Layout detection test
python tests/Layout_Detection/test.py

# Formula recognition test
python tests/Formula/test.py

# Seal detection test
python tests/Seal/test.py

# Image preprocessing test
python tests/PreProcess/test.py
```

### 🌐 Service Deployment

#### MCP Server Deployment

```bash
# Start OCR service
paddleocr_mcp --pipeline OCR --ppocr_source local --port 8090 --http

# Start complete document processing service
paddleocr_mcp --pipeline PP-StructureV3 --ppocr_source local --host 0.0.0.0 --port 8090 --http --pipeline_config PaddleOCR.yaml --device gpu
```

#### Configuration Example

```json
{
  "mcpServers": {
    "paddleocr-ocr": {
      "command": "paddleocr_mcp",
      "args": [],
      "env": {
        "PADDLEOCR_MCP_PIPELINE": "PP-StructureV3",
        "PADDLEOCR_MCP_PPOCR_SOURCE": "local"
      }
    }
  }
}
```

### 📊 Performance Monitoring

Built-in Prometheus monitoring:

```python
from utils.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_monitoring()

# Execute processing tasks
results = pipeline.predict(input_path)

# Get performance report
report = monitor.get_report()
print(report)
```

### 🔧 Configuration Optimization

#### GPU Memory Optimization
- Use batch processing: `batch_size=6`
- Enable mixed precision: `precision="fp16"`
- Set reasonable image size: `img_size=640`

#### CPU Optimization
- Enable MKL-DNN: `enable_mkldnn=True`
- Adjust thread count: `cpu_threads=8`
- Set cache: `mkldnn_cache_capacity=10`

### 🐛 Common Issues

#### 1. TensorRT Related
- **Issue**: `libnvjpeg.so.11: cannot open shared object file`
- **Solution**: Install complete CUDA 11.8 toolkit

#### 2. High Performance Inference
- **Issue**: `fused_rms_norm_ext` import error
- **Solution**: Comment out related operators or use standard inference mode

#### 3. Memory Insufficient
- **Solution**: Reduce `batch_size`, use `precision="fp16"`

### 📚 Learning Resources

#### 🎯 Essential Documentation
1. **[PP-StructureV3 Module Guide](docs/Learn_PaddleOCR_From_Scratch.md)**
   - Functional introduction of 7 core modules
   - Detailed parameter configuration instructions
   - Complete code examples

2. **[Complete Installation & Deployment Guide](docs/note.md)**
   - Environment setup steps
   - TensorRT acceleration configuration
   - Cloud deployment solutions
   - Performance monitoring setup

#### 📖 Official Documentation
- [PaddleOCR Official Documentation](https://www.paddleocr.ai/)
- [PaddleX Documentation](https://paddlepaddle.github.io/PaddleX/)
- [PaddlePaddle Documentation](https://www.paddlepaddle.org.cn/)

### 🤝 Contributing

Welcome to contribute code! Please follow these steps:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

### 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

- [PaddlePaddle](https://www.paddlepaddle.org.cn/) - Deep learning framework
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR toolkit
- [PaddleX](https://github.com/PaddlePaddle/PaddleX) - Low-code development tool

### 📞 Contact

- Project Homepage: [GitHub Repository](https://github.com/your-username/PaddleOCR-Toolkit)
- Issue Reports: [Issues](https://github.com/your-username/PaddleOCR-Toolkit/issues)
- Email: your-email@example.com

---

⭐ If this project helps you, please give us a Star!
