from paddleocr import LayoutDetection
import os
import paddle
from paddle.inference import Config as PredictConfig

# TensorRT 加速推理参数 
def create_optimized_predictor(model_path, params_path):
    config = PredictConfig(model_path, params_path)
    config.enable_use_gpu(100, 0)
    config.enable_memory_optim()
    config.switch_ir_optim(True)
    
    # TensorRT详细配置
    config.enable_tensorrt_engine(
        workspace_size=1 << 30,        # 1GB工作空间
        max_batch_size=1,              # 最大批处理大小
        min_subgraph_size=4,           # 最小子图大小
        precision_mode=paddle.inference.PrecisionType.Half,  # FP16精度
        use_static=True,               # 使用静态引擎
        use_calib_mode=False           # 不使用校准模式
    )
    
    # 针对特定硬件的优化
    config.exp_disable_tensorrt_ops(["elementwise_add"])
    
    return config

model = LayoutDetection(
    model_name="PP-DocLayout_plus-L",
    # model_dir="/root/projects/PaddleOCR/pretrained_models/layout/PP-DocLayout_plus-L",
    device="gpu:0",
    enable_hpi=True,
    use_tensorrt=True,
    precision="fp16",
    enable_mkldnn=False,
    # mkldnn_cache_capacity=10,
    cpu_threads=8,
    threshold=0.5,  # 置信度阈值
    layout_nms=True,  # 启用NMS过滤重叠框
    layout_unclip_ratio=1.0,  # 检测框缩放倍数
    layout_merge_bboxes_mode="large",  # 框合并模式
)

print("开始版面分析...")
input_path = "/root/projects/PaddleOCR/tests/Layout_Detection/files/cover.png"
output = model.predict(input_path, batch_size=4)
save_path = "/root/projects/PaddleOCR/tests/Layout_Detection/outputs/cover_layout/"
os.makedirs(save_path, exist_ok=True)

for res in output:
    res.print()

    res.save_to_img(save_path=save_path)
    res.save_to_json(save_path=save_path)

    json_result = res.json
    
    print("\n" + "="*80)
    print("版面分析结果详情：")
    print(f"输入路径: {json_result['res'].get('input_path', 'N/A')}")
    print(f"页面索引: {json_result['res'].get('page_index', 'N/A')}")
    
    boxes = json_result['res'].get('boxes', [])
    print(f"\n检测到 {len(boxes)} 个版面元素：")
    print("-" * 80)
    
    # 统计各类别数量
    from collections import Counter
    label_counts = Counter([box['label'] for box in boxes])
    
    for label, count in label_counts.items():
        print(f"  {label}: {count} 个")
    
    print("-" * 80)
    
    # 显示每个检测框的详细信息
    print("\n详细信息：")
    for i, box in enumerate(boxes, 1):
        print(f"{i:2d}. [{box['label']:15s}] 置信度: {box['score']:.4f}, "
              f"坐标: [{box['coordinate'][0]:.1f}, {box['coordinate'][1]:.1f}, "
              f"{box['coordinate'][2]:.1f}, {box['coordinate'][3]:.1f}]")
    
    print("="*80)

print("\n结果已保存到 ./output_layout/ 目录")