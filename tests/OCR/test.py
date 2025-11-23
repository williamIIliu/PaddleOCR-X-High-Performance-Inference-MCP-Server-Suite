from paddleocr import PaddleOCR
import json

# 导出yaml文件进行测试
pipeline = PaddleOCR()
pipeline.export_paddlex_config_to_yaml("/root/projects/PaddleOCR/tests/OCR/config/PaddleOCR.yaml")
# 进行调用
pipeline = PaddleOCR(paddlex_config="/root/projects/PaddleOCR/tests/OCR/config/PaddleOCR.yaml")

# 正式流程
ocr = PaddleOCR(
    # ========== 模型路径配置 ==========
    # 文档方向分类
    doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
    # doc_orientation_classify_model_dir="/root/projects/PaddleOCR/pretrained_models/doc_ori/PP-LCNet_x1_0_doc_ori_infer",
    
    # 文本图像矫正
    doc_unwarping_model_name="UVDoc",
    doc_unwarping_model_dir="/root/projects/PaddleOCR/pretrained_models/doc_unwarping/UVDoc_infer",
    
    # 文本检测
    text_detection_model_name="PP-OCRv5_server_det",
    # text_detection_model_dir="/root/projects/PaddleOCR/pretrained_models/text_detection/PP-OCRv5_server_det_infer",
    
    # 文本行方向分类
    textline_orientation_model_name="PP-LCNet_x0_25_textline_ori",
    # textline_orientation_model_dir="/root/projects/PaddleOCR/pretrained_models/textline_orientation/PP-LCNet_x0_25_textline_ori_infer",
    textline_orientation_batch_size=1,
    
    # 文本识别
    text_recognition_model_name="PP-OCRv5_server_rec",
    # text_recognition_model_dir="/root/projects/PaddleOCR/pretrained_models/text_recognition/PP-OCRv5_server_rec_infer",
    text_recognition_batch_size=1,
    
    # ========== 模块开关 ==========
    use_doc_orientation_classify=False,   # 是否使用文档方向分类
    use_doc_unwarping=False,             # 是否使用文本图像矫正
    use_textline_orientation=False,      # 是否使用文本行方向分类
    
    # ========== 文本检测参数 ==========
    text_det_limit_side_len=960,         # 图像边长限制
    text_det_limit_type="max",           # 边长限制类型: "min" 或 "max"
    text_det_thresh=0.3,                 # 文本检测像素阈值
    text_det_box_thresh=0.6,             # 文本检测框阈值
    text_det_unclip_ratio=1.5,           # 文本检测扩张系数
    
    # ========== 文本识别参数 ==========
    text_rec_score_thresh=0.5,           # 文本识别阈值
    
    # ========== 设备配置 ==========
    device="gpu:0",                      # 推理设备: "cpu", "gpu:0", "npu:0" 等
    enable_hpi=False,                    # 是否启用高性能推理
    use_tensorrt=False,                  # 是否使用TensorRT
    precision="fp16",                    # 计算精度: "fp32" 或 "fp16"
    enable_mkldnn=True,                  # 是否使用MKL-DNN加速(CPU)
    mkldnn_cache_capacity=10,            # MKL-DNN缓存容量
    cpu_threads=10,                       # CPU线程数
)

# 预测路径参数
input_path = "/root/projects/PaddleOCR/tests/OCR/files"
outputs_path = "/root/projects/PaddleOCR/tests/OCR/outputs"
# 预测单张图片
result = ocr.predict(input_path + "/test_one_column_page_015.png")

# 或者预测多张图片
# result = ocr.predict([
#     "test_one_column_page_015.png",
#     "test_one_column_page_027.png"
# ])

# 或者预测整个目录
# result = ocr.predict(input_path)

for res in result:
    # 打印结果
    res.print()
    
    # 保存结果
    res.save_to_img(outputs_path)
    res.save_to_json(outputs_path)
    
    # 获取 JSON 结果
    json_result = res.json
    
    print("\n" + "="*80)
    print("JSON 结果的所有键:")
    print(json_result.keys())
    
    # 访问 res 键
    ocr_res = json_result['res']
    print("\nocr_res 的类型:", type(ocr_res))
    
    if isinstance(ocr_res, dict):
        print("ocr_res 的键:", ocr_res.keys())
        print("\n完整的 OCR 结果:")
        print(json.dumps(ocr_res, indent=4, ensure_ascii=False, default=str))
    elif isinstance(ocr_res, list):
        print(f"\nocr_res 是列表，长度: {len(ocr_res)}")
        print("\n前几个结果:")
        for i, item in enumerate(ocr_res[:5]):  # 只打印前5个
            print(f"{i+1}. {item}")
    else:
        print("\nocr_res 内容:")
        print(ocr_res)
    
    print("="*80)