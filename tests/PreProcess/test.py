from paddleocr import DocPreprocessor
import os

os.makedirs("./output_docpp", exist_ok=True)

pipeline = DocPreprocessor(
    doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
    # doc_orientation_classify_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/doc_ori/PP-LCNet_x1_0_doc_ori_infer",
    doc_unwarping_model_name="UVDoc",
    # doc_unwarping_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/doc_unwarping/UVDoc_infer",
    use_doc_orientation_classify=True,  # 启用文档方向分类
    use_doc_unwarping=True,             # 启用文本图像矫正
    device="gpu:0",
    enable_hpi=False,
    use_tensorrt=False,
    precision="fp16",
    enable_mkldnn=True,
    mkldnn_cache_capacity=10,
    cpu_threads=8,
)

print("开始处理图像...")
image_path = "/root/projects/PaddleOCR/tests/PreProcess/files/image.jpg"
output = pipeline.predict(image_path)
save_path="/root/projects/PaddleOCR/tests/PreProcess/outputs/"

for res in output:
    res.print()

    res.save_to_img("/root/projects/PaddleOCR/tests/PreProcess/outputs/image.png")
    res.save_to_json("/root/projects/PaddleOCR/tests/PreProcess/outputs/image.json")

    json_result = res.json
    
    print("\n" + "="*80)
    print("文档预处理结果详情：")
    print(f"输入路径: {json_result.get('input_path', 'N/A')}")
    print(f"页面索引: {json_result.get('page_index', 'N/A')}")
    print(f"文档方向角度: {json_result.get('angle', 'N/A')}°")
    
    if 'model_settings' in json_result:
        print(f"模型设置:")
        print(f"  - 使用文档方向分类: {json_result['model_settings'].get('use_doc_orientation_classify', False)}")
        print(f"  - 使用文本图像矫正: {json_result['model_settings'].get('use_doc_unwarping', False)}")
    
    print("="*80)

    img_dict = res.img
    print(f"\n可视化图像键: {list(img_dict.keys())}")
    if 'preprocessed_img' in img_dict:
        print("预处理后的图像已生成")
        
    import json
    print("\n完整JSON结果:")
    print(json.dumps(json_result, indent=4, ensure_ascii=False, default=str))

print(f"\n结果已保存到 {save_path} 目录")