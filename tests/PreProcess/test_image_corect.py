from paddleocr import TextImageUnwarping

model = TextImageUnwarping(
    model_name="UVDoc",
    # model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/doc_unwarping/UVDoc_infer",
    device="gpu:0",
    enable_hpi=False,
    use_tensorrt=False,
    precision="fp32",
    enable_mkldnn=True,
    mkldnn_cache_capacity=10,
    cpu_threads=8,
)
image_path = "/root/projects/PaddleOCR/tests/PreProcess/files/image_correct.jpg"
output = model.predict(image_path, batch_size=1)

# 处理结果
for res in output:
    # 打印结果
    res.print()
    
    # 保存矫正后的图像
    res.save_to_img(save_path="/root/projects/PaddleOCR/tests/PreProcess/outputs/image_correct.png")
    
    # 保存JSON结果
    res.save_to_json(save_path="/root/projects/PaddleOCR/tests/PreProcess/outputs//image_correct.json")
    
    # 获取结果
    json_result = res.json
    print("\n" + "="*80)
    print("矫正结果：")
    print(f"输入路径: {json_result['res']['input_path']}")
    print(f"页面索引: {json_result['res']['page_index']}")
    print("矫正图像已保存")
    print("="*80)