from paddleocr import FormulaRecognitionPipeline
import os
import json

print("初始化公式识别产线...")
print("="*80)

# 实例化公式识别产线（完整参数配置）
pipeline = FormulaRecognitionPipeline(
    # 文档方向分类模型配置
    doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
    # doc_orientation_classify_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/doc_ori/PP-LCNet_x1_0_doc_ori_infer",
    doc_orientation_classify_batch_size=1,
    
    # 文本图像矫正模型配置
    doc_unwarping_model_name="UVDoc",
    # doc_unwarping_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/doc_unwarping/UVDoc_infer",
    doc_unwarping_batch_size=1,
    
    # 功能开关
    use_doc_orientation_classify=True,   # 是否使用文档方向分类
    use_doc_unwarping=False,             # 是否使用文本图像矫正
    
    # 版面区域检测模型配置
    layout_detection_model_name="PP-DocLayout_plus-L",
    # layout_detection_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/layout/PP-DocLayout_plus-L_infer",
    layout_detection_batch_size=1,
    
    # 版面检测参数
    layout_threshold=0.5,                # 置信度阈值
    layout_nms=True,                     # 是否使用NMS
    layout_unclip_ratio=1.0,             # 检测框扩张系数
    layout_merge_bboxes_mode="large",    # 重叠框过滤方式：large/small/union
    
    use_layout_detection=True,           # 是否使用版面区域检测
    
    # 公式识别模型配置
    formula_recognition_model_name="PP-FormulaNet_plus-M",
    # formula_recognition_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/formula_recognition/PP-FormulaNet_plus-M_infer",
    formula_recognition_batch_size=1,
    
    # 设备配置
    device="gpu:0",                      # cpu/gpu:0/npu:0/xpu:0/mlu:0/dcu:0
    
    # 性能优化参数
    enable_hpi=False,                    # 是否启用高性能推理
    use_tensorrt=False,                  # 是否启用TensorRT加速
    precision="fp16",                    # 计算精度：fp32/fp16
    
    # CPU优化参数
    enable_mkldnn=True,                  # 是否启用MKL-DNN加速
    mkldnn_cache_capacity=10,            # MKL-DNN缓存容量
    cpu_threads=8,                       # CPU线程数
    
    # paddlex_config="./config.yaml",    # PaddleX产线配置文件路径
)

print("✓ 产线初始化完成\n")

# 输入图像路径
input_image = "/root/projects/PaddleOCR/tests/Formula/files/test.jpg"
print(f"处理图像: {input_image}")
print("="*80)
output_path = "/root/projects/PaddleOCR/tests/Formula/outputs/"
os.makedirs(output_path, exist_ok=True)

try:
    # 执行预测（可以在预测时覆盖初始化参数）
    output = pipeline.predict(
        input=input_image,
        use_layout_detection=True,
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        layout_threshold=0.5,
        layout_nms=True,
        layout_unclip_ratio=1.0,
        layout_merge_bboxes_mode="large",
    )
    
    # 处理预测结果
    for idx, res in enumerate(output, 1):
        print(f"\n处理结果 {idx}:")
        print("="*80)
        
        # 1. 打印结果到终端
        res.print()
        
        # 2. 保存可视化图像
        res.save_to_img(save_path=output_path)
        
        # 3. 保存JSON结果
        res.save_to_json(save_path=output_path)
        
        # 4. 获取预测结果的JSON数据
        result_json = res.json
        
        print("\n" + "="*80)
        print("格式化的完整JSON结果:")
        print("="*80)
        print(json.dumps(result_json, indent=4, ensure_ascii=False))
        
        print("\n" + "="*80)
        print("详细分析:")
        print("="*80)
        
        # 基本信息
        print(f"\n输入路径: {result_json.get('input_path', 'N/A')}")
        print(f"页面索引: {result_json.get('page_index', 'N/A')}")
        
        # 模型配置
        model_settings = result_json.get('model_settings', {})
        print("\n模型配置:")
        print(f"  使用文档预处理: {model_settings.get('use_doc_preprocessor', False)}")
        print(f"  使用版面检测: {model_settings.get('use_layout_detection', False)}")
        
        # 文档预处理结果
        if 'doc_preprocessor_res' in result_json:
            doc_res = result_json['doc_preprocessor_res']
            print("\n文档预处理结果:")
            
            doc_settings = doc_res.get('model_settings', {})
            print(f"  使用方向分类: {doc_settings.get('use_doc_orientation_classify', False)}")
            print(f"  使用图像矫正: {doc_settings.get('use_doc_unwarping', False)}")
            
            angle = doc_res.get('angle', -1)
            angle_map = {0: "0°", 1: "90°", 2: "180°", 3: "270°", -1: "未启用"}
            print(f"  文档旋转角度: {angle_map.get(angle, str(angle))}")
        
        # 版面检测结果
        if 'layout_det_res' in result_json:
            layout_res = result_json['layout_det_res']
            boxes = layout_res.get('boxes', [])
            print(f"\n版面检测:")
            print(f"  检测到的区域数量: {len(boxes)}")
            
            for i, box in enumerate(boxes, 1):
                cls_id = box.get('cls_id')
                label = box.get('label', 'Unknown')
                score = box.get('score', 0)
                coord = box.get('coordinate', [])
                
                print(f"\n  区域 {i}:")
                print(f"    类别ID: {cls_id}")
                print(f"    类别名称: {label}")
                print(f"    置信度: {score:.4f}")
                if coord:
                    print(f"    坐标: [{coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}, {coord[3]:.1f}]")
        
        # 公式识别结果
        if 'formula_res_list' in result_json:
            formula_list = result_json['formula_res_list']
            print(f"\n公式识别:")
            print(f"  识别到的公式数量: {len(formula_list)}")
            
            for i, formula in enumerate(formula_list, 1):
                rec_formula = formula.get('rec_formula', '')
                formula_id = formula.get('formula_region_id', -1)
                dt_polys = formula.get('dt_polys', [])
                
                print(f"\n  公式 {i}:")
                print(f"    区域ID: {formula_id}")
                
                if dt_polys:
                    print(f"    坐标: [{dt_polys[0]:.1f}, {dt_polys[1]:.1f}, {dt_polys[2]:.1f}, {dt_polys[3]:.1f}]")
                
                print(f"    LaTeX公式:")
                print(f"    {'-'*76}")
                # 显示公式，如果太长则截断
                if len(rec_formula) > 200:
                    print(f"    {rec_formula[:200]}...")
                    print(f"    ... (总长度: {len(rec_formula)} 字符)")
                else:
                    print(f"    {rec_formula}")
                print(f"    {'-'*76}")
                
                # 公式复杂度分析
                complexity = {
                    '分数': rec_formula.count('\\frac'),
                    '积分': rec_formula.count('\\int'),
                    '求和': rec_formula.count('\\sum'),
                    '上下标': rec_formula.count('^') + rec_formula.count('_'),
                    '括号对': rec_formula.count('{'),
                }
                
                print(f"\n    复杂度分析:")
                for key, count in complexity.items():
                    if count > 0:
                        print(f"      {key}: {count}")
        
        print("\n" + "="*80)
        print("文件保存位置:")
        print("="*80)
        print("  可视化图像目录: /root/projects/PaddleOCR/tests/Formula/outputs/")
        print("  JSON结果文件:   /root/projects/PaddleOCR/tests/Formula/outputs/")
        print("="*80)
        
        # 5. 获取可视化图像
        img_dict = res.img
        print("\n可用的可视化图像:")
        for key in img_dict.keys():
            print(f"  - {key}")
        
        # 可以单独保存某个可视化图像
        if 'preprocessed_img' in img_dict:
            print("  → 包含预处理图像")
        if 'layout_det_res' in img_dict:
            print("  → 包含版面检测图像")
        if 'formula_res_img' in img_dict:
            print("  → 包含公式识别图像")

except Exception as e:
    print(f"\n处理过程中出错: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ 处理完成！")