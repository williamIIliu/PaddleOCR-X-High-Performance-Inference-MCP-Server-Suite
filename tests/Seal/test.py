from paddleocr import SealRecognition
import os
import json

os.makedirs("./output_seal", exist_ok=True)

print("初始化印章文本识别产线...")
print("="*80)

pipeline = SealRecognition(

    doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
    # doc_orientation_classify_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/doc_ori/PP-LCNet_x1_0_doc_ori_infer",
    doc_unwarping_model_name="UVDoc",
    # doc_unwarping_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/doc_unwarping/UVDoc_infer",
    layout_detection_model_name="PP-DocLayout_plus-L",
    # layout_detection_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/layout/PP-DocLayout_plus-L_infer",
    seal_text_detection_model_name="PP-OCRv4_server_seal_det",
    # seal_text_detection_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/seal_det/PP-OCRv4_server_seal_det_infer",
    text_recognition_model_name="PP-OCRv5_server_rec",
    # text_recognition_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/text_recognition/PP-OCRv5_server_rec_infer",
    
    # 功能开关
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    use_layout_detection=True,
    
    # 版面检测参数
    layout_threshold=0.5,
    layout_nms=True,
    layout_unclip_ratio=1.0,
    layout_merge_bboxes_mode="large",
    
    # 印章文本检测参数
    seal_det_limit_side_len=736,
    seal_det_limit_type="min",
    seal_det_thresh=0.2,
    seal_det_box_thresh=0.6,
    seal_det_unclip_ratio=0.5,
    
    # 文本识别参数
    text_recognition_batch_size=6,
    seal_rec_score_thresh=0.0,
    
    # 设备配置
    device="gpu:0",
    enable_hpi=False,
    use_tensorrt=False,
    precision="fp32",
    
    # CPU参数
    enable_mkldnn=True,
    mkldnn_cache_capacity=10,
    cpu_threads=8,
)

print("✓ 产线初始化完成\n")

# 输入图像路径
input_image = "/root/projects/PaddleOCR/tests/Seal/files/test.jpg"
print(f"处理图像: {input_image}")
print("="*80)
output_path = "/root/projects/PaddleOCR/tests/Seal/outputs/"
os.makedirs(output_path, exist_ok=True)

try:
    output = pipeline.predict(
        input_image,
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_layout_detection=True,
        seal_det_limit_side_len=736,
        seal_det_thresh=0.2,
        seal_det_box_thresh=0.6,
    )

    for idx, res in enumerate(output, 1):
        print(f"\n处理结果 {idx}:")
        print("="*80)
        
        # 1. 简单打印（不使用不支持的参数）
        res.print()
        
        # 2. 保存可视化图像
        res.save_to_img(output_path)
        
        # 3. 保存JSON结果
        res.save_to_json(output_path)
        
        # 4. 获取预测结果的JSON数据并手动格式化输出
        result_json = res.json
        
        print("\n" + "="*80)
        print("格式化的完整JSON结果:")
        print("="*80)
        print(json.dumps(result_json, indent=4, ensure_ascii=False))
        
        print("\n" + "="*80)
        print("详细统计信息:")
        print("="*80)
        
        # 输入信息
        print(f"\n输入路径: {result_json.get('input_path')}")
        print(f"页面索引: {result_json.get('page_index', 'N/A')}")
        
        # 模型配置
        model_settings = result_json.get('model_settings', {})
        print("\n模型配置:")
        print(f"  使用文档预处理: {model_settings.get('use_doc_preprocessor', False)}")
        print(f"  使用版面检测: {model_settings.get('use_layout_detection', False)}")
        
        # 版面检测结果
        if 'layout_det_res' in result_json:
            layout_res = result_json['layout_det_res']
            boxes = layout_res.get('boxes', [])
            print(f"\n版面检测:")
            print(f"  检测到的印章区域数量: {len(boxes)}")
            
            for i, box in enumerate(boxes, 1):
                print(f"  区域 {i}:")
                print(f"    类别ID: {box.get('cls_id')}")
                print(f"    置信度: {box.get('score', 0):.4f}")
                coord = box.get('coordinate', [])
                print(f"    坐标: [{coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}, {coord[3]:.1f}]")
        
        # 印章识别结果
        if 'seal_res_list' in result_json:
            seal_res_list = result_json['seal_res_list']
            print(f"\n印章识别:")
            print(f"  识别到的印章数量: {len(seal_res_list)}")
            
            for i, seal_res in enumerate(seal_res_list, 1):
                print(f"\n  印章 {i}:")
                
                # 文档预处理结果
                if 'doc_preprocessor_res' in seal_res:
                    doc_res = seal_res['doc_preprocessor_res']
                    angle = doc_res.get('angle', -1)
                    angle_map = {0: "0°", 1: "90°", 2: "180°", 3: "270°", -1: "未启用"}
                    print(f"    文档旋转角度: {angle_map.get(angle, str(angle))}")
                
                # 文本检测参数
                text_det_params = seal_res.get('text_det_params', {})
                if text_det_params:
                    print(f"    文本检测参数:")
                    print(f"      边长限制: {text_det_params.get('limit_side_len', 'N/A')}")
                    print(f"      限制类型: {text_det_params.get('limit_type', 'N/A')}")
                    print(f"      像素阈值: {text_det_params.get('thresh', 'N/A')}")
                    print(f"      检测框阈值: {text_det_params.get('box_thresh', 'N/A')}")
                    print(f"      扩张系数: {text_det_params.get('unclip_ratio', 'N/A')}")
                    print(f"      文本类型: {text_det_params.get('text_type', 'N/A')}")
                
                # 文本检测结果
                dt_polys = seal_res.get('dt_polys', [])
                dt_scores = seal_res.get('dt_scores', [])
                print(f"    检测到的文本框数量: {len(dt_polys)}")
                
                if dt_scores:
                    avg_det_score = sum(dt_scores) / len(dt_scores)
                    print(f"    平均检测置信度: {avg_det_score:.4f}")
                
                # 文本识别结果
                rec_texts = seal_res.get('rec_texts', [])
                rec_scores = seal_res.get('rec_scores', [])
                rec_polys = seal_res.get('rec_polys', [])
                
                rec_score_thresh = seal_res.get('text_rec_score_thresh', 0.0)
                print(f"    文本识别阈值: {rec_score_thresh}")
                
                if rec_texts:
                    print(f"    识别出的文本数量: {len(rec_texts)}")
                    print("\n    识别文本详情:")
                    print("    " + "-"*76)
                    print(f"    {'序号':<6}{'文本':<40}{'置信度':<15}{'坐标'}")
                    print("    " + "-"*76)
                    
                    for j, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, rec_polys), 1):
                        display_text = text[:35] + "..." if len(text) > 35 else text
                        coord_str = f"[{poly[0][0]:.0f},{poly[0][1]:.0f}]" if len(poly) > 0 else "N/A"
                        print(f"    {j:<6}{display_text:<40}{score:<15.4f}{coord_str}")
                    
                    print("    " + "-"*76)
                    
                    # 统计信息
                    avg_score = sum(rec_scores) / len(rec_scores)
                    print(f"\n    平均识别置信度: {avg_score:.4f}")
                    print(f"    完整印章文本: {''.join(rec_texts)}")
                    print(f"    文本总长度: {sum(len(text) for text in rec_texts)} 字符")
                else:
                    print("    未识别到文本（可能阈值过高）")
        
        print("\n" + "="*80)
        print("文件保存位置:")
        print("="*80)
        print("  可视化图像目录: /root/projects/PaddleOCR/tests/Seal/outputs/")
        print("  JSON结果文件:   /root/projects/PaddleOCR/tests/Seal/outputs/")
        print("="*80)
        
        # 5. 获取可视化图像
        img_dict = res.img
        print("\n可用的可视化图像:")
        for key in img_dict.keys():
            print(f"  - {key}")
            # 可以单独获取图像对象
            # img = img_dict[key]
            # img.save(f"./output_seal/{key}.jpg")

except Exception as e:
    print(f"\n处理过程中出错: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ 处理完成！")