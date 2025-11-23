from paddleocr import TableRecognitionPipelineV2
import os
import json

os.makedirs("./output_table_v2_full", exist_ok=True)

print("初始化通用表格识别v2产线...")
print("="*80)

# 指定所有模型路径
pipeline = TableRecognitionPipelineV2(
    # 版面检测模型
    layout_detection_model_name="PP-DocLayout_plus-L",
    # layout_detection_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/layout/PP-DocLayout_plus-L_infer",
    
    # 表格分类模型
    table_classification_model_name="PP-LCNet_x1_0_table_cls",
    # table_classification_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/table_cls/PP-LCNet_x1_0_table_cls_infer",
    
    # 有线表格结构识别模型
    wired_table_structure_recognition_model_name="SLANet_plus",
    # wired_table_structure_recognition_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/table_structure/SLANet_plus_infer",
    
    # 无线表格结构识别模型
    # wireless_table_structure_recognition_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/table_structure_recognition/SLANet_plus_wireless_infer",
    wireless_table_structure_recognition_model_name="SLANet_plus",
    # wireless_table_structure_recognition_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/table_structure/SLANet_plus_infer",
    
    # 有线表格单元格检测模型
    wired_table_cells_detection_model_name="RT-DETR-L_wired_table_cell_det",
    # wired_table_cells_detection_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/table_cells_detection/RT-DETR-L_wired_table_cell_det_infer",
    
    # 无线表格单元格检测模型
    # wireless_table_cells_detection_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/table_cells_detection/RT-DETR-L_wireless_table_cell_det_infer",
    wireless_table_cells_detection_model_name="RT-DETR-L_wireless_table_cell_det",
    # wireless_table_cells_detection_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/table_cells_detection/RT-DETR-L_wireless_table_cell_det_infer",
    
    # 文本检测模型
    text_detection_model_name="PP-OCRv5_server_det",
    # text_detection_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/text_detection/PP-OCRv5_server_det_infer",
    
    # 文本识别模型
    text_recognition_model_name="PP-OCRv5_server_rec",
    # text_recognition_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/text_recognition/PP-OCRv5_server_rec_infer",
    
    # 文档方向分类模型
    doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
    # doc_orientation_classify_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/doc_ori/PP-LCNet_x1_0_doc_ori_infer",
    
    # 文本图像矫正模型
    doc_unwarping_model_name="UVDoc",
    # doc_unwarping_model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/doc_unwarping/UVDoc_infer",
    
    # 功能开关
    use_doc_orientation_classify=True,  # 启用文档方向分类
    use_doc_unwarping=True,            # 启用文本图像矫正
    use_layout_detection=True,         # 启用版面检测
    use_ocr_model=True,                # 启用OCR
    
    # 设备配置
    device="gpu:0",
    enable_hpi=False,
    use_tensorrt=False,
    precision="fp16",
    
    # 文本检测参数
    text_det_limit_side_len=960,
    text_det_limit_type="max",
    text_det_thresh=0.3,
    text_det_box_thresh=0.6,
    text_det_unclip_ratio=2.0,
    
    # 文本识别参数
    text_recognition_batch_size=6,
    text_rec_score_thresh=0.5,
    
    # CPU参数
    enable_mkldnn=True,
    mkldnn_cache_capacity=10,
    cpu_threads=8,
)

print("✓ 产线初始化完成\n")

input_image = "/root/projects/PaddleOCR/tests/Table/files/Table1.png"
output_path = "/root/projects/PaddleOCR/tests/Table/outputs/"
os.makedirs(output_path, exist_ok=True)
print(f"处理图像: {input_image}")
print("="*80)

try:
    output = pipeline.predict(
        input_image,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_layout_detection=True,
        use_ocr_model=True,
        use_table_orientation_classify=True,
        use_ocr_results_with_table_cells=True,
    )
    
    for idx, res in enumerate(output, 1):
        print(f"\n处理结果 {idx}:")
        print("="*80)

        res.print()

        print("\n保存结果...")
        res.save_to_img( output_path + "res/")
        res.save_to_xlsx(output_path + "xlsx/")
        res.save_to_html(output_path + "html/")
        res.save_to_json(output_path + "./Table1.json")

        # 获取JSON结果用于详细统计
        result = res.json
        
        print("\n" + "="*80)
        print("详细统计信息:")
        print("="*80)

        # 模型配置
        model_settings = result.get('model_settings', {})
        print("\n模型配置:")
        for key, value in model_settings.items():
            print(f"  {key}: {value}")

        # 文档预处理结果
        if 'doc_preprocessor_res' in result:
            doc_res = result['doc_preprocessor_res']
            print("\n文档预处理:")
            print(f"  文档旋转角度: {doc_res.get('angle', 'N/A')}°")

        # 版面检测结果
        if 'layout_det_res' in result:
            layout_res = result['layout_det_res']
            boxes = layout_res.get('boxes', [])
            print(f"\n版面检测:")
            print(f"  检测到的区域数量: {len(boxes)}")
            
            for i, box in enumerate(boxes, 1):
                print(f"  区域 {i}:")
                print(f"    类别ID: {box.get('cls_id')}")
                print(f"    置信度: {box.get('score', 0):.4f}")
                coord = box.get('coordinate', [])
                print(f"    坐标: [{coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}, {coord[3]:.1f}]")

        # OCR结果
        if 'ocr_res' in result:
            ocr_res = result['ocr_res']
            
            dt_polys = ocr_res.get('dt_polys', [])
            rec_texts = ocr_res.get('rec_texts', [])
            rec_scores = ocr_res.get('rec_scores', [])
            
            print(f"\nOCR结果:")
            print(f"  检测到的文本框数量: {len(dt_polys)}")
            print(f"  识别出的文本数量: {len(rec_texts)}")
            
            if rec_texts:
                print("\n  识别文本详情:")
                print("  " + "-"*76)
                print(f"  {'序号':<6}{'文本':<40}{'置信度':<15}{'坐标'}")
                print("  " + "-"*76)
                
                for i, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, ocr_res.get('rec_polys', [])), 1):
                    display_text = text[:35] + "..." if len(text) > 35 else text
                    coord_str = f"[{poly[0][0]:.0f},{poly[0][1]:.0f}]"
                    print(f"  {i:<6}{display_text:<40}{score:<15.4f}{coord_str}")
                
                print("  " + "-"*76)

                avg_score = sum(rec_scores) / len(rec_scores)
                print(f"\n  平均识别置信度: {avg_score:.4f}")

        # 表格识别结果
        if 'table_res' in result:
            tables = result['table_res']
            print(f"\n表格识别:")
            print(f"  检测到的表格数量: {len(tables)}")
            
            for i, table in enumerate(tables, 1):
                print(f"\n  表格 {i}:")
                print(f"    表格类型: {table.get('table_type', 'N/A')}")
                
                bbox = table.get('bbox', [])
                if bbox:
                    print(f"    表格位置: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

                html = table.get('html', '')
                if html:
                    row_count = html.count('<tr')
                    col_count = html.count('<td')
                    print(f"    表格行数: {row_count}")
                    print(f"    表格单元格数: {col_count}")
        
        print("\n" + "="*80)
        print("文件保存位置:")
        print("="*80)
        print("  可视化图像: /root/projects/PaddleOCR/tests/Table/outputs/res/")
        print("  JSON结果:   /root/projects/PaddleOCR/tests/Table/outputs/res.json")
        print("  Excel文件:  /root/projects/PaddleOCR/tests/Table/outputs/xlsx/*.xlsx")
        print("  HTML文件:   /root/projects/PaddleOCR/tests/Table/outputs/html/*.html")
        print("="*80)

        img_dict = res.img
        print("\n可用的可视化图像:")
        for key in img_dict.keys():
            print(f"  - {key}")

        print("\n" + "="*80)
        print("完整JSON结果（格式化）:")
        print("="*80)
        print(json.dumps(result, indent=2, ensure_ascii=False))

except Exception as e:
    print(f"\n处理过程中出错: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ 处理完成！")