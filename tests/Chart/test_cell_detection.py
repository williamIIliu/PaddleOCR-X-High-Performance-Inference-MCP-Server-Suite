from paddleocr import TableCellsDetection
import os
import cv2
import numpy as np

print("初始化表格单元格检测模型...")
model = TableCellsDetection(
    model_name="RT-DETR-L_wired_table_cell_det",
    # model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/table_cells_detection/RT-DETR-L_wired_table_cell_det_infer",
    device="gpu:0",
)

input_image = "/root/projects/PaddleOCR/tests/Table/files/Table1.png"
output_path = "/root/projects/PaddleOCR/tests/Table/outputs/cells/"
os.makedirs(output_path, exist_ok=True)
print(f"\n处理图像: {input_image}\n")

output = model.predict(input_image, threshold=0.3, batch_size=1)

for res in output:
    print("="*80)
    print("表格单元格检测结果:")
    print("="*80)
    res.print()

    res.save_to_img(output_path + "cells_visualized.png")
    res.save_to_json(output_path + "cell.json")

    result = res.json['res']
    boxes = result.get('boxes', [])
    
    print(f"\n检测到的单元格数量: {len(boxes)}")
    
    if boxes:
        img = cv2.imread(input_image)
        if img is not None:
            img_annotated = img.copy()

            for i, box in enumerate(boxes, 1):
                coord = box.get('coordinate', [])
                score = box.get('score', 0)

                x1, y1, x2, y2 = int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3])

                if score > 0.95:
                    color = (0, 255, 0)      # 绿色 - 高置信度
                elif score > 0.90:
                    color = (0, 255, 255)    # 黄色 - 中等置信度
                else:
                    color = (0, 128, 255)    # 橙色 - 较低置信度

                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)

                text = f"{score:.3f}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, font_thickness
                )

                text_x = x1
                text_y = y1 - 5

                if text_y < text_height + 5:
                    text_y = y1 + text_height + 5

                padding = 2
                bg_x1 = text_x - padding
                bg_y1 = text_y - text_height - padding
                bg_x2 = text_x + text_width + padding
                bg_y2 = text_y + baseline + padding

                bg_x1 = max(0, bg_x1)
                bg_y1 = max(0, bg_y1)
                bg_x2 = min(img_annotated.shape[1], bg_x2)
                bg_y2 = min(img_annotated.shape[0], bg_y2)

                overlay = img_annotated.copy()
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                img_annotated = cv2.addWeighted(overlay, 0.6, img_annotated, 0.4, 0)

                cv2.putText(img_annotated, text, (text_x, text_y),
                           font, font_scale, color, font_thickness, cv2.LINE_AA)

                cell_num_text = str(i)
                cv2.putText(img_annotated, cell_num_text, 
                           (x1 + 5, y1 + 20),
                           font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            stats_y = 30
            stats_bg_height = 100
            overlay = img_annotated.copy()
            cv2.rectangle(overlay, (10, 10), (600, stats_bg_height), (0, 0, 0), -1)
            img_annotated = cv2.addWeighted(overlay, 0.7, img_annotated, 0.3, 0)

            scores = [box['score'] for box in boxes]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            cv2.putText(img_annotated, "Table Cell Detection Result", 
                       (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            stats_y += 30
            cv2.putText(img_annotated, f"Total Cells: {len(boxes)}", 
                       (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            stats_y += 25
            cv2.putText(img_annotated, f"Avg Confidence: {avg_score:.4f} | Range: [{min_score:.3f}, {max_score:.3f}]", 
                       (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imwrite(output_path + "./cells_fully_annotated.jpg", img_annotated)
            print(f"\n✓ 完整标注图像已保存到: /root/projects/PaddleOCR/tests/Table/outputs/cells/cells_fully_annotated.jpg")

            img_simple = img.copy()
            
            for box in boxes:
                coord = box.get('coordinate', [])
                score = box.get('score', 0)
                
                x1, y1, x2, y2 = int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3])

                color = (0, 255, 0)

                cv2.rectangle(img_simple, (x1, y1), (x2, y2), color, 2)

                text = f"{score:.3f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                font_thickness = 1

                text_x = x1 + 2
                text_y = y1 - 5
                
                if text_y < 15:
                    text_y = y1 + 15

                (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                cv2.rectangle(img_simple, 
                            (text_x - 1, text_y - th - 1), 
                            (text_x + tw + 1, text_y + 2), 
                            (0, 0, 0), -1)

                cv2.putText(img_simple, text, (text_x, text_y),
                           font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
            cv2.imwrite(output_path + "./cells_simple_annotated.jpg", img_simple)
            print(f"✓ 简洁标注图像已保存到: /root/projects/PaddleOCR/tests/Table/outputs/cells/cells_simple_annotated.jpg")

            print("\n" + "="*80)
            print("单元格详情:")
            print("="*80)
            print(f"{'序号':<6}{'置信度':<12}{'坐标':<45}{'宽度':<10}{'高度'}")
            print("-"*80)
            
            for i, box in enumerate(boxes, 1):
                score = box['score']
                coord = box['coordinate']
                
                width = coord[2] - coord[0]
                height = coord[3] - coord[1]
                
                coord_str = f"[{coord[0]:6.1f}, {coord[1]:6.1f}, {coord[2]:6.1f}, {coord[3]:6.1f}]"
                print(f"{i:<6}{score:<12.6f}{coord_str:<45}{width:<10.1f}{height:.1f}")
            
            print("="*80)

            print("\n置信度统计:")
            print(f"  总单元格数: {len(boxes)}")
            print(f"  平均置信度: {avg_score:.6f} ({avg_score*100:.2f}%)")
            print(f"  最高置信度: {max_score:.6f} ({max_score*100:.2f}%)")
            print(f"  最低置信度: {min_score:.6f} ({min_score*100:.2f}%)")

            high_conf = sum(1 for s in scores if s > 0.95)
            mid_conf = sum(1 for s in scores if 0.90 < s <= 0.95)
            low_conf = sum(1 for s in scores if s <= 0.90)
            
            print("\n置信度分布:")
            print(f"  高置信度 (>0.95):        {high_conf} 个 ({high_conf/len(boxes)*100:.1f}%)")
            print(f"  中等置信度 (0.90-0.95):  {mid_conf} 个 ({mid_conf/len(boxes)*100:.1f}%)")
            print(f"  较低置信度 (<=0.90):     {low_conf} 个 ({low_conf/len(boxes)*100:.1f}%)")

print("\n" + "="*80)
print("✓ 处理完成！")
print("="*80)
print("生成的文件:")
print("  1. /root/projects/PaddleOCR/tests/Table/outputs/cells/cells_fully_annotated.jpg  (带序号和详细标注)")
print("  2. /root/projects/PaddleOCR/tests/Table/outputs/cells/cells_simple_annotated.jpg (简洁版本)")
print("  3. /root/projects/PaddleOCR/tests/Table/outputs/cells/res.json                   (JSON结果)")
print("  4. /root/projects/PaddleOCR/tests/Table/outputs/cells/原始可视化图片              (模型自动生成)")
print("="*80)