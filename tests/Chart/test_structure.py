from paddleocr import TableStructureRecognition
import os
import cv2
import numpy as np



model = TableStructureRecognition(
    model_name="SLANet_plus",
    # model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/table_structure/SLANet_plus_infer",
    device="gpu:0",
)

image_path = "/root/projects/PaddleOCR/tests/Table/files/Table1.png"
print(f"处理表格图像: {image_path}\n")
output = model.predict(image_path, batch_size=1)
output_path = "/root/projects/PaddleOCR/tests/Table/outputs/structure/"
os.makedirs(output_path, exist_ok=True)

for res in output:
    json_result = res.json
    result_data = json_result['res']
    
    bbox = result_data.get('bbox', [])
    structure = result_data.get('structure', [])
    structure_score = result_data.get('structure_score', 0)
    
    print(f"检测到 {len(bbox)} 个单元格")
    print(f"结构置信度: {structure_score:.4f}\n")

    print("="*80)
    print("单元格坐标详细解释：")
    print("="*80)
    
    for i, cell in enumerate(bbox[:5], 1):  # 只显示前5个
        if len(cell) >= 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = cell
            
            # 计算尺寸
            width = x2 - x1
            height = y3 - y1
            
            print(f"\n单元格 {i}:")
            print(f"  原始坐标: {cell}")
            print(f"  左上角 (Top-Left):     ({x1:6.1f}, {y1:6.1f})")
            print(f"  右上角 (Top-Right):    ({x2:6.1f}, {y2:6.1f})")
            print(f"  右下角 (Bottom-Right): ({x3:6.1f}, {y3:6.1f})")
            print(f"  左下角 (Bottom-Left):  ({x4:6.1f}, {y4:6.1f})")
            print(f"  尺寸: 宽度={width:.1f}px, 高度={height:.1f}px")
            print(f"  中心点: ({(x1+x2)/2:.1f}, {(y1+y3)/2:.1f})")
    
    if len(bbox) > 5:
        print(f"\n... 还有 {len(bbox)-5} 个单元格")
    
    print("\n" + "="*80)
    
    # 读取原图
    img = cv2.imread(image_path)
    
    if img is not None:
        # 创建标注图像
        img_annotated = img.copy()
        
        # 设置颜色
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 黄色
        ]
        
        for i, cell in enumerate(bbox):
            if len(cell) >= 8:
                # 提取四个顶点
                x1, y1, x2, y2, x3, y3, x4, y4 = cell
                
                # 转换为整数坐标
                pts = np.array([
                    [int(x1), int(y1)],  # 左上
                    [int(x2), int(y2)],  # 右上
                    [int(x3), int(y3)],  # 右下
                    [int(x4), int(y4)]   # 左下
                ], np.int32)
                
                # 选择颜色
                color = colors[i % len(colors)]
                
                # 绘制多边形边框
                cv2.polylines(img_annotated, [pts], True, color, 2)
                
                # 标注四个顶点
                cv2.circle(img_annotated, (int(x1), int(y1)), 5, (0, 255, 0), -1)    # 左上 - 绿色
                cv2.circle(img_annotated, (int(x2), int(y2)), 5, (255, 0, 0), -1)    # 右上 - 蓝色
                cv2.circle(img_annotated, (int(x3), int(y3)), 5, (0, 0, 255), -1)    # 右下 - 红色
                cv2.circle(img_annotated, (int(x4), int(y4)), 5, (255, 255, 0), -1)  # 左下 - 青色
                
                # 添加单元格编号
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y3) / 2)
                cv2.putText(img_annotated, f"#{i+1}", 
                           (center_x-15, center_y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(img_annotated, f"#{i+1}", 
                           (center_x-15, center_y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 添加图例
        legend_y = 30
        cv2.putText(img_annotated, "Legend:", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.circle(img_annotated, (100, legend_y-5), 5, (0, 255, 0), -1)
        cv2.putText(img_annotated, "Top-Left", (110, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.circle(img_annotated, (220, legend_y-5), 5, (255, 0, 0), -1)
        cv2.putText(img_annotated, "Top-Right", (230, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.circle(img_annotated, (350, legend_y-5), 5, (0, 0, 255), -1)
        cv2.putText(img_annotated, "Bottom-Right", (360, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.circle(img_annotated, (510, legend_y-5), 5, (255, 255, 0), -1)
        cv2.putText(img_annotated, "Bottom-Left", (520, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 保存可视化结果
        cv2.imwrite(output_path + "/table_cells_visualization.jpg", img_annotated)
        print(f"\n✓ 可视化结果已保存到: {output_path}")
    
    # 创建坐标说明文档
    with open(output_path + "./coordinates_explanation.txt", 'w', encoding='utf-8') as f:
        f.write("表格单元格坐标说明\n")
        f.write("="*80 + "\n\n")
        f.write("坐标格式: [x1, y1, x2, y2, x3, y3, x4, y4]\n\n")
        f.write("含义:\n")
        f.write("  (x1, y1) - 左上角 (Top-Left)\n")
        f.write("  (x2, y2) - 右上角 (Top-Right)\n")
        f.write("  (x3, y3) - 右下角 (Bottom-Right)\n")
        f.write("  (x4, y4) - 左下角 (Bottom-Left)\n\n")
        f.write("="*80 + "\n\n")
        
        for i, cell in enumerate(bbox, 1):
            if len(cell) >= 8:
                x1, y1, x2, y2, x3, y3, x4, y4 = cell
                width = x2 - x1
                height = y3 - y1
                
                f.write(f"单元格 {i}:\n")
                f.write(f"  原始坐标: {cell}\n")
                f.write(f"  左上角: ({x1:.1f}, {y1:.1f})\n")
                f.write(f"  右上角: ({x2:.1f}, {y2:.1f})\n")
                f.write(f"  右下角: ({x3:.1f}, {y3:.1f})\n")
                f.write(f"  左下角: ({x4:.1f}, {y4:.1f})\n")
                f.write(f"  尺寸: {width:.1f} x {height:.1f} 像素\n")
                f.write(f"  中心: ({(x1+x2)/2:.1f}, {(y1+y3)/2:.1f})\n")
                f.write("\n")
    
    print(f"✓ 坐标说明已保存到: {output_path}/coordinates_explanation.txt")
    
    # 保存HTML结构
    html_content = ''.join(structure)
    with open(output_path + "./table_structure.html", 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✓ HTML结构已保存到: {output_path} + './table_structure.html'")

print("\n处理完成！")