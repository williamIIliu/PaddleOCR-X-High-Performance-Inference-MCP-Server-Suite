from paddleocr import TableClassification
import os
import cv2



model = TableClassification(
    model_name="PP-LCNet_x1_0_table_cls",
    # model_dir="/mnt/2/hc/PaddleOCR/pretrained_models/table_cls/PP-LCNet_x1_0_table_cls_infer",
    device="gpu:0",
)

input_image = "/root/projects/PaddleOCR/tests/Table/files/Table1.png"
output_path = "/root/projects/PaddleOCR/tests/Table/outputs/class/"
os.makedirs(output_path, exist_ok=True)

output = model.predict(input_image, batch_size=1)
for res in output:
    res.print()
    res.save_to_json(output_path+"table_class.json")

    result = res.json['res']
    scores = result['scores']
    labels = result['label_names']

    import numpy as np
    if isinstance(scores, list):
        scores = np.array(scores)
    
    max_idx = int(scores.argmax())
    
    print(f"\n类型: {labels[max_idx]}, 置信度: {scores[max_idx]*100:.2f}%")

    img = cv2.imread(input_image)

    text = f"{labels[max_idx]} ({scores[max_idx]*100:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # 获取文本大小
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 设置padding
    padding = 8
    box_width = text_width + padding * 2
    box_height = text_height + baseline + padding * 2
    
    # 绘制更小的框
    cv2.rectangle(img, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
    cv2.putText(img, text, 
               (10 + padding, 10 + text_height + padding), 
               font, font_scale, (0, 255, 0), thickness)
    
    cv2.imwrite(output_path + "tabel_class.jpg", img)

print("\n✓ 完成！")