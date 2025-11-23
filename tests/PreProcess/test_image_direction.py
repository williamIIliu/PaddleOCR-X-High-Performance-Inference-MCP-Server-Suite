from paddleocr import DocImgOrientationClassification

model = DocImgOrientationClassification(
    model_name="PP-LCNet_x1_0_doc_ori",
    # model_dir="./pretrained_models/doc_ori/PP-LCNet_x1_0_doc_ori_infer",
    device="gpu:0",  # 或 "cpu"
    enable_mkldnn=True,
    cpu_threads=8
)

image_path = "/root/projects/PaddleOCR/tests/PreProcess/files/image_direction.jpg"
output = model.predict(image_path, batch_size=1)

for res in output:
    res.print(json_format=False)

    res.save_to_img("/root/projects/PaddleOCR/tests/PreProcess/outputs/image_direction.png")
    res.save_to_json("/root/projects/PaddleOCR/tests/PreProcess/outputs/image_direction.json")

    print("JSON结果:", res.json)