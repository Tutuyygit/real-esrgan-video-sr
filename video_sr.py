import cv2
import numpy as np
import onnxruntime
import time
import os

# === 1. 加载模型 ===
session = onnxruntime.InferenceSession("Real-ESRGAN-General-x4v3.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# === 2. 打开视频文件 ===
video_path = "input_720p.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"❌ 无法打开视频源：{video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📹 输入视频分辨率：{width}x{height} @ {fps:.0f}fps")

# === 3. 配置输出 4K 视频写入器 ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter("output_4k.mp4", fourcc, fps, (width * 4, height * 4))

frame_count = 0
start_time = time.time()

# === 4. 帧循环处理 ===
while True:
    ret, frame = cap.read()
    if not ret:
        break
    sr_start = time.time()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w, _ = img.shape
    h_pad = (128 - h % 128) % 128
    w_pad = (128 - w % 128) % 128
    img_padded = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), mode='reflect')

    H, W = img_padded.shape[:2]
    h_blocks = H // 128
    w_blocks = W // 128

    # 储存 super-res 块
    sr_blocks = []

    for i in range(h_blocks):
        row_blocks = []
        for j in range(w_blocks):
            patch = img_padded[i*128:(i+1)*128, j*128:(j+1)*128, :]
            patch = np.transpose(patch, (2, 0, 1))  # HWC -> CHW
            patch = np.expand_dims(patch, axis=0)  # -> NCHW

            # 推理
            output = session.run(None, {input_name: patch})[0]
            output = np.clip(output, 0, 1)[0]
            output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
            row_blocks.append(output)
        sr_blocks.append(np.concatenate(row_blocks, axis=1))

    sr_image = np.concatenate(sr_blocks, axis=0)

    # 裁剪去padding
    sr_image = sr_image[:h*4, :w*4, :]
    output_bgr = cv2.cvtColor((sr_image * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

    out_writer.write(output_bgr)
    cv2.imshow("Super Resolution - 4K", output_bgr)
    if cv2.waitKey(1) == 27: break

    frame_count += 1
    sr_end = time.time()
    print(f"✅ 第 {frame_count} 帧处理完成，耗时 {sr_end - sr_start:.2f}s")

# === 5. 资源释放 ===
cap.release()
out_writer.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time
print(f"🎞️ 视频处理完成，共 {frame_count} 帧，用时 {total_time:.2f} 秒")
print(f"📁 输出视频保存在：{os.path.abspath('output_4k.mp4')}")