# Real-ESRGAN Video Super-Resolution (ONNX Inference)

> 🚀 A lightweight, GPU-accelerated video upscaling tool based on Real-ESRGAN and ONNX runtime.  
> 🎬 Input: 720p video → Output: 4K super-resolved video.

---

## 🧠 Features

- ✅ Supports **ONNXRuntime** inference with **CUDA** acceleration.
- 📦 No need to install PyTorch or original Real-ESRGAN dependencies.
- 🎥 Converts videos frame-by-frame with upscaling factor **x4**.
- 🖥️ Tested on 720p → 4K, ~9 seconds per frame on CPU, real-time possible with GPU.

---

## 📁 Directory Structure
Real-ESRGAN-SR/
├── RealESRGAN_x4plus-dynamic.onnx    # ONNX model file (x4 upscaling)
├── video_sr.py                       # Super-resolution script
├── requirements.txt                  # Lightweight dependency list
└── README.md                         # This file
---

## 🔧 Installation

Create and activate a Python environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows

Then install dependencies:
pip install -r requirements.txt

▶️ Usage
Put your input video (e.g., input_720p.mp4) in the project directory, then run:
Output: output_4k.mp4 in the same folder.

The script:
	•	Loads the ONNX model
	•	Reads input video frame by frame
	•	Performs x4 super-resolution
	•	Writes the output as 4K video

⸻

📷 Example

Input:


Output (Super-resolved):


(Replace with your own before/after images if you want)

⸻

💬 Notes
	•	The current ONNX model (RealESRGAN_x4plus-dynamic.onnx) must support dynamic input shape.
	•	Make sure your GPU supports CUDA and is recognized by ONNXRuntime.
	•	Do not include model weights > 100MB on GitHub (use Git LFS if needed).

⸻

📜 License

This project is adapted from Real-ESRGAN, which is released under the BSD-3-Clause License.

⸻

🙋‍♂️ Contact

Maintainer: tutuyygit
Pull requests and issues welcome!