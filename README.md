# Garbage Classification Project

This repository contains a complete end-to-end pipeline for a 6-class garbage classification task, from data preparation and model training in PyTorch, through conversion to TensorFlow formats, to inference with TensorFlow Lite.

---

## 📂 Repository Structure

submission/
├── saved_model/           # TensorFlow SavedModel (server/cloud)
│   ├── saved_model.pb
│   └── variables/
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── tflite/                # TensorFlow Lite (mobile/embedded)
│   ├── model.tflite
│   └── label.txt
├── tfjs_model/            # TensorFlow.js (browser/JavaScript)
│   ├── model.json
│   ├── group1-shard1of23.bin
│   ├── group1-shard2of23.bin
│   └── …  
│   └── group1-shard23of23.bin
├── notebook.ipynb         # All steps in one Colab notebook
├── requirements.txt       # Python dependencies
└── README.md              # This file

---

## 🗂️ Dataset & Splitting

- **Classes**: `paper`, `cardboard`, `glass`, `metal`, `plastic`, `trash`
- **Total images**: ~2500
- **Split**:
  - **Train**: 70%
  - **Validation**: 15%
  - **Test**: 15%
- Script automatically creates `garbage_split/{train,val,test}/{class}/` and copies images.

---

## 🛠️ Model Training (PyTorch)

- **Architecture**: ResNet-50 (pretrained base, fine-tuned all layers)
- **Transforms**: Resize → RandomHorizontalFlip → Normalize
- **Optimizer**: Adam (LR=1e-3) + StepLR scheduler
- **Early Stopping**: patience = 5
- **Result:  Best validation accuracy: 0.8628 (86.28%)**


- **Checkpoint**: saved as `best_resnet50.pth`

---

## 🔄 Conversion to TensorFlow Formats

1. **PyTorch → ONNX**  
 ```bash
 torch.onnx.export(model, dummy, "resnet50.onnx", opset_version=11)
````

2. **ONNX → TensorFlow SavedModel**

   ```bash
   pip install tf2onnx
   python -m tf2onnx.convert --input resnet50.onnx --saved-model saved_model
   ```

3. **SavedModel → TFLite**

   ```python
   converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
   tflite_model = converter.convert()
   open("tflite/model.tflite","wb").write(tflite_model)
   ```

4. **SavedModel → TFJS**

   ```bash
   pip install tensorflowjs
   tensorflowjs_converter \
     --input_format=tf_saved_model \
     --saved_model_tags=serve \
     saved_model tfjs_model
   ```

---

## 🤖 Inference with TFLite

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load labels
labels = open("tflite/label.txt").read().splitlines()

# Load interpreter
interpreter = tf.lite.Interpreter("tflite/model.tflite")
interpreter.allocate_tensors()

# Preprocess
img = Image.open("sample.jpg").resize((224,224))
data = np.expand_dims(np.array(img)/255.0,0).astype(np.float32)

# Run
inp, out = interpreter.get_input_details(), interpreter.get_output_details()
interpreter.set_tensor(inp[0]['index'], data)
interpreter.invoke()
pred = interpreter.get_tensor(out[0]['index'])
idx = np.argmax(pred)

print(f"Predicted: {labels[idx]} ({pred[0][idx]:.2f})")
```

*Example:*
![Inference Example]([/Screenshot 2025-05-12 131218.jpg](https://github.com/Fadelhamkaa/Klasifikasi-Gambar-Pengembangan-Machine-Learning/blob/main/Screenshot%202025-05-12%20131218.jpg))

---

## 📋 Requirements


torch
torchvision
onnx
tf2onnx
tensorflow>=2.11
tensorflowjs
numpy
Pillow
matplotlib
scikit-learn


---

## 🚀 How to Run

1. **Clone** this repo


   git clone <[your-repo-url](https://github.com/Fadelhamkaa/Klasifikasi-Gambar-Pengembangan-Machine-Learning.git)>
   cd submission


2. **Install dependencies**


   pip install -r requirements.txt


3. **Open** `notebook.ipynb` in Colab and run cells end-to-end.

4. **Inspect**:

   * `saved_model/` for server deployment
   * `tflite/` for mobile/embedded
   * `tfjs_model/` for web

---

## 📖 License

This project is released under the MIT License.
Feel free to adapt and reuse for your own experiments!

---

*Prepared by \[Muhammad Fadel Hamka], 2025*

```
```





