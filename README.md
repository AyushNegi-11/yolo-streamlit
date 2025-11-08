# 🗑️ Waste Detection - YOLOv8 ONNX Streamlit App

A real-time waste detection application using YOLOv8 ONNX model deployed with Streamlit.

## 🎯 Features

- Detect 35 different classes of waste materials and food items
- Real-time object detection with bounding boxes
- Adjustable confidence and NMS thresholds
- Support for custom ONNX models
- Easy-to-use web interface
- Fixed color rendering issue for accurate object visualization

## 📋 Classes Detected

The model can identify the following items:
- **Waste Materials**: battery, can, cardboard, plastic bottles, bags, cups, and more
- **Food Items**: beef, chicken, pork, vegetables (cabbage, carrot, cucumber, etc.)

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AyushNegi-11/yolo-streamlit.git
cd yolo-streamlit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run streamlit_yolo_onnx_app.py
```

## 📁 Project Structure

```
├── streamlit_yolo_onnx_app.py  # Main Streamlit application
├── app.py                       # Alternative app version
├── best.onnx                    # Trained YOLOv8 ONNX model
├── best.pt                      # Trained YOLOv8 PyTorch model
├── data.yaml                    # Dataset configuration
├── requirements.txt             # Python dependencies
├── packages.txt                 # System packages
└── runtime.txt                  # Python runtime version
```

## 🎮 Usage

1. Launch the app
2. Upload a `.onnx` model (optional) or use the default model
3. Adjust detection parameters:
   - Confidence threshold
   - NMS IoU threshold
   - Input image size
4. Upload an image
5. Click "Run Detection" to see results

## 🛠️ Model Information

- **Framework**: YOLOv8
- **Format**: ONNX (optimized for inference)
- **Input Size**: 640x640 (configurable)
- **Classes**: 35 (waste materials + food items)

## 📊 Dataset

The model is trained on a comprehensive waste detection dataset including:
- Various plastic products
- Metal items (cans, batteries)
- Paper and cardboard
- Food items
- Other recyclable materials

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

AyushNegi-11

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Streamlit for the web framework
- ONNX Runtime for optimized inference
