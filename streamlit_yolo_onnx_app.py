
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import onnxruntime as ort
import os
import tempfile
import time

st.set_page_config(layout="centered",
                   page_title="Waste Detection YOLO - ONNX Inference")

DEFAULT_MODEL_PATH = r"c:\Users\ayush\OneDrive\Pictures\Documents\Desktop\waste dataset 10\best.onnx"

# Class names from data.yaml
DEFAULT_CLASS_NAMES = ['battery', 'can', 'cardboard_bowl', 'cardboard_box', 'chemical_plastic_bottle',
                       'chemical_plastic_gallon', 'chemical_spray_can', 'light_bulb', 'paint_bucket',
                       'plastic_bag', 'plastic_bottle', 'plastic_bottle_cap', 'plastic_box',
                       'plastic_cultery', 'plastic_cup', 'plastic_cup_lid', 'reuseable_paper',
                       'scrap_paper', 'scrap_plastic', 'snack_bag', 'stick', 'straw', 'Beef',
                       'Cabbage', 'Carrot', 'Chicken', 'Cucumber', 'Egg', 'Eggplant', 'Leek',
                       'Onion', 'Pork', 'Potato', 'Radish', 'Tomato']


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    # resize
    resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # pad
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def xywh2xyxy(x):
    # x: (n,4) with cx,cy,w,h
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2]/2
    y[:, 1] = x[:, 1] - x[:, 3]/2
    y[:, 2] = x[:, 0] + x[:, 2]/2
    y[:, 3] = x[:, 1] + x[:, 3]/2
    return y


def nms_numpy(boxes, scores, iou_threshold=0.45):
    # boxes: Nx4 (x1,y1,x2,y2), scores: N
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


@st.cache_resource
def load_onnx_session(model_path):
    providers = None
    # Let ONNX Runtime choose; if user wants GPU, they should install onnxruntime-gpu
    try:
        sess = ort.InferenceSession(model_path, providers=[
                                    'CPUExecutionProvider'])
    except Exception:
        sess = ort.InferenceSession(model_path)
    return sess


def preprocess_image(pil_img, img_size=640):
    img = np.array(pil_img.convert("RGB"))
    # Keep image in RGB format (no BGR conversion needed)
    img_resized, ratio, (pad_x, pad_y) = letterbox(
        img, new_shape=(img_size, img_size))
    # Convert to float32, normalize to [0,1]
    img_resized = img_resized.astype(np.float32) / 255.0
    # HWC to CHW
    img_resized = img_resized.transpose(2, 0, 1)
    # Add batch dimension
    img_resized = np.expand_dims(img_resized, axis=0).astype(np.float32)
    return img_resized, ratio, pad_x, pad_y, img


def postprocess_output(output, ratio, pad_x, pad_y, orig_img_shape, conf_thres=0.25, iou_thres=0.45, class_names=None):
    """
    Generic postprocessing for common YOLOv8 ONNX exports that output (1, N, 85)
    where 85 = [x, y, w, h, obj_conf, class_probs...]
    """
    if isinstance(output, (list, tuple)):
        output = output[0]
    out = output.squeeze()
    if out.size == 0:
        return []
    # handle both (N,85) and (N,6) formats
    if out.shape[1] >= 5:
        # xywh, objectness, classes...
        boxes_xywh = out[:, 0:4].copy()
        scores_obj = out[:, 4:5]
        class_probs = out[:, 5:]
        # compute class-specific scores
        scores_all = scores_obj * class_probs  # (N, num_classes)
        # flatten detections per class
        boxes_xyxy = xywh2xyxy(boxes_xywh)
        detections = []
        num_classes = scores_all.shape[1]
        for cls in range(num_classes):
            cls_scores = scores_all[:, cls]
            mask = cls_scores > conf_thres
            if not np.any(mask):
                continue
            cls_boxes = boxes_xyxy[mask]
            cls_scores_f = cls_scores[mask]
            # boxes are relative to padded/resized image; convert to original image coords
            # remove padding and divide by ratio
            cls_boxes[:, [0, 2]] -= pad_x
            cls_boxes[:, [1, 3]] -= pad_y
            cls_boxes /= ratio
            # clip to original image
            h0, w0 = orig_img_shape[0], orig_img_shape[1]
            cls_boxes[:, 0] = np.clip(cls_boxes[:, 0], 0, w0-1)
            cls_boxes[:, 1] = np.clip(cls_boxes[:, 1], 0, h0-1)
            cls_boxes[:, 2] = np.clip(cls_boxes[:, 2], 0, w0-1)
            cls_boxes[:, 3] = np.clip(cls_boxes[:, 3], 0, h0-1)
            # NMS
            keep = nms_numpy(cls_boxes, cls_scores_f, iou_threshold=iou_thres)
            for i in keep:
                score = float(cls_scores_f[i])
                box = cls_boxes[i].tolist()
                label = class_names[cls] if class_names and cls < len(
                    class_names) else str(cls)
                detections.append(
                    {"box": box, "score": score, "class_id": int(cls), "label": label})
        return detections
    elif out.shape[1] == 6:
        # assume format [x1,y1,x2,y2,score,classid]
        detections = []
        for row in out:
            score = float(row[4])
            if score < conf_thres:
                continue
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            cls = int(row[5])
            label = class_names[cls] if class_names and cls < len(
                class_names) else str(cls)
            detections.append(
                {"box": [x1, y1, x2, y2], "score": score, "class_id": cls, "label": label})
        return detections
    else:
        return []


def draw_detections(orig_img_bgr, detections):
    # orig_img_bgr is numpy RGB image (despite the parameter name)
    img = orig_img_bgr.copy()
    # No color conversion needed - image is already in RGB format
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        score = det["score"]
        label = det["label"]
        # draw
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"{label} {score:.2f}"
        # Use textbbox instead of deprecated textsize
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")
        draw.text((x1, y1 - text_height), text, fill="white", font=font)
    return pil_img


st.title("🗑️ Waste Detection - YOLO ONNX Model")
st.markdown("""
This app uses a trained YOLO model (ONNX format) to detect waste items and food in uploaded images.  
The model can identify 35 different classes including various types of waste materials and food items.  
**Upload an image to get started!**
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Model Settings")
    model_file = st.file_uploader(
        "Upload a .onnx model (optional)", type=["onnx"])
    model_path_input = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    img_size = st.number_input(
        "Model input image size (px)", min_value=64, max_value=2048, value=640, step=1)
    conf_thres = st.slider("Confidence threshold", 0.0, 1.0, 0.25)
    iou_thres = st.slider("NMS IoU threshold", 0.0, 1.0, 0.45)

    use_default_names = st.checkbox("Use default class names", value=True)

    if use_default_names:
        class_names = DEFAULT_CLASS_NAMES
        st.info(f"Using {len(class_names)} default classes")
    else:
        class_names_text = st.text_area(
            "Class names (comma-separated)", height=80)
        if class_names_text:
            class_names = [x.strip()
                           for x in class_names_text.split(",") if x.strip()]
        else:
            class_names = None

with col2:
    st.subheader("📸 Upload Image")
    uploaded_img = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"])
    run_button = st.button("🚀 Run Detection", type="primary")

if model_file is not None:
    # write temporary model file
    tpath = os.path.join(tempfile.gettempdir(), model_file.name)
    with open(tpath, "wb") as f:
        f.write(model_file.read())
    model_path = tpath
else:
    model_path = model_path_input

if run_button:
    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
    elif uploaded_img is None:
        st.error("Please upload an image to run inference.")
    else:
        st.info("Loading model...")
        try:
            sess = load_onnx_session(model_path)
        except Exception as e:
            st.exception(e)
            st.stop()

        # show model input info
        inp = sess.get_inputs()[0]
        st.write("Model input name:", inp.name)
        st.write("Model input shape (may contain dynamic dims):", inp.shape)
        st.write("Model input type:", inp.type)

        # preprocess
        pil_img = Image.open(uploaded_img).convert("RGB")
        img_tensor, ratio, pad_x, pad_y, orig_bgr = preprocess_image(
            pil_img, img_size)
        input_name = sess.get_inputs()[0].name
        t0 = time.time()
        try:
            outputs = sess.run(None, {input_name: img_tensor})
        except Exception as e:
            st.exception(e)
            st.stop()
        t1 = time.time()

        st.write(f"Inference done in {(t1 - t0) * 1000:.1f} ms (model only).")
        # postprocess
        detections = postprocess_output(
            outputs, ratio, pad_x, pad_y, orig_bgr.shape, conf_thres, iou_thres, class_names)

        if len(detections) == 0:
            st.warning("No detections above confidence threshold.")
            st.image(pil_img, caption="Input image", use_column_width=True)
        else:
            annotated = draw_detections(orig_bgr, detections)
            st.image(annotated, caption="Detections", use_column_width=True)
            st.subheader("Detections")
            for d in detections:
                st.write(
                    f"{d['label']} — {d['score']:.3f} — box: {np.round(d['box'], 1).tolist()}")

        st.subheader("Raw model outputs (first tensor)")
        st.write([o.shape for o in outputs])
        # show a small slice of data to help debug
        try:
            arr = outputs[0]
            st.write("example slice of the output array (first 5 rows):")
            st.write(np.array(arr).reshape(-1, arr.shape[-1])[:5].tolist())
        except Exception:
            pass
