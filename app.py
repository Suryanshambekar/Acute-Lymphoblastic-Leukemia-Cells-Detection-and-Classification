import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Class names (Edit based on your model training)
class_names = ['Benign', 'Early', 'Pre', 'Pro']

# Title and Description
st.title("Acute Lymphoblastic Leukemia Cell Detection")
st.markdown("Upload an image to detect and classify leukemia cells using YOLOv8.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Process image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run inference
    st.write("Running model inference...")
    results = model.predict(image)

    # Get predictions
    boxes = results[0].boxes
    num_cells = len(boxes)
    confidences = [float(box.conf[0]) for box in boxes]
    classes = [int(box.cls[0]) for box in boxes]
    class_labels = [model.names[cls] for cls in classes]

    if num_cells == 0:
        st.warning("No leukemia cells detected.")
    else:
        # Show results
        st.success(f"Number of cells detected: {num_cells}")

        for i, (cls, conf) in enumerate(zip(class_labels, confidences), start=1):
            st.write(f"**Cell {i}**: Class = {cls}, Confidence = {conf:.2f}")

        avg_conf = np.mean(confidences)
        st.write(f"**Average Confidence**: {avg_conf:.2f}")

        # Final Classification as Most Common Class
        from collections import Counter
        most_common_class = Counter(class_labels).most_common(1)[0][0]
        st.write(f"**Final Classification (Most Common)**: {most_common_class}")

        # Display image with bounding boxes
        results[0].show()
        st.image(results[0].plot(), caption="Detected Cells", use_container_width=True)
