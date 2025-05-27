import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pandas as pd
from PIL import Image

st.set_page_config(layout="wide")
st.title("🍋 Mango SER meas")

uploaded_file = st.file_uploader("Upload an image of mangoes (top view)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    # --- Detect large image and prompt for quality ---
    large_image = (h * w > 2000 * 2000)  # Example threshold
    if large_image:
        st.warning(
            f"Your image is very large ({w}x{h}). High quality mode may be slow. "
            "Choose 'Normal' to process a downscaled version."
        )
        quality_choice = st.radio(
            "Select image processing quality:",
            ["Normal (recommended)", "High Quality (slow)"],
            index=0
        )
    else:
        quality_choice = "High Quality (slow)"

    # --- Downscale if needed ---
    if quality_choice == "Normal (recommended)":
        max_dim = 1200  # You can adjust this
        scale = min(max_dim / h, max_dim / w, 1.0)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            image = image.resize(new_size, Image.LANCZOS)
            image_np = np.array(image)
            h, w = image_np.shape[:2]
            st.info(f"Image downscaled to {w}x{h} for faster processing.")

    # --- Step 1: Set scale ---
    st.markdown("## 1️⃣ Draw a line on the scale bar in the image")
    scale_canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=5,
        background_image=image,  # Use PIL Image for compatibility
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode="line",
        key="scale_canvas",
    )

    scale_length_mm = st.number_input("Enter the real-world length of the drawn line (mm):", min_value=0.1)
    scale_px = None

    if scale_canvas.json_data and len(scale_canvas.json_data["objects"]) > 0:
        obj = scale_canvas.json_data["objects"][-1]
        # For a line, use x1, y1, x2, y2
        if obj["type"] == "line":
            x0, y0 = obj["x1"], obj["y1"]
            x1, y1 = obj["x2"], obj["y2"]
            scale_px = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            st.info(f"Line length: {scale_px:.2f} pixels")

    if scale_px and scale_length_mm > 0:
        mm_per_px = scale_length_mm / scale_px
        st.success(f"Scale set: 1 pixel = {mm_per_px:.4f} mm")

        # --- Step 2: Mango sampling ---
        st.markdown("## 2️⃣ Draw around a mango (one at a time)")
        drawing_mode = st.selectbox("Drawing mode", ["freedraw", "circle", "rect"], index=1)
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",
            stroke_width=3,
            background_image=image,  # Use PIL Image for compatibility
            update_streamlit=True,
            height=h,
            width=w,
            drawing_mode=drawing_mode,
            key="mango_canvas",
        )

        if "samples" not in st.session_state:
            st.session_state.samples = []

        if canvas_result.image_data is not None and np.any(canvas_result.image_data != 255):
            mask = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
            mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]

            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

            # Mask for green/yellow mango surface (tune these ranges as needed)
            green_lower = np.array([20, 40, 40])
            green_upper = np.array([90, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)

            yellow_lower = np.array([15, 80, 80])
            yellow_upper = np.array([40, 255, 255])
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

            # Mask for black/brown lesions
            lesion_lower = np.array([0, 0, 0])
            lesion_upper = np.array([40, 255, 120])
            lesion_mask = cv2.inRange(hsv, lesion_lower, lesion_upper)

            healthy_mask = cv2.bitwise_or(green_mask, yellow_mask)
            total_mango_mask = cv2.bitwise_or(healthy_mask, lesion_mask)
            total_mango_mask = cv2.bitwise_and(total_mango_mask, mask)

            lesion_mask = cv2.bitwise_and(lesion_mask, mask)

            mango_area_px = np.sum(total_mango_mask == 255)
            lesion_area_px = np.sum(lesion_mask == 255)

            mango_area_mm2 = mango_area_px * (mm_per_px ** 2)
            lesion_area_mm2 = lesion_area_px * (mm_per_px ** 2)
            lesion_percent = (lesion_area_mm2 / mango_area_mm2 * 100) if mango_area_mm2 else 0

            st.markdown("### 🟩 Selected Mango & Lesions")
            col1, col2 = st.columns(2)
            col1.image(total_mango_mask, caption="Total Mango Area (Green/Yellow + Lesions)", use_column_width=True)
            col2.image(lesion_mask, caption="Lesion Area (Black/Brown)", use_column_width=True)

            result = {
                "Sample #": len(st.session_state.samples) + 1,
                "Total Area (mm²)": round(mango_area_mm2, 2),
                "Lesion Area (mm²)": round(lesion_area_mm2, 2),
                "Lesion %": round(lesion_percent, 2)
            }
            st.markdown("### 📊 Current Sample Result")
            st.dataframe(pd.DataFrame([result]))

            if st.button("Add this mango as a sample"):
                st.session_state.samples.append(result)
                st.success(f"Sample {result['Sample #']} added! Draw the next mango.")

        if st.session_state.samples:
            st.markdown("### 🥭 All Mango Samples")
            all_samples_df = pd.DataFrame(st.session_state.samples)
            st.dataframe(all_samples_df)

            # Add delete buttons for each row
            for idx, row in all_samples_df.iterrows():
                col1, col2 = st.columns([8, 1])
                with col1:
                    st.write(row.to_dict())
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{idx}"):
                        st.session_state.samples.pop(idx)
                        st.experimental_rerun()

            # --- CSV export with custom filename ---
            csv_filename = st.text_input(
                "Enter filename for CSV export (without .csv):",
                value="mango_lesion_samples"
            )
            csv = all_samples_df.to_csv(index=False).encode()
            st.download_button(
                "📥 Download All Samples as CSV",
                csv,
                f"{csv_filename}.csv",
                "text/csv"
            )

# --- Footer ---
st.markdown("""
---
<center>
Plant Pathology Laboratory, Institute of Plant Breeding, CAFS, UPLB;  
Contact: <a href="mailto:jsmendoza5@up.edu.ph">jsmendoza5@up.edu.ph</a>
</center>
""", unsafe_allow_html=True)