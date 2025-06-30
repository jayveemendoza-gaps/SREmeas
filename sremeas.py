import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageEnhance

st.set_page_config(layout="wide")
st.title("🍋 Mango SER meas")

uploaded_file = st.file_uploader("Upload an image of mangoes (top view)", type=["png", "jpg", "jpeg"])

@st.cache_data
def convert_to_hsv(image_np):
    """Cache HSV conversion to avoid recomputation."""
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

def resize_with_aspect_ratio(image, target_width=800):
    """Resize an image while maintaining its aspect ratio."""
    w, h = image.size
    aspect_ratio = w / h
    if aspect_ratio > 1:  # Wider than tall
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:  # Taller than wide
        new_height = target_width
        new_width = int(target_width * aspect_ratio)
    return image.resize((new_width, new_height), Image.LANCZOS), new_width, new_height

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure no alpha channel
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
        max_dim = 800  # Reduce further for faster processing
        image, new_width, new_height = resize_with_aspect_ratio(image, target_width=max_dim)
        image_np = np.array(image)
        h, w = new_height, new_width
        st.info(f"Image downscaled to {w}x{h} for faster processing.")

    # --- Step 1: Set scale ---
    st.markdown("## 1️⃣ Draw a line on the scale bar in the image")
    image, new_width, new_height = resize_with_aspect_ratio(image, target_width=800)
    scale_canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=5,
        background_image=image,  # Maintain aspect ratio
        update_streamlit=True,
        height=new_height,  # Adjust height
        width=new_width,    # Adjust width
        drawing_mode="line",
        key="scale_canvas",
    )

    scale_length_mm = st.number_input(
        "Enter the real-world length of the drawn line (mm):",
        min_value=0.1,
        value=10.0  # Set default value to 10 mm
    )
    scale_px = None

    if scale_canvas.json_data and len(scale_canvas.json_data["objects"]) > 0:
        obj = scale_canvas.json_data["objects"][-1]
        if obj["type"] == "line":
            x0, y0 = obj["x1"], obj["y1"]
            x1, y1 = obj["x2"], obj["y2"]
            scale_px = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            st.info(f"Line length: {scale_px:.2f} pixels")

    if scale_px and scale_length_mm > 0:
        mm_per_px = scale_length_mm / scale_px
        st.success(f"Scale set: 1 pixel = {mm_per_px:.4f} mm")

        # --- Step 2: Mango sampling ---
        st.markdown("## 2️⃣ Adjust Brightness (Optional)")

        # Add a slider for brightness adjustment
        brightness_factor = st.slider(
            "Adjust image brightness (default: 1.0)",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1
        )

        # Apply brightness adjustment using PIL
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        image_np = np.array(image)  # Update the numpy array after brightness adjustment

        st.markdown("## 3️⃣ Draw around a mango (one at a time)")
        drawing_mode = st.selectbox("Drawing mode", ["freedraw", "circle", "rect"], index=1)
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",
            stroke_width=3,
            background_image=image,  # Use PIL Image for best compatibility
            update_streamlit=True,
            height=h,
            width=w,
            drawing_mode=drawing_mode,
            key="mango_canvas",
        )

        if "samples" not in st.session_state:
            st.session_state.samples = []

        if canvas_result.image_data is not None and np.any(canvas_result.image_data != 255):
            # Convert the canvas result to a binary mask
            mask = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
            mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]

            # Convert the image to HSV for color-based masking
            hsv = convert_to_hsv(image_np)

            # Define color ranges for masks
            green_yellow_lower = np.array([15, 40, 40])  # Combined green and yellow range
            green_yellow_upper = np.array([90, 255, 255])
            lesion_lower = np.array([0, 0, 0])
            lesion_upper = np.array([40, 255, 120])

            # Precompute the HSV mask
            hsv_mask = (hsv[:, :, 0] >= green_yellow_lower[0]) & (hsv[:, :, 0] <= green_yellow_upper[0]) & \
                       (hsv[:, :, 1] >= green_yellow_lower[1]) & (hsv[:, :, 1] <= green_yellow_upper[1]) & \
                       (hsv[:, :, 2] >= green_yellow_lower[2]) & (hsv[:, :, 2] <= green_yellow_upper[2])

            lesion_mask = (hsv[:, :, 0] >= lesion_lower[0]) & (hsv[:, :, 0] <= lesion_upper[0]) & \
                          (hsv[:, :, 1] >= lesion_lower[1]) & (hsv[:, :, 1] <= lesion_upper[1]) & \
                          (hsv[:, :, 2] >= lesion_lower[2]) & (hsv[:, :, 2] <= lesion_upper[2])

            # Apply user mask directly to boolean arrays (faster)
            mango_mask_bool = (hsv_mask | lesion_mask) & (mask == 255)
            lesion_mask_bool = lesion_mask & (mask == 255)

            # Calculate areas directly from boolean arrays
            mango_area_px = np.sum(mango_mask_bool)
            lesion_area_px = np.sum(lesion_mask_bool)

            # Convert to uint8 only for display
            mango_mask = mango_mask_bool.astype(np.uint8) * 255
            lesion_mask = lesion_mask_bool.astype(np.uint8) * 255

            # Convert pixel areas to mm²
            mango_area_mm2 = mango_area_px * (mm_per_px ** 2)
            lesion_area_mm2 = lesion_area_px * (mm_per_px ** 2)
            lesion_percent = (lesion_area_mm2 / mango_area_mm2 * 100) if mango_area_mm2 else 0

            # Display results
            st.markdown("### 🟩 Selected Mango & Lesions")
            col1, col2 = st.columns(2)
            col1.image(mango_mask, caption="Total Mango Area (Green/Yellow + Lesions)", use_column_width=True)
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