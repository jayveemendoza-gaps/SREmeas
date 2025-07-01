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
def convert_to_hsv(_image_np):
    """Cache HSV conversion to avoid recomputation. Use underscore to ignore hash."""
    return cv2.cvtColor(_image_np, cv2.COLOR_RGB2HSV)

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
    # Use consistent dimensions for both canvases
    display_image, display_width, display_height = resize_with_aspect_ratio(image, target_width=800)
    scale_canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=5,
        background_image=display_image,
        update_streamlit=True,
        height=display_height,
        width=display_width,
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
            background_image=display_image,  # Use same image as scale canvas
            update_streamlit=True,
            height=display_height,  # Use same dimensions
            width=display_width,
            drawing_mode=drawing_mode,
            key="mango_canvas",
        )

        if "samples" not in st.session_state:
            st.session_state.samples = []

        if canvas_result.image_data is not None and np.any(canvas_result.image_data != 255):
            # Show processing indicator
            with st.spinner("Processing mango area..."):
                # Get user mask first (much faster)
                mask = canvas_result.image_data[:, :, 3]  # Use alpha channel directly
                user_mask = mask > 10
                
                # Early exit if no pixels selected
                if not np.any(user_mask):
                    st.warning("No area selected. Please draw on the mango.")
                    st.stop()
                
                # Only process pixels within the user mask (HUGE speed boost)
                y_coords, x_coords = np.where(user_mask)
                if len(y_coords) == 0:
                    st.warning("No valid area selected.")
                    st.stop()
                
                # Get bounding box to reduce processing area even further
                y_min, y_max = max(0, y_coords.min()), min(image_np.shape[0], y_coords.max() + 1)
                x_min, x_max = max(0, x_coords.min()), min(image_np.shape[1], x_coords.max() + 1)
                
                # Validate bounding box
                if y_max <= y_min or x_max <= x_min:
                    st.error("Invalid selection area. Please try again.")
                    st.stop()
                
                # Crop image to bounding box only
                crop_image = image_np[y_min:y_max, x_min:x_max]
                crop_mask = user_mask[y_min:y_max, x_min:x_max]
                
                # Convert only the cropped area to HSV (much faster)
                hsv_crop = cv2.cvtColor(crop_image, cv2.COLOR_RGB2HSV)
                
                # Define color ranges for masks
                green_yellow_lower = np.array([15, 40, 40])
                green_yellow_upper = np.array([90, 255, 255])
                lesion_lower = np.array([0, 0, 0])
                lesion_upper = np.array([40, 255, 120])
                
                # Apply masks only to cropped area
                hsv_mask_crop = cv2.inRange(hsv_crop, green_yellow_lower, green_yellow_upper)
                lesion_mask_crop = cv2.inRange(hsv_crop, lesion_lower, lesion_upper)
                
                # Apply user mask to cropped masks
                hsv_mask_crop = (hsv_mask_crop == 255) & crop_mask
                lesion_mask_crop = (lesion_mask_crop == 255) & crop_mask
                
                # Calculate areas directly from cropped boolean arrays
                mango_area_px = np.sum(hsv_mask_crop | lesion_mask_crop)
                lesion_area_px = np.sum(lesion_mask_crop)
                
                # Create full-size masks for display (only if needed)
                mango_mask = np.zeros_like(user_mask, dtype=np.uint8)
                lesion_mask = np.zeros_like(user_mask, dtype=np.uint8)
                
                mango_mask[y_min:y_max, x_min:x_max] = (hsv_mask_crop | lesion_mask_crop).astype(np.uint8) * 255
                lesion_mask[y_min:y_max, x_min:x_max] = lesion_mask_crop.astype(np.uint8) * 255
            # Convert pixel areas to mm² with safety checks
            if mm_per_px <= 0:
                st.error("Invalid scale conversion. Please set the scale again.")
                st.stop()
                
            mango_area_mm2 = mango_area_px * (mm_per_px ** 2)
            lesion_area_mm2 = lesion_area_px * (mm_per_px ** 2)
            lesion_percent = (lesion_area_mm2 / mango_area_mm2 * 100) if mango_area_mm2 > 0 else 0

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

            # Safe delete implementation with better state management
            st.markdown("#### Remove Samples")
            if len(st.session_state.samples) > 0:
                sample_to_delete = st.selectbox(
                    "Select sample to delete:",
                    options=range(len(st.session_state.samples)),
                    format_func=lambda x: f"Sample {st.session_state.samples[x]['Sample #']}",
                    key="delete_selector"
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🗑️ Delete Selected", key="delete_button"):
                        if 0 <= sample_to_delete < len(st.session_state.samples):
                            deleted_sample = st.session_state.samples.pop(sample_to_delete)
                            # Renumber remaining samples
                            for i, sample in enumerate(st.session_state.samples):
                                sample["Sample #"] = i + 1
                            st.success(f"Deleted sample {deleted_sample['Sample #']}")
                            st.rerun()
                with col2:
                    if st.button("🗑️ Clear All Samples", key="clear_all"):
                        st.session_state.samples = []
                        st.success("All samples cleared!")
                        st.rerun()

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