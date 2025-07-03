import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageEnhance
import io

# Use a better page config with caching hints
st.set_page_config(
    layout="wide",
    page_title="Mango SER meas",
    page_icon="🍋"
)
st.title("🍋 Mango SER meas")


@st.cache_data(show_spinner=False)
def load_image(_uploaded_file):
    """Load image from an uploaded file with error handling and caching."""
    try:
        image = Image.open(_uploaded_file)
        return image.convert("RGB")  # Standardize to RGB
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def resize_with_aspect_ratio(_image, target_width=800):
    """Resize an image while maintaining its aspect ratio, with caching."""
    w, h = _image.size
    aspect_ratio = w / h
    if aspect_ratio > 1:  # Wider than tall
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:  # Taller than wide
        new_height = target_width
        new_width = int(target_width * aspect_ratio)
    return _image.resize((new_width, new_height), Image.LANCZOS), new_width, new_height


# Remove caching from this function to prevent dependency issues
def create_display_image(image, target_width=800):
    """Create a properly sized display image for canvas backgrounds."""
    display_img, width, height = resize_with_aspect_ratio(image, target_width)
    return display_img, width, height


def convert_to_hsv(_image_np):
    """Cache HSV conversion to avoid recomputation. Use underscore to ignore hash."""
    return cv2.cvtColor(_image_np, cv2.COLOR_RGB2HSV)


def apply_color_mask(_hsv_img, _lower, _upper, _user_mask=None):
    """Cache mask application to avoid recomputation."""
    mask = cv2.inRange(_hsv_img, _lower, _upper)
    if _user_mask is not None:
        mask = (mask == 255) & _user_mask
    return mask


# Simplified area calculation without caching
def calculate_areas(mask1, mask2):
    """Calculate areas for better performance."""
    combined = mask1 | mask2
    return np.sum(combined), np.sum(mask2)


# Add proper file upload guidance
uploaded_file = st.file_uploader(
    "Upload an image of mangoes (top view)", 
    type=["png", "jpg", "jpeg"],
    help="For best results, use a clear image with good lighting"
)

if uploaded_file:
    # Initialize session state
    if "samples" not in st.session_state:
        st.session_state.samples = []
    
    try:
        # Load and immediately downscale image for all further use (faster)
        # Pass the uploaded_file object directly to the cached function
        image = load_image(uploaded_file)
        if image is None:
            st.error("Cannot load image. Please try a different file.")
            st.stop()

        max_dim = 600  # Use a smaller dimension for faster canvas and processing
        # Pass the image object to the cached resize function
        image, new_width, new_height = resize_with_aspect_ratio(image, target_width=max_dim)
        
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        st.info(f"Image downscaled to {w}x{h} for fast processing and display.")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.stop()

    # --- Step 1: Set scale ---
    st.markdown("## 1️⃣ Draw a line on the scale bar in the image")
    try:
        # No further resizing, use the already downscaled image
        display_image = image
        display_width, display_height = w, h
    except Exception as e:
        st.error(f"Error preparing display: {str(e)}")
        st.stop()
    with st.container():
        try:
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
        except Exception as e:
            st.error(f"Canvas initialization error: {str(e)}")
            st.info("Try refreshing the page or using a different browser.")
            st.stop()

    scale_length_mm = st.number_input(
        "Enter the real-world length of the drawn line (mm):",
        min_value=0.1,
        value=10.0
    )
    scale_px = None

    if scale_canvas.json_data and len(scale_canvas.json_data["objects"]) > 0:
        obj = scale_canvas.json_data["objects"][-1]
        if obj["type"] == "line":
            x0, y0 = obj["x1"], obj["y1"]
            x1, y1 = obj["x2"], obj["y2"]
            dx, dy = x1 - x0, y1 - y0
            scale_px = np.sqrt(dx*dx + dy*dy)
            st.info(f"Line length: {scale_px:.2f} pixels")

    if scale_px and scale_length_mm > 0:
        mm_per_px = scale_length_mm / scale_px
        st.success(f"Scale set: 1 pixel = {mm_per_px:.4f} mm")

        # --- Step 2: Mango sampling ---
        st.markdown("## 2️⃣ Adjust Brightness (Optional)")
        brightness_factor = st.slider(
            "Adjust image brightness (default: 1.0)",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1
        )
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            brightness_adjusted_image = enhancer.enhance(brightness_factor)
        else:
            brightness_adjusted_image = image
        brightness_adjusted_np = np.array(brightness_adjusted_image)
        # Use the same size for display as for processing
        brightness_display_image = brightness_adjusted_image

        # --- Step 3: Draw around a mango ---
        st.markdown("## 3️⃣ Draw around a mango (one at a time)")
        drawing_mode = st.selectbox(
            "Drawing mode", 
            ["circle", "rect", "freedraw", "transform"], 
            index=0,
            help="Circle mode is recommended - easiest to position and resize. Use 'transform' mode to move/resize existing shapes."
        )
        if drawing_mode == "transform":
            st.info("🔄 **Transform Mode**: Click and drag existing shapes to move them, or drag corners/edges to resize.")
        else:
            st.info(f"✏️ **{drawing_mode.title()} Mode**: Draw a new {drawing_mode}. Switch to 'transform' mode to move/resize existing shapes.")
        if drawing_mode == "freedraw":
            st.warning("Free draw mode may be slower. For best results, draw slowly and steadily.")

        with st.container():
            try:
                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.2)",
                    stroke_width=3,
                    background_image=brightness_display_image,
                    update_streamlit=True,
                    height=display_height,
                    width=display_width,
                    drawing_mode=drawing_mode,
                    key="mango_canvas",
                )
            except Exception as e:
                st.error(f"Canvas error: {str(e)}")
                st.stop()

        if canvas_result.image_data is not None and np.any(canvas_result.image_data != 255):
            with st.spinner("Processing mango area..."):
                mask = canvas_result.image_data[:, :, 3]
                user_mask = mask > 10
                if not np.any(user_mask):
                    st.warning("No area selected. Please draw on the mango.")
                    st.stop()
                y_indices, x_indices = np.nonzero(user_mask)
                if len(y_indices) == 0:
                    st.warning("No valid area selected.")
                    st.stop()
                y_min, y_max = np.min(y_indices), np.max(y_indices) + 1
                x_min, x_max = np.min(x_indices), np.max(x_indices) + 1
                y_min, y_max = max(0, y_min), min(brightness_adjusted_np.shape[0], y_max)
                x_min, x_max = max(0, x_min), min(brightness_adjusted_np.shape[1], x_max)
                if y_max <= y_min or x_max <= x_min:
                    st.error("Invalid selection area. Please try again.")
                    st.stop()
                crop_image = brightness_adjusted_np[y_min:y_max, x_min:x_max]
                crop_mask = user_mask[y_min:y_max, x_min:x_max]
                hsv_crop = cv2.cvtColor(crop_image, cv2.COLOR_RGB2HSV)
                green_yellow_lower = np.array([15, 40, 40])
                green_yellow_upper = np.array([90, 255, 255])
                lesion_lower = np.array([0, 0, 0])
                lesion_upper = np.array([40, 255, 120])
                hsv_mask_crop = cv2.inRange(hsv_crop, green_yellow_lower, green_yellow_upper)
                lesion_mask_crop = cv2.inRange(hsv_crop, lesion_lower, lesion_upper)
                hsv_mask_crop = (hsv_mask_crop == 255) & crop_mask
                lesion_mask_crop = (lesion_mask_crop == 255) & crop_mask
                mango_area_px, lesion_area_px = calculate_areas(hsv_mask_crop, lesion_mask_crop)
                mango_mask = np.zeros_like(user_mask, dtype=np.uint8)
                lesion_mask = np.zeros_like(user_mask, dtype=np.uint8)
                slice_height, slice_width = y_max - y_min, x_max - x_min
                combined_mask = (hsv_mask_crop | lesion_mask_crop).astype(np.uint8) * 255
                lesion_display_mask = lesion_mask_crop.astype(np.uint8) * 255
                # No resizing needed, all arrays are from the same (downscaled) image
                mango_mask[y_min:y_max, x_min:x_max] = combined_mask
                lesion_mask[y_min:y_max, x_min:x_max] = lesion_display_mask

            st.success("Processing complete! Review the results below.")
            try:
                if mm_per_px <= 0:
                    st.error("Invalid scale conversion. Please set the scale again.")
                    st.stop()
                if mango_area_px == 0:
                    st.warning("No mango area detected. Try adjusting the brightness or drawing a different area.")
                    st.stop()
                mm_per_px_squared = mm_per_px ** 2
                mango_area_mm2 = mango_area_px * mm_per_px_squared
                lesion_area_mm2 = lesion_area_px * mm_per_px_squared
                lesion_percent = (lesion_area_px / mango_area_px * 100) if mango_area_px > 0 else 0
                if mango_area_mm2 < 0 or lesion_area_mm2 < 0:
                    st.error("Invalid area calculations. Please try again.")
                    st.stop()
            except Exception as e:
                st.error(f"Error in area calculation: {str(e)}")
                st.stop()

            st.markdown("### 🟩 Selected Mango & Lesions")
            col1, col2 = st.columns(2)
            col1.image(mango_mask.astype(np.uint8), caption="Total Mango Area", use_column_width=True)
            col2.image(lesion_mask.astype(np.uint8), caption="Lesion Area", use_column_width=True)
            result = {
                "Sample #": len(st.session_state.samples) + 1,
                "Total Area (mm²)": round(mango_area_mm2, 2),
                "Lesion Area (mm²)": round(lesion_area_mm2, 2),
                "Lesion %": round(lesion_percent, 2)
            }
            st.markdown("### 📊 Current Sample Result")
            st.dataframe(pd.DataFrame([result]))
            if st.button("Add this mango as a sample"):
                try:
                    st.session_state.samples.append(result)
                    st.success(f"Sample {result['Sample #']} added! Draw the next mango.")
                except Exception as e:
                    st.error(f"Error adding sample: {str(e)}")

        if st.session_state.samples:
            # Simplified samples dataframe
            all_samples_df = pd.DataFrame(st.session_state.samples)
            st.markdown("### 🥭 All Mango Samples")
            st.dataframe(all_samples_df, use_container_width=True)

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
            
            # Use a try/except block to handle CSV export errors
            try:
                # Simplified CSV creation
                csv = all_samples_df.to_csv(index=False).encode()
                st.download_button(
                    "📥 Download All Samples as CSV",
                    csv,
                    f"{csv_filename}.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"Error preparing CSV: {str(e)}")
else:
    # Show helpful guidance when no file is uploaded
    st.info("👆 Upload an image to get started with mango measurements")

# --- Footer ---
st.markdown("""
---
<center>
Plant Pathology Laboratory, Institute of Plant Breeding, CAFS, UPLB;  
Contact: <a href="mailto:jsmendoza5@up.edu.ph">jsmendoza5@up.edu.ph</a>
</center>
""", unsafe_allow_html=True)