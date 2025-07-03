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

# Enhanced cache for image loading with better parameters
@st.cache_data(ttl=3600, max_entries=2)
def load_image(file_bytes):
    """Cache image loading to avoid repeated processing of the same image."""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

# Optimized display image creation for canvases
@st.cache_data
def create_display_image(image, target_width=800):
    """Create a properly sized display image for canvas backgrounds."""
    display_img, width, height = resize_with_aspect_ratio(image, target_width)
    return display_img, width, height

@st.cache_data
def convert_to_hsv(_image_np):
    """Cache HSV conversion to avoid recomputation. Use underscore to ignore hash."""
    return cv2.cvtColor(_image_np, cv2.COLOR_RGB2HSV)

@st.cache_data
def apply_color_mask(_hsv_img, _lower, _upper, _user_mask=None):
    """Cache mask application to avoid recomputation."""
    mask = cv2.inRange(_hsv_img, _lower, _upper)
    if _user_mask is not None:
        mask = (mask == 255) & _user_mask
    return mask

# Add this new cached function for efficient mask calculations
@st.cache_data
def calculate_areas(_mask1, _mask2):
    """Cache area calculations for better performance."""
    combined = _mask1 | _mask2
    return np.sum(combined), np.sum(_mask2)

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

# Add proper file upload guidance
uploaded_file = st.file_uploader(
    "Upload an image of mangoes (top view)", 
    type=["png", "jpg", "jpeg"],
    help="For best results, use a clear image with good lighting"
)

if uploaded_file:
    # Initialize session state at the very beginning
    if "samples" not in st.session_state:
        st.session_state.samples = []
    
    # Optimize image loading with better error handling
    try:
        with st.spinner("Loading image..."):
            # Cache the file content to avoid rereading on rerun
            file_bytes = uploaded_file.getvalue()
            image = load_image(file_bytes)
            image_np = np.array(image, dtype=np.uint8)  # Specify dtype for better memory usage
            h, w = image_np.shape[:2]
            
            # More aggressive handling for extremely large images
            if h * w > 8000 * 8000:
                st.warning("Image is extremely large and will be automatically resized.")
                image, new_width, new_height = resize_with_aspect_ratio(image, target_width=2000)
                image_np = np.array(image, dtype=np.uint8)
                h, w = new_height, new_width
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        st.stop()

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
    
    # Optimize display image creation for scale canvas
    try:
        with st.spinner("Preparing canvas..."):
            # Use the cached function to create display image
            display_image, display_width, display_height = create_display_image(image)
    except Exception as e:
        st.error(f"Error preparing image for display: {str(e)}")
        st.info("Try using a different image format or a smaller image.")
        st.stop()
    
    # Use st.container to reduce redraws and improve canvas performance
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
        value=10.0  # Set default value to 10 mm
    )
    scale_px = None

    if scale_canvas.json_data and len(scale_canvas.json_data["objects"]) > 0:
        obj = scale_canvas.json_data["objects"][-1]
        if obj["type"] == "line":
            x0, y0 = obj["x1"], obj["y1"]
            x1, y1 = obj["x2"], obj["y2"]
            # Use faster math operations
            dx, dy = x1 - x0, y1 - y0
            scale_px = np.sqrt(dx*dx + dy*dy)
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

        # Only adjust brightness if needed (skip if factor is 1.0)
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            brightness_adjusted_image = enhancer.enhance(brightness_factor)
        else:
            brightness_adjusted_image = image
            
        brightness_adjusted_np = np.array(brightness_adjusted_image)
        
        # Use brightness-adjusted image for display only
        brightness_display_image, _, _ = resize_with_aspect_ratio(brightness_adjusted_image, target_width=800)

        # --- Step 3: Draw around a mango ---
        st.markdown("## 3️⃣ Draw around a mango (one at a time)")
        
        # Keep the drawing mode selection exactly as is to avoid canvas crashes
        drawing_mode = st.selectbox(
            "Drawing mode", 
            ["circle", "rect", "freedraw", "transform"], 
            index=0,
            help="Circle mode is recommended - easiest to position and resize. Use 'transform' mode to move/resize existing shapes."
        )

        # Remove the unnecessary columns and checkboxes entirely
        if drawing_mode == "transform":
            st.info("🔄 **Transform Mode**: Click and drag existing shapes to move them, or drag corners/edges to resize.")
        else:
            st.info(f"✏️ **{drawing_mode.title()} Mode**: Draw a new {drawing_mode}. Switch to 'transform' mode to move/resize existing shapes.")

        if drawing_mode == "freedraw":
            st.warning("Free draw mode may be slower. For best results, draw slowly and steadily.")

        # Use st.container to reduce redraws
        with st.container():
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

        if canvas_result.image_data is not None and np.any(canvas_result.image_data != 255):
            # Show a more informative processing indicator
            with st.spinner("Processing mango area... (this may take a moment on slow connections)"):
                # Optimize mask processing with numpy operations
                mask = canvas_result.image_data[:, :, 3]  # Use alpha channel directly
                user_mask = mask > 10
                
                # Early exit if no pixels selected
                if not np.any(user_mask):
                    st.warning("No area selected. Please draw on the mango.")
                    st.stop()
                
                # Optimize bounding box calculation
                y_indices, x_indices = np.nonzero(user_mask)
                if len(y_indices) == 0:
                    st.warning("No valid area selected.")
                    st.stop()
                
                # Use numpy min/max for faster bounds calculation
                y_min, y_max = np.min(y_indices), np.max(y_indices) + 1
                x_min, x_max = np.min(x_indices), np.max(x_indices) + 1
                
                # Safety bounds check
                y_min, y_max = max(0, y_min), min(brightness_adjusted_np.shape[0], y_max)
                x_min, x_max = max(0, x_min), min(brightness_adjusted_np.shape[1], x_max)
                
                # Validate bounding box
                if y_max <= y_min or x_max <= x_min:
                    st.error("Invalid selection area. Please try again.")
                    st.stop()
                
                # Crop image and mask to bounding box (huge speed boost)
                crop_image = brightness_adjusted_np[y_min:y_max, x_min:x_max]
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
                mango_area_px, lesion_area_px = calculate_areas(hsv_mask_crop, lesion_mask_crop)
                
                # Create full-size masks for display (only if needed)
                mango_mask = np.zeros_like(user_mask, dtype=np.uint8)
                lesion_mask = np.zeros_like(user_mask, dtype=np.uint8)
                
                # Get the actual dimensions of the slices
                slice_height, slice_width = y_max - y_min, x_max - x_min
                
                # Prepare combined mask and lesion mask
                combined_mask = (hsv_mask_crop | lesion_mask_crop).astype(np.uint8) * 255
                lesion_display_mask = lesion_mask_crop.astype(np.uint8) * 255
                
                # Verify dimensions before assignment (important to avoid broadcasting errors)
                if combined_mask.shape != (slice_height, slice_width):
                    # Resize if necessary
                    combined_mask = cv2.resize(combined_mask, (slice_width, slice_height), 
                                               interpolation=cv2.INTER_NEAREST)
                    
                if lesion_display_mask.shape != (slice_height, slice_width):
                    # Resize if necessary
                    lesion_display_mask = cv2.resize(lesion_display_mask, (slice_width, slice_height), 
                                                     interpolation=cv2.INTER_NEAREST)
                
                # Now safely assign to the full mask
                try:
                    mango_mask[y_min:y_max, x_min:x_max] = combined_mask
                    lesion_mask[y_min:y_max, x_min:x_max] = lesion_display_mask
                except ValueError as e:
                    # Fallback for broadcasting errors
                    st.warning("Adjusting mask dimensions...")
                    # Force dimensions to match exactly
                    target_height, target_width = y_max - y_min, x_max - x_min
                    combined_mask = cv2.resize(combined_mask, (target_width, target_height), 
                                            interpolation=cv2.INTER_NEAREST)
                    lesion_display_mask = cv2.resize(lesion_display_mask, (target_width, target_height),
                                                  interpolation=cv2.INTER_NEAREST)
                    mango_mask[y_min:y_max, x_min:x_max] = combined_mask
                    lesion_mask[y_min:y_max, x_min:x_max] = lesion_display_mask
                
            # Add a status indicator for better feedback
            st.success("Processing complete! Review the results below.")
            
            # Convert pixel areas to mm² with comprehensive safety checks
            try:
                if mm_per_px <= 0:
                    st.error("Invalid scale conversion. Please set the scale again.")
                    st.stop()
                
                if mango_area_px == 0:
                    st.warning("No mango area detected. Try adjusting the brightness or drawing a different area.")
                    st.stop()
                
                # Square the mm_per_px value once instead of in each calculation
                mm_per_px_squared = mm_per_px ** 2
                mango_area_mm2 = mango_area_px * mm_per_px_squared
                lesion_area_mm2 = lesion_area_px * mm_per_px_squared
                lesion_percent = (lesion_area_px / mango_area_px * 100) if mango_area_px > 0 else 0
                
                # Validate results
                if mango_area_mm2 < 0 or lesion_area_mm2 < 0:
                    st.error("Invalid area calculations. Please try again.")
                    st.stop()
                    
            except Exception as e:
                st.error(f"Error in area calculation: {str(e)}")
                st.stop()

            # Display results - use more efficient image display
            st.markdown("### 🟩 Selected Mango & Lesions")
            col1, col2 = st.columns(2)
            # Convert masks to uint8 before display (more efficient)
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
            # Compute dataframe once and cache it
            @st.cache_data
            def get_samples_df(samples):
                return pd.DataFrame(samples)
                
            all_samples_df = get_samples_df(st.session_state.samples)
            st.markdown("### 🥭 All Mango Samples")
            try:
                # Cache the dataframe creation to avoid redundant operations
                st.dataframe(all_samples_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying samples: {str(e)}")
                # Reset samples if corrupted
                st.session_state.samples = []
                st.experimental_rerun()

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
                            st.experimental_rerun()
                with col2:
                    if st.button("🗑️ Clear All Samples", key="clear_all"):
                        st.session_state.samples = []
                        st.success("All samples cleared!")
                        st.experimental_rerun()

            # --- CSV export with custom filename ---
            csv_filename = st.text_input(
                "Enter filename for CSV export (without .csv):",
                value="mango_lesion_samples"
            )
            
            # Use a try/except block to handle CSV export errors
            try:
                # Create CSV on-demand rather than keeping it in memory
                @st.cache_data
                def get_csv(_df):
                    return _df.to_csv(index=False).encode()
                    
                csv = get_csv(all_samples_df)
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