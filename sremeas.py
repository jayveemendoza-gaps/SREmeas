import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import gc
from io import BytesIO
import time

# Cloud-optimized page config
st.set_page_config(
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_title="Mango Lesion Analyzer",
    page_icon="🥭"
)

st.title("🥭 Mango Lesion Analyzer")

# Optimized constants
MAX_IMAGE_SIZE = 320
CANVAS_SIZE = 900
MAX_FILE_SIZE_MB = 8
MEMORY_CLEANUP_INTERVAL = 60
MAX_SAMPLES = 25

# Initialize session state efficiently
session_defaults = {
    "samples": [],
    "mm_per_px": None,
    "polygon_drawing": False,
    "last_cleanup": time.time(),
    "last_cache_clear": 0
}

for key, default in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Periodic cleanup
current_time = time.time()
if current_time - st.session_state.get('last_cleanup', 0) > 60:
    gc.collect()
    st.session_state.last_cleanup = current_time

def aggressive_cleanup():
    """Memory cleanup for cloud stability"""
    gc.collect()
    
    # Clear specific cache keys
    cleanup_keys = ['temp_canvas_data', 'temp_masks', 'large_arrays', 'canvas_cache', 'temp_images']
    for key in cleanup_keys:
        if key in st.session_state:
            try:
                del st.session_state[key]
            except Exception:
                pass
    
    st.session_state.last_cleanup = time.time()
    
    # Periodic cache clearing
    try:
        if (time.time() - st.session_state.get('last_cache_clear', 0)) > MEMORY_CLEANUP_INTERVAL:
            st.cache_data.clear()
            st.session_state.last_cache_clear = time.time()
    except Exception:
        pass

def check_memory_limit():
    """Check memory usage and limits"""
    if len(st.session_state.samples) > MAX_SAMPLES:
        st.error(f"💾 Maximum samples reached ({MAX_SAMPLES}). Please download and clear samples.")
        return True
    
    if len(st.session_state.samples) > 12:
        st.warning("⚠️ Many samples stored. Consider downloading results soon.")
    
    return False

def emergency_reset():
    """Emergency session state reset"""
    try:
        # Clear problematic keys
        problem_keys = ['polygon_drawing', 'temp_canvas_data', 'temp_masks', 'large_arrays', 'temp_images']
        for key in problem_keys:
            if key in st.session_state:
                try:
                    del st.session_state[key]
                except Exception:
                    pass
        
        # Clear canvas-related keys
        canvas_keys = [k for k in st.session_state.keys() if 'canvas' in k.lower()]
        for key in canvas_keys:
            if key in st.session_state:
                try:
                    del st.session_state[key]
                except Exception:
                    pass
        
        aggressive_cleanup()
        return True
    except Exception:
        return False

@st.cache_data(max_entries=1, ttl=90, show_spinner=False)
def process_uploaded_image(uploaded_file, max_dim=MAX_IMAGE_SIZE):
    """Optimized image processing"""
    if not uploaded_file or len(uploaded_file) == 0:
        return None, None, None
    
    file_size_mb = len(uploaded_file) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large: {file_size_mb:.1f}MB. Use files under {MAX_FILE_SIZE_MB}MB.")
        return None, None, None
    
    try:
        image = Image.open(BytesIO(uploaded_file))
        original_size = image.size
        
        if original_size[0] * original_size[1] == 0:
            st.error("❌ Invalid image dimensions")
            return None, None, None
        
        # Optimize for large images
        if original_size[0] * original_size[1] > 1200000:
            st.warning("Large image detected. Optimizing for cloud speed.")
            max_dim = min(max_dim, 260)
        
        if image.mode != 'RGB':
            image = image.convert("RGB")
        
        scale = min(max_dim / image.height, max_dim / image.width, 1.0)
        if scale < 1.0:
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)
            st.info(f"Resized: {original_size[0]}x{original_size[1]} → {new_size[0]}x{new_size[1]}")
        
        image_np = np.array(image, dtype=np.uint8)
        return image_np, original_size, scale
        
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        return None, None, None

def quick_color_analysis(image_np, mask, mm_per_px):
    """Optimized color analysis with preserved masking formula"""
    if image_np is None or mask is None or mm_per_px is None:
        return 0, 0, 0, None, None
    
    if mask.size == 0 or np.max(mask) == 0:
        return 0, 0, 0, None, None
    
    if mask.shape != image_np.shape[:2]:
        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # HSV conversion
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    # Healthy mango detection
    mango_mask1 = cv2.inRange(hsv, (20, 40, 60), (85, 255, 255))
    mango_mask2 = cv2.inRange(hsv, (15, 25, 40), (80, 200, 200))
    mango_mask3 = cv2.inRange(hsv, (25, 20, 100), (75, 150, 255))
    healthy_mask = cv2.bitwise_or(mango_mask1, cv2.bitwise_or(mango_mask2, mango_mask3))
    
    # Lesion detection
    lesion_mask1 = cv2.inRange(hsv, (8, 50, 30), (25, 255, 140))
    lesion_mask2 = cv2.inRange(hsv, (0, 60, 25), (15, 255, 120))
    lesion_mask3 = cv2.inRange(hsv, (0, 0, 0), (180, 255, 45))
    lesion_mask4 = cv2.inRange(hsv, (0, 10, 20), (180, 80, 100))
    raw_lesion_mask = cv2.bitwise_or(cv2.bitwise_or(lesion_mask1, lesion_mask2), 
                                     cv2.bitwise_or(lesion_mask3, lesion_mask4))
    
    # Apply user mask
    healthy_mask = cv2.bitwise_and(healthy_mask, mask)
    raw_lesion_mask = cv2.bitwise_and(raw_lesion_mask, mask)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_lesion_mask = cv2.morphologyEx(raw_lesion_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_lesion_mask = cv2.morphologyEx(cleaned_lesion_mask, cv2.MORPH_OPEN, kernel)
    
    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_lesion_mask, connectivity=8)
    final_lesion_mask = np.zeros_like(cleaned_lesion_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 3:
            final_lesion_mask[labels == i] = 255
    
    total_mango_mask = cv2.bitwise_or(healthy_mask, final_lesion_mask)
    
    # Refined morphological operations
    base_kernel_size = max(2, int(np.sqrt(np.count_nonzero(mask)) / 50))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_kernel_size, base_kernel_size))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_kernel_size + 1, base_kernel_size + 1))
    
    final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_CLOSE, kernel_medium)
    final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_OPEN, kernel_small)
    final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_CLOSE, kernel_small)
    final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_OPEN, kernel_small)
    
    # Calculate areas
    mango_area_px = np.count_nonzero(total_mango_mask)
    lesion_area_px = np.count_nonzero(final_lesion_mask)
    
    if mango_area_px == 0:
        return 0, 0, 0, None, None
    
    mm_per_px_sq = mm_per_px * mm_per_px
    mango_area_mm2 = mango_area_px * mm_per_px_sq
    lesion_area_mm2 = lesion_area_px * mm_per_px_sq
    lesion_percent = (lesion_area_mm2 / mango_area_mm2 * 100) if mango_area_mm2 > 0 else 0
    
    return mango_area_mm2, lesion_area_mm2, lesion_percent, total_mango_mask, final_lesion_mask

def safe_rerun():
    """Safe app rerun with fallbacks"""
    aggressive_cleanup()
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            # Safe session state cleanup
            keys_to_preserve = ['samples', 'mm_per_px', 'last_cleanup']
            keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_preserve]
            
            for key in keys_to_delete:
                if key in st.session_state:
                    try:
                        del st.session_state[key]
                    except Exception:
                        pass
            
            st.info("🔄 App optimized. Continue with your analysis.")

def safe_add_sample(result):
    """Safely add sample with validation"""
    try:
        if 'samples' not in st.session_state:
            st.session_state.samples = []
        
        if len(st.session_state.samples) >= MAX_SAMPLES:
            st.warning(f"⚠️ Maximum samples reached ({MAX_SAMPLES}). Clear some samples before adding more.")
            return False
        
        if not result or not isinstance(result, dict):
            st.error("❌ Invalid result data")
            return False
        
        required_keys = ["Sample", "Area (mm²)", "Lesions (mm²)", "Lesion %"]
        if not all(key in result for key in required_keys):
            st.error("❌ Missing required result fields")
            return False
        
        # Validate numeric values
        try:
            area = float(result["Area (mm²)"])
            lesions = float(result["Lesions (mm²)"])
            percent = float(result["Lesion %"])
            
            if area < 0 or lesions < 0 or percent < 0:
                st.error("❌ Invalid negative values")
                return False
        except (ValueError, TypeError):
            st.error("❌ Invalid numeric values")
            return False
        
        st.session_state.samples.append(result)
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to add sample: {str(e)}")
        return False

# File uploader
uploaded_file = st.file_uploader(
    "Upload mango image (JPG/PNG)", 
    type=["png", "jpg", "jpeg"],
    help=f"Max size: {MAX_FILE_SIZE_MB}MB"
)

if uploaded_file:
    try:
        # Memory check
        if check_memory_limit():
            st.error("❌ Memory limit reached. Please try a smaller image or clear samples.")
            st.stop()
        
        # File size validation
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size > MAX_FILE_SIZE_MB:
            st.error(f"📁 File too large: {file_size:.1f}MB. Maximum allowed: {MAX_FILE_SIZE_MB}MB")
            st.stop()
        
        # File size feedback
        if file_size > 4:
            st.warning(f"⚠️ Large file: {file_size:.1f}MB - expect slower cloud processing")
        elif file_size > 2:
            st.info(f"📏 File size: {file_size:.1f}MB")
        else:
            st.success(f"✅ File size: {file_size:.1f}MB")
        
        # Process image
        with st.spinner("🔄 Processing image..."):
            image_np, original_size, scale = process_uploaded_image(uploaded_file.getvalue())
        
        if image_np is None:
            st.error("❌ Failed to process image. Try a smaller file or different format.")
            st.stop()
        
        # Store original for analysis
        original_image_np = image_np.copy()
        h, w = image_np.shape[:2]
        st.success(f"✅ Image loaded: {w}x{h} pixels (scale: {scale:.2f})")
        
        # Calculate display size
        display_scale = min(CANVAS_SIZE / w, CANVAS_SIZE / h, 1.0)
        display_w = int(w * display_scale)
        display_h = int(h * display_scale)
        
        # Ensure minimum usable size
        MIN_CANVAS_SIZE = 300
        if display_w < MIN_CANVAS_SIZE or display_h < MIN_CANVAS_SIZE:
            min_scale = max(MIN_CANVAS_SIZE / w, MIN_CANVAS_SIZE / h)
            if min_scale <= 3.0:
                display_w = int(w * min_scale)
                display_h = int(h * min_scale)
                display_scale = min_scale
                st.info(f"🔧 Canvas enlarged for better usability: {display_w}x{display_h}")
        
        # Cloud optimization limits
        MAX_DISPLAY_DIM = 2200
        if display_w > MAX_DISPLAY_DIM or display_h > MAX_DISPLAY_DIM:
            scale_factor = min(MAX_DISPLAY_DIM / display_w, MAX_DISPLAY_DIM / display_h)
            display_w = int(display_w * scale_factor)
            display_h = int(display_h * scale_factor)
            display_scale *= scale_factor
        
        # Create display image
        display_image = Image.fromarray(image_np)
        if display_scale != 1.0:
            display_image = display_image.resize((display_w, display_h), Image.LANCZOS)
        
        # Step 1: Scale Setting
        st.markdown("## 1️⃣ Set Scale")
        st.info("📏 Draw a line on a known measurement (ruler/scale bar)")
        st.success(f"✅ Canvas size: {display_w}x{display_h} pixels")
        
        scale_canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=3,
            stroke_color="rgba(255,0,0,1)",
            background_image=display_image,
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode="line",
            key="scale_canvas",
        )
        
        scale_length_mm = st.number_input(
            "Real length of drawn line (mm):",
            min_value=0.1,
            value=10.0,
            step=1.0
        )
        
        # Calculate scale
        scale_px = None
        if scale_canvas and hasattr(scale_canvas, 'json_data') and scale_canvas.json_data:
            objects = scale_canvas.json_data.get("objects", [])
            if objects and isinstance(objects, list):
                obj = objects[-1]
                if obj and isinstance(obj, dict) and obj.get("type") == "line":
                    try:
                        x1, y1 = float(obj.get("x1", 0)), float(obj.get("y1", 0))
                        x2, y2 = float(obj.get("x2", 0)), float(obj.get("y2", 0))
                        scale_px = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        
                        if scale_px > 0:
                            if display_scale != 1.0:
                                scale_px /= display_scale
                            st.success(f"📏 Line drawn: {scale_px:.1f} pixels")
                        else:
                            scale_px = None
                    except (KeyError, ValueError, TypeError):
                        scale_px = None
        
        if scale_px and scale_length_mm > 0:
            st.session_state.mm_per_px = scale_length_mm / scale_px
            st.success(f"✅ Scale set: {st.session_state.mm_per_px:.4f} mm/pixel")
            
            # Step 2: Mango Analysis
            st.markdown("## 2️⃣ Analyze Mango")
            st.info("🥭 Draw around one mango at a time for accurate analysis")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                drawing_mode = st.radio(
                    "Drawing mode:",
                    ["circle", "rect", "polygon", "transform"],
                    horizontal=True
                )
            
            with col2:
                brightness = st.slider("🔆 Brightness:", 0.7, 1.5, 1.0, 0.1)
            
            # Apply brightness adjustment
            if brightness != 1.0:
                try:
                    display_array = np.array(display_image, dtype=np.float32)
                    display_array = np.clip(display_array * brightness, 0, 255).astype(np.uint8)
                    adjusted_display_image = Image.fromarray(display_array)
                except Exception:
                    adjusted_display_image = display_image
            else:
                adjusted_display_image = display_image
            
            # Mode instructions
            mode_instructions = {
                "transform": "🔄 **Transform Mode**: Click and drag shapes to move, drag corners/edges to resize.",
                "polygon": "📐 **Polygon Mode**: Click to place points, double-click to close polygon.",
                "circle": "✏️ **Circle Mode**: Draw a new circle around the mango.",
                "rect": "✏️ **Rectangle Mode**: Draw a new rectangle around the mango."
            }
            st.info(mode_instructions.get(drawing_mode, ""))
            
            # Main analysis canvas
            canvas_result = st_canvas(
                fill_color="rgba(255,165,0,0.2)",
                stroke_width=3,
                stroke_color="rgba(255,165,0,1)",
                background_image=adjusted_display_image,
                update_streamlit=True,
                height=display_h,
                width=display_w,
                drawing_mode=drawing_mode,
                key="mango_canvas_persistent",
            )
            
            # Process analysis
            process_analysis = False
            if (canvas_result and hasattr(canvas_result, 'image_data') and 
                canvas_result.image_data is not None):
                
                if (len(canvas_result.image_data.shape) >= 3 and 
                    canvas_result.image_data.shape[2] >= 4):
                    
                    alpha_channel = canvas_result.image_data[:,:,3]
                    has_drawing = np.any(alpha_channel > 0)
                    
                    if has_drawing:
                        if drawing_mode == "polygon":
                            if (canvas_result.json_data and 
                                canvas_result.json_data.get("objects")):
                                objects = canvas_result.json_data["objects"]
                                polygon_objects = [obj for obj in objects if obj and obj.get("type") == "polygon"]
                                
                                if polygon_objects:
                                    if not st.session_state.get("polygon_drawing", False):
                                        st.success("✅ Polygon completed! Computing analysis...")
                                        st.session_state.polygon_drawing = True
                                    process_analysis = True
                                else:
                                    st.info("🔄 Drawing polygon... Double-click to close")
                                    st.session_state.polygon_drawing = False
                        else:
                            process_analysis = True
            
            if process_analysis:
                try:
                    # Create mask
                    image_data = canvas_result.image_data
                    alpha_channel = image_data[:,:,3]
                    mask = (alpha_channel > 0).astype(np.uint8) * 255
                    
                    if np.max(mask) == 0:
                        st.warning("⚠️ No selection detected. Please draw on the image.")
                        st.stop()
                    
                    # Scale mask to match processing image
                    if display_scale != 1.0:
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Analyze
                    with st.spinner("🔬 Analyzing mango..."):
                        mango_area_mm2, lesion_area_mm2, lesion_percent, total_mask, lesion_mask = quick_color_analysis(
                            original_image_np, mask, st.session_state.mm_per_px
                        )
                    
                    if mango_area_mm2 > 0:
                        # Display results
                        st.markdown("### 📊 Analysis Results")
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            st.image(total_mask, caption="🥭 Total Mango Area", width=180)
                        with col2:
                            st.image(lesion_mask, caption="🔴 Lesion Areas", width=180)
                        with col3:
                            st.metric("Total Area", f"{mango_area_mm2:.1f} mm²")
                            st.metric("Lesion Area", f"{lesion_area_mm2:.1f} mm²")
                            st.metric("Lesion %", f"{lesion_percent:.1f}%")
                        
                        # Results table
                        result = {
                            "Sample": len(st.session_state.samples) + 1,
                            "Area (mm²)": round(mango_area_mm2, 1),
                            "Lesions (mm²)": round(lesion_area_mm2, 1),
                            "Lesion %": round(lesion_percent, 1)
                        }
                        
                        st.dataframe(pd.DataFrame([result]), use_container_width=True)
                        
                        # Manual correction tools
                        st.markdown("### ✏️ Manual Corrections")
                        st.info("Use colored pens to correct misclassified areas")
                        
                        correction_col1, correction_col2 = st.columns([1, 1])
                        
                        with correction_col1:
                            correction_mode = st.radio(
                                "Correction Mode:",
                                ["🟡 Yellow Pen (Mark as Healthy)", "⚫ Black Pen (Mark as Lesion)"],
                                key="correction_mode"
                            )
                        
                        with correction_col2:
                            pen_size = st.slider("Pen Size:", 2, 15, 5, key="pen_size")
                            if st.button("🔄 Recalculate"):
                                st.session_state.apply_corrections = True
                        
                        # Set stroke color
                        stroke_color = "rgba(255, 255, 0, 1)" if "Yellow" in correction_mode else "rgba(0, 0, 0, 1)"
                        
                        # Create overlay image
                        try:
                            overlay_image = np.array(adjusted_display_image)
                            
                            if display_scale != 1.0:
                                display_total_mask = cv2.resize(total_mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
                                display_lesion_mask = cv2.resize(lesion_mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
                            else:
                                display_total_mask = total_mask
                                display_lesion_mask = lesion_mask

                            # Add colored overlays
                            healthy_overlay = display_total_mask > 0
                            lesion_overlay = display_lesion_mask > 0

                            overlay_image[healthy_overlay & ~lesion_overlay] = overlay_image[healthy_overlay & ~lesion_overlay] * 0.7 + np.array([0, 100, 0]) * 0.3
                            overlay_image[lesion_overlay] = overlay_image[lesion_overlay] * 0.7 + np.array([100, 0, 0]) * 0.3

                            overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)
                            overlay_pil = Image.fromarray(overlay_image)
                        except Exception:
                            overlay_pil = adjusted_display_image

                        # Correction canvas
                        correction_canvas_key = f"correction_canvas_{correction_mode}_{pen_size}"
                        correction_canvas = st_canvas(
                            fill_color="rgba(0,0,0,0)",
                            stroke_width=pen_size,
                            stroke_color=stroke_color,
                            background_image=overlay_pil,
                            update_streamlit=True,
                            height=display_h,
                            width=display_w,
                            drawing_mode="freedraw",
                            key=correction_canvas_key,
                        )

                        st.caption("🟢 Green tint = Healthy areas | 🔴 Red tint = Detected lesions")
                        
                        # Apply corrections
                        if st.session_state.get('apply_corrections', False):
                            st.session_state.apply_corrections = False
                            
                            try:
                                if correction_canvas.image_data is not None:
                                    correction_data = correction_canvas.image_data
                                    
                                    # Extract correction masks
                                    yellow_correction = np.zeros((correction_data.shape[0], correction_data.shape[1]), dtype=np.uint8)
                                    black_correction = np.zeros((correction_data.shape[0], correction_data.shape[1]), dtype=np.uint8)
                                    
                                    # Detect strokes
                                    yellow_pixels = ((correction_data[:,:,0] > 200) & 
                                                   (correction_data[:,:,1] > 200) & 
                                                   (correction_data[:,:,2] < 100) & 
                                                   (correction_data[:,:,3] > 0))
                                    yellow_correction[yellow_pixels] = 255
                                    
                                    black_pixels = ((correction_data[:,:,0] < 50) & 
                                                  (correction_data[:,:,1] < 50) & 
                                                  (correction_data[:,:,2] < 50) & 
                                                  (correction_data[:,:,3] > 0))
                                    black_correction[black_pixels] = 255
                                    
                                    # Scale corrections
                                    if display_scale != 1.0:
                                        yellow_correction = cv2.resize(yellow_correction, (w, h), interpolation=cv2.INTER_NEAREST)
                                        black_correction = cv2.resize(black_correction, (w, h), interpolation=cv2.INTER_NEAREST)
                                    
                                    # Apply corrections
                                    corrected_lesion_mask = lesion_mask.copy()
                                    corrected_total_mask = total_mask.copy()
                                    
                                    # Yellow pen: remove from lesions, add to healthy
                                    corrected_lesion_mask[yellow_correction > 0] = 0
                                    corrected_total_mask[yellow_correction > 0] = 255
                                    
                                    # Black pen: add to lesions
                                    corrected_lesion_mask[black_correction > 0] = 255
                                    
                                    # Recalculate areas
                                    corrected_mango_area_px = np.count_nonzero(corrected_total_mask)
                                    corrected_lesion_area_px = np.count_nonzero(corrected_lesion_mask)
                                    
                                    if corrected_mango_area_px > 0:
                                        mm_per_px_sq = st.session_state.mm_per_px * st.session_state.mm_per_px
                                        corrected_mango_area_mm2 = corrected_mango_area_px * mm_per_px_sq
                                        corrected_lesion_area_mm2 = corrected_lesion_area_px * mm_per_px_sq
                                        corrected_lesion_percent = (corrected_lesion_area_mm2 / corrected_mango_area_mm2 * 100)
                                        
                                        # Display corrected results
                                        st.markdown("### 📊 Corrected Results")
                                        corr_col1, corr_col2, corr_col3 = st.columns([1, 1, 1])
                                        
                                        with corr_col1:
                                            st.image(corrected_total_mask, caption="🥭 Corrected Total Area", width=180)
                                        with corr_col2:
                                            st.image(corrected_lesion_mask, caption="🔴 Corrected Lesions", width=180)
                                        with corr_col3:
                                            area_change = corrected_mango_area_mm2 - mango_area_mm2
                                            lesion_change = corrected_lesion_percent - lesion_percent
                                            
                                            st.metric("Corrected Area", f"{corrected_mango_area_mm2:.1f} mm²", delta=f"{area_change:+.1f} mm²")
                                            st.metric("Corrected Lesions", f"{corrected_lesion_area_mm2:.1f} mm²")
                                            st.metric("Corrected %", f"{corrected_lesion_percent:.1f}%", delta=f"{lesion_change:+.1f}%")
                                        
                                        # Update result
                                        result = {
                                            "Sample": len(st.session_state.samples) + 1,
                                            "Area (mm²)": round(corrected_mango_area_mm2, 1),
                                            "Lesions (mm²)": round(corrected_lesion_area_mm2, 1),
                                            "Lesion %": round(corrected_lesion_percent, 1)
                                        }
                                        
                                        st.dataframe(pd.DataFrame([result]), use_container_width=True)
                                        st.success("✅ Corrections applied!")
                            except Exception as e:
                                st.error(f"❌ Correction error: {str(e)}")
                        
                        if st.button("✅ Add Sample", type="primary"):
                            if safe_add_sample(result):
                                st.success(f"✅ Sample {len(st.session_state.samples)} added successfully!")
                                aggressive_cleanup()
                                safe_rerun()
                    else:
                        st.warning("⚠️ No mango detected. Try adjusting your selection or brightness.")
                        
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")
                    aggressive_cleanup()
            
            # Sample management
            if st.session_state.samples:
                st.markdown("### 📊 All Samples")
                df = pd.DataFrame(st.session_state.samples)
                st.dataframe(df, use_container_width=True)
                
                # Summary statistics
                if len(st.session_state.samples) > 1:
                    avg_lesion = df["Lesion %"].mean()
                    max_lesion = df["Lesion %"].max()
                    min_lesion = df["Lesion %"].min()
                    
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    with summary_col1:
                        st.metric("Average Lesion %", f"{avg_lesion:.1f}%")
                    with summary_col2:
                        st.metric("Maximum Lesion %", f"{max_lesion:.1f}%")
                    with summary_col3:
                        st.metric("Minimum Lesion %", f"{min_lesion:.1f}%")
                
                # Management controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🗑️ Clear All", use_container_width=True):
                        st.session_state.samples = []
                        aggressive_cleanup()
                        st.success("✅ All samples cleared!")
                        safe_rerun()
                        
                with col2:
                    if st.session_state.samples:
                        sample_to_delete = st.selectbox(
                            "🗑️ Delete specific sample:",
                            options=[f"Sample {i}" for i in range(1, len(st.session_state.samples) + 1)],
                            key="delete_sample_select"
                        )
                        if st.button("Delete Selected", use_container_width=True):
                            sample_idx = int(sample_to_delete.split()[1]) - 1
                            st.session_state.samples.pop(sample_idx)
                            # Renumber remaining samples
                            for i, sample in enumerate(st.session_state.samples):
                                sample["Sample"] = i + 1
                            st.success(f"✅ Deleted: Sample {sample_idx + 1}")
                            aggressive_cleanup()
                            safe_rerun()
                
                with col3:
                    custom_filename = st.text_input(
                        "📁 Filename:",
                        value="mango_analysis",
                        help="Enter filename for CSV export",
                        key="custom_filename"
                    )
                    
                    if custom_filename:
                        safe_filename = "".join(c for c in custom_filename if c.isalnum() or c in "._-")
                        if not safe_filename:
                            safe_filename = "mango_analysis"
                        
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            f"{safe_filename}.csv",
                            "text/csv",
                            use_container_width=True
                        )
        else:
            st.info("👆 **Step 1:** Draw a line on the scale bar and set its real length to continue")
            
    except MemoryError:
        aggressive_cleanup()
        st.error("❌ Cloud memory limit reached!")
        if st.button("🧹 Clear Memory"):
            keys_to_preserve = ['samples']
            keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_preserve]
            
            for key in keys_to_delete:
                if key in st.session_state:
                    try:
                        del st.session_state[key]
                    except Exception:
                        pass
            
            aggressive_cleanup()
            safe_rerun()

    except Exception as e:
        aggressive_cleanup()
        st.error(f"❌ Processing error: {str(e)}")
        if st.button("🔄 Restart Application"):
            safe_rerun()

# Sidebar controls
with st.sidebar:
    st.markdown("### 🔧 Quick Actions")
    if st.button("🔄 Reset App"):
        if emergency_reset():
            try:
                st.rerun()
            except Exception:
                st.info("App reset. Please refresh page if needed.")
    
    if st.button("🧹 Clear Memory"):
        aggressive_cleanup()
        st.success("✅ Memory cleared")
