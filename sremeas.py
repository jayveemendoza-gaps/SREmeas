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

# Initialize session state efficiently with memory tracking
for key, default in [
    ("samples", []), 
    ("mm_per_px", None), 
    ("polygon_drawing", False),
    ("last_cleanup", time.time())
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Ultra-optimized settings for maximum cloud stability
MAX_IMAGE_SIZE = 350      # Further reduced for cloud stability
CANVAS_SIZE = 500         # Kept optimal size
MAX_FILE_SIZE_MB = 10     # Increased file size limit
MEMORY_CLEANUP_INTERVAL = 10  # More aggressive cleanup
MAX_SAMPLES = 30          # Prevent memory buildup

def aggressive_cleanup():
    """Aggressive memory cleanup for cloud stability"""
    gc.collect()
    try:
        cv2.setUseOptimized(True)
        # Clear specific large objects from session state
        for key in ['temp_canvas_data', 'temp_masks', 'large_arrays']:
            if key in st.session_state:
                del st.session_state[key]
    except Exception:
        pass
    st.session_state.last_cleanup = time.time()
    # More aggressive temp image cleanup
    if hasattr(st.session_state, 'temp_images'):
        if len(getattr(st.session_state, 'temp_images', [])) > 1:  # More aggressive
            del st.session_state.temp_images

def check_memory_limit():
    """Check if we're approaching memory limits using fallback method"""
    # More aggressive memory monitoring
    if len(st.session_state.samples) > MAX_SAMPLES:
        st.error(f"💾 Maximum samples reached ({MAX_SAMPLES}). Please download and clear samples.")
        return True
    if len(st.session_state.samples) > 15:
        st.warning("⚠️ Many samples stored. Consider downloading results soon.")
        return True
    if hasattr(st.session_state, 'temp_images') and len(getattr(st.session_state, 'temp_images', [])) > 2:
        st.warning("⚠️ Multiple images in memory. Consider clearing for better performance.")
        return True
    return False

def should_cleanup():
    """Check if cleanup is needed"""
    return time.time() - st.session_state.last_cleanup > MEMORY_CLEANUP_INTERVAL

@st.cache_data(max_entries=1, ttl=120, show_spinner=False)  # Reduced TTL for faster cleanup
def process_uploaded_image(uploaded_file, max_dim=MAX_IMAGE_SIZE):
    """Optimized image processing for cloud stability"""
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
        # More aggressive size limits for cloud
        if original_size[0] * original_size[1] > 1500000:  # Reduced threshold
            st.warning("Large image detected. Downscaling for cloud stability.")
            max_dim = min(max_dim, 280)  # More aggressive downscaling
        if image.mode != 'RGB':
            image = image.convert("RGB")
        scale = min(max_dim / image.height, max_dim / image.width, 1.0)
        if scale < 1.0:
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)
            st.info(f"Resized: {original_size[0]}x{original_size[1]} → {new_size[0]}x{new_size[1]}")
        image_np = np.array(image, dtype=np.uint8)
        del image  # Explicit cleanup
        aggressive_cleanup()
        return image_np, original_size, scale
    except Exception as e:
        aggressive_cleanup()
        st.error(f"Processing failed: {str(e)}")
        return None, None, None

def quick_color_analysis(image_np, mask, mm_per_px):
    """Color analysis with preserved masking formula and safe memory handling"""
    if image_np is None or mask is None or mm_per_px is None:
        return 0, 0, 0, None, None
    if mask.size == 0 or np.max(mask) == 0:
        return 0, 0, 0, None, None
    if mask.shape != image_np.shape[:2]:
        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    mango_mask1 = cv2.inRange(hsv, (20, 40, 60), (85, 255, 255))
    mango_mask2 = cv2.inRange(hsv, (15, 25, 40), (80, 200, 200))
    mango_mask3 = cv2.inRange(hsv, (25, 20, 100), (75, 150, 255))
    healthy_mask = cv2.bitwise_or(mango_mask1, cv2.bitwise_or(mango_mask2, mango_mask3))
    shadow_mask1 = cv2.inRange(hsv, (15, 20, 25), (85, 120, 110))
    shadow_mask2 = cv2.inRange(hsv, (10, 10, 15), (90, 60, 90))
    shadow_mask = cv2.bitwise_or(shadow_mask1, shadow_mask2)
    lesion_mask1 = cv2.inRange(hsv, (8, 50, 30), (25, 255, 140))
    lesion_mask2 = cv2.inRange(hsv, (0, 60, 25), (15, 255, 120))
    lesion_mask3 = cv2.inRange(hsv, (0, 0, 0), (180, 255, 45))
    lesion_mask4 = cv2.inRange(hsv, (0, 10, 20), (180, 80, 100))
    raw_lesion_mask = cv2.bitwise_or(lesion_mask1, lesion_mask2)
    raw_lesion_mask = cv2.bitwise_or(raw_lesion_mask, lesion_mask3)
    raw_lesion_mask = cv2.bitwise_or(raw_lesion_mask, lesion_mask4)
    healthy_mask = cv2.bitwise_and(healthy_mask, mask)
    raw_lesion_mask = cv2.bitwise_and(raw_lesion_mask, mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_lesion_mask = cv2.morphologyEx(raw_lesion_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_lesion_mask = cv2.morphologyEx(cleaned_lesion_mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_lesion_mask, connectivity=8)
    final_lesion_mask = np.zeros_like(cleaned_lesion_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 3:
            final_lesion_mask[labels == i] = 255
    total_mango_mask = cv2.bitwise_or(healthy_mask, final_lesion_mask)
    base_kernel_size = max(2, int(np.sqrt(np.count_nonzero(mask)) / 50))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_kernel_size, base_kernel_size))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_kernel_size + 1, base_kernel_size + 1))
    final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_CLOSE, kernel_medium)
    final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_OPEN, kernel_small)
    final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_CLOSE, kernel_small)
    final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_OPEN, kernel_small)
    mango_area_px = np.count_nonzero(total_mango_mask)
    lesion_area_px = np.count_nonzero(final_lesion_mask)
    if mango_area_px == 0:
        return 0, 0, 0, None, None
    mm_per_px_sq = mm_per_px * mm_per_px
    mango_area_mm2 = mango_area_px * mm_per_px_sq
    lesion_area_mm2 = lesion_area_px * mm_per_px_sq
    lesion_percent = (lesion_area_mm2 / mango_area_mm2 * 100) if mango_area_mm2 > 0 else 0
    return mango_area_mm2, lesion_area_mm2, lesion_percent, total_mango_mask, final_lesion_mask

# File uploader with enhanced cloud optimization messaging
uploaded_file = st.file_uploader(
    "Upload mango image (JPG/PNG)", 
    type=["png", "jpg", "jpeg"],
    help=f"Max size: {MAX_FILE_SIZE_MB}MB. For best cloud performance: Use JPG format, <3MB files recommended"
)

def safe_rerun():
    """Ultra-safe rerun function with multiple fallbacks"""
    aggressive_cleanup()
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            try:
                # Clear session state to force refresh
                for key in list(st.session_state.keys()):
                    if key not in ['samples', 'mm_per_px']:  # Keep important data
                        try:
                            del st.session_state[key]
                        except:
                            pass
                st.info("🔄 Page refreshed. Please continue with your analysis.")
            except Exception:
                st.write("🔄 Please refresh the page manually (F5 or Ctrl+R)")

def safe_add_sample(result):
    """Safely add sample with enhanced validation and memory limits"""
    try:
        if 'samples' not in st.session_state:
            st.session_state.samples = []
        
        # Prevent excessive samples that could cause memory issues on cloud
        if len(st.session_state.samples) >= MAX_SAMPLES:
            st.warning(f"⚠️ Maximum samples reached ({MAX_SAMPLES}). Clear some samples before adding more.")
            return False
        
        # Enhanced validation
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
        aggressive_cleanup()  # Cleanup after adding sample
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to add sample: {str(e)}")
        return False

if uploaded_file:
    try:
        # Periodic cleanup check with memory monitoring
        if should_cleanup():
            aggressive_cleanup()
        
        # Enhanced memory check before processing
        if check_memory_limit():
            st.error("❌ Memory limit reached. Please try a smaller image or clear samples.")
            st.stop()
        
        # File size validation with better messaging
        try:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        except Exception:
            st.error("❌ Cannot read file. Please try uploading again.")
            st.stop()
            
        if file_size > MAX_FILE_SIZE_MB:
            st.error(f"📁 File too large: {file_size:.1f}MB. Maximum allowed: {MAX_FILE_SIZE_MB}MB")
            st.info("💡 **Tips to reduce file size:**\n- Use JPG instead of PNG\n- Reduce image resolution\n- Compress image before upload")
            st.stop()
        
        if file_size > 5:
            st.warning(f"⚠️ Large file: {file_size:.1f}MB - processing may be slower on cloud")
        elif file_size > 3:
            st.info(f"📏 File size: {file_size:.1f}MB - good for cloud processing")
        else:
            st.success(f"✅ File size: {file_size:.1f}MB - optimal for cloud")
        
        # Process image with enhanced error handling and progress indication
        with st.spinner("🔄 Processing image... (this may take a moment on cloud)"):
            try:
                image_np, original_size, scale = process_uploaded_image(uploaded_file.getvalue())
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
                st.info("🔄 Please try again with a smaller image (< 5MB recommended)")
                aggressive_cleanup()
                st.stop()
        
        if image_np is None:
            st.error("❌ Failed to process image. Try a smaller file or different format.")
            st.info("💡 **Supported formats:** JPG (recommended), PNG\n**Recommended size:** < 5MB for best cloud performance")
            st.stop()
        
        h, w = image_np.shape[:2]
        st.success(f"✅ Image loaded: {w}x{h} pixels (scale: {scale:.2f})")
        
        # Calculate optimal display size maintaining aspect ratio with cloud limits
        display_scale = min(CANVAS_SIZE / w, CANVAS_SIZE / h, 1.0)
        display_w = int(w * display_scale)
        display_h = int(h * display_scale)
        
        # Ensure minimum usable size while maintaining aspect ratio
        if display_w < 400 or display_h < 300:  # Increased minimum for better usability
            min_scale = max(400 / w, 300 / h)
            if min_scale <= 2.0:  # Allow reasonable upscaling for small images
                display_w = int(w * min_scale)
                display_h = int(h * min_scale)
                display_scale = min_scale
        
        # Memory check before creating display image
        if display_w * display_h * 3 > 1500000:  # Reduced threshold for cloud stability
            st.warning("⚠️ Large display size. Using reduced canvas for cloud stability.")
            display_scale = min(display_scale, 0.7)  # More aggressive scaling
            display_w = int(w * display_scale)
            display_h = int(h * display_scale)
        
        # Create display image maintaining aspect ratio with error handling
        try:
            display_image = Image.fromarray(image_np)
            if display_scale != 1.0:
                display_image = display_image.resize((display_w, display_h), Image.LANCZOS)
        except Exception as e:
            st.error(f"❌ Display image creation failed: {str(e)}")
            aggressive_cleanup()
            st.stop()
        
        # --- Step 1: Scale Setting ---
        st.markdown("## 1️⃣ Set Scale")
        st.info("📏 Draw a line on a known measurement (ruler/scale bar)")
        
        # Enhanced scale canvas with better usability
        scale_canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=3,  # Increased for better visibility on larger canvas
            stroke_color="rgba(255,0,0,1)",  # Red color for better visibility
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
        
        # Safe scale calculation with error handling
        scale_px = None
        if scale_canvas.json_data and scale_canvas.json_data.get("objects"):
            try:
                obj = scale_canvas.json_data["objects"][-1]
                if obj.get("type") == "line":
                    dx = obj["x2"] - obj["x1"]
                    dy = obj["y2"] - obj["y1"]
                    scale_px = np.sqrt(dx*dx + dy*dy)
                    if display_scale != 1.0:
                        scale_px /= display_scale
                    st.success(f"📏 Line drawn: {scale_px:.1f} pixels")
            except Exception as e:
                st.warning("⚠️ Please draw a clear line on the scale")
        
        if scale_px and scale_length_mm > 0:
            st.session_state.mm_per_px = scale_length_mm / scale_px
            st.success(f"✅ Scale set: {st.session_state.mm_per_px:.4f} mm/pixel")
            
            # --- Step 2: Mango Analysis ---
            st.markdown("## 2️⃣ Analyze Mango")
            st.info("🥭 Draw around one mango at a time for accurate analysis")
            
            # Enhanced drawing mode selection
            col1, col2 = st.columns([2, 1])
            with col1:
                drawing_mode = st.radio(
                    "Drawing mode:",
                    ["circle", "rect", "polygon", "transform"],
                    horizontal=True,
                    help="Circle for round mangoes, polygon for irregular shapes, transform to adjust existing shapes"
                )
            
            with col2:
                # Brightness adjustment for better visibility
                brightness = st.slider(
                    "🔆 Brightness:",
                    min_value=0.7,
                    max_value=1.5,
                    value=1.0,
                    step=0.1,
                    help="Adjust brightness for better visibility"
                )
            
            # Apply brightness adjustment safely with memory management
            try:
                if brightness != 1.0:
                    # Process in chunks to avoid memory issues on cloud
                    display_image_array = np.array(display_image, dtype=np.float32)
                    
                    # Check array size before processing - more aggressive threshold
                    if display_image_array.size > 1000000:  # Reduced threshold
                        st.info("🔧 Applying brightness adjustment (large image)...")
                    
                    display_image_array = np.clip(display_image_array * brightness, 0, 255).astype(np.uint8)
                    adjusted_display_image = Image.fromarray(display_image_array)
                    
                    # Immediate cleanup
                    del display_image_array
                    aggressive_cleanup()
                else:
                    adjusted_display_image = display_image
            except MemoryError:
                st.warning("⚠️ Brightness adjustment skipped due to memory limits")
                adjusted_display_image = display_image
                aggressive_cleanup()
            except Exception as e:
                st.warning(f"⚠️ Brightness adjustment failed: {str(e)}")
                adjusted_display_image = display_image
            
            # Mode-specific instructions
            if drawing_mode == "transform":
                st.info("🔄 **Transform Mode**: Click and drag shapes to move, drag corners/edges to resize.")
            elif drawing_mode == "polygon":
                st.info("📐 **Polygon Mode**: Click to place points, double-click to close polygon.")
            else:
                st.info(f"✏️ **{drawing_mode.title()} Mode**: Draw a new {drawing_mode} around the mango.")
            
            # Enhanced canvas with better usability
            canvas_result = st_canvas(
                fill_color="rgba(255,165,0,0.2)",  # Slightly more visible fill
                stroke_width=3,  # Increased stroke width for better visibility
                stroke_color="rgba(255,165,0,1)",
                background_image=adjusted_display_image,
                update_streamlit=True,
                height=display_h,
                width=display_w,
                drawing_mode=drawing_mode,
                key="mango_canvas",
            )

            # Safe analysis processing with enhanced error handling
            process_analysis = False
            
            if canvas_result.image_data is not None and np.any(canvas_result.image_data[:,:,3] > 0):
                if drawing_mode == "polygon":
                    try:
                        if canvas_result.json_data and canvas_result.json_data.get("objects"):
                            polygon_objects = [obj for obj in canvas_result.json_data["objects"] if obj.get("type") == "polygon"]
                            path_objects = [obj for obj in canvas_result.json_data["objects"] if obj.get("type") == "path"]
                            
                            if polygon_objects:
                                if not st.session_state.polygon_drawing:
                                    st.success("✅ Polygon completed! Computing analysis...")
                                    st.session_state.polygon_drawing = True
                                process_analysis = True
                            elif path_objects:
                                st.info("🔄 Drawing polygon... Double-click to close")
                                st.session_state.polygon_drawing = False
                    except Exception:
                        st.warning("⚠️ Polygon drawing error. Please try again.")
                else:
                    process_analysis = True
            
            if process_analysis:
                try:
                    # Memory check before analysis
                    if check_memory_limit():
                        st.error("❌ Memory limit reached. Please try a smaller image or clear samples.")
                        st.stop()
                    
                    # Create mask safely
                    try:
                        mask = (canvas_result.image_data[:,:,3] > 0).astype(np.uint8) * 255
                        if mask.size == 0 or np.max(mask) == 0:
                            st.warning("⚠️ No selection detected. Please draw on the image.")
                            st.stop()
                    except Exception as e:
                        st.error(f"❌ Mask creation failed: {str(e)}")
                        st.stop()
                    
                    # Scale mask to match processing image safely
                    try:
                        if display_scale != 1.0:
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            if mask.size == 0:
                                st.error("❌ Mask scaling failed - empty result")
                                st.stop()
                    except Exception as e:
                        st.error(f"❌ Mask scaling error: {str(e)}")
                        aggressive_cleanup()
                        st.stop()
                    
                    # Analyze with enhanced error handling and progress
                    with st.spinner("🔬 Analyzing mango (cloud processing may take longer)..."):
                        try:
                            mango_area_mm2, lesion_area_mm2, lesion_percent, total_mask, lesion_mask = quick_color_analysis(
                                image_np, mask, st.session_state.mm_per_px
                            )
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
                            aggressive_cleanup()
                            st.stop()
                    
                    if mango_area_mm2 > 0:
                        # Display results with enhanced layout
                        st.markdown("### 📊 Analysis Results")
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            st.image(total_mask, caption="🥭 Total Mango Area", width=180)  # Larger for better visibility
                        with col2:
                            st.image(lesion_mask, caption="🔴 Lesion Areas", width=180)
                        with col3:
                            st.metric("Total Area", f"{mango_area_mm2:.1f} mm²")
                            st.metric("Lesion Area", f"{lesion_area_mm2:.1f} mm²")
                            st.metric("Lesion %", f"{lesion_percent:.1f}%")
                        
                        # Results table with better formatting
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
                        
                        # Correction mode selection
                        correction_col1, correction_col2 = st.columns([1, 1])
                        
                        with correction_col1:
                            correction_mode = st.radio(
                                "Correction Mode:",
                                ["🟡 Yellow Pen (Mark as Healthy)", "⚫ Black Pen (Mark as Lesion)", "🔄 Transform"],
                                key="correction_mode"
                            )
                        
                        with correction_col2:
                            pen_size = st.slider(
                                "Pen Size:",
                                min_value=2,
                                max_value=15,
                                value=5,
                                key="pen_size"
                            )
                            
                            if st.button("🔄 Recalculate", help="Apply corrections and recalculate"):
                                st.session_state.apply_corrections = True
                        
                        # Set up canvas for corrections
                        if correction_mode == "🟡 Yellow Pen (Mark as Healthy)":
                            stroke_color = "rgba(255, 255, 0, 1)"  # Yellow
                            canvas_mode = "freedraw"
                        elif correction_mode == "⚫ Black Pen (Mark as Lesion)":
                            stroke_color = "rgba(0, 0, 0, 1)"  # Black
                            canvas_mode = "freedraw"
                        else:
                            stroke_color = "rgba(255, 165, 0, 1)"  # Orange for transform
                            canvas_mode = "transform"
                        
                        # Create an overlay image showing current classification with memory safety
                        try:
                            overlay_image = np.array(adjusted_display_image)
                            if display_scale != 1.0:
                                display_total_mask = cv2.resize(total_mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
                                display_lesion_mask = cv2.resize(lesion_mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
                            else:
                                display_total_mask = total_mask
                                display_lesion_mask = lesion_mask
                            
                            # Memory check for overlay processing
                            if overlay_image.size > 1000000:  # Reduced threshold
                                st.info("🔧 Creating overlay (large image)...")
                            
                            # Add colored overlays safely
                            healthy_overlay = display_total_mask > 0
                            lesion_overlay = display_lesion_mask > 0
                            
                            overlay_image[healthy_overlay & ~lesion_overlay] = overlay_image[healthy_overlay & ~lesion_overlay] * 0.7 + np.array([0, 100, 0]) * 0.3
                            overlay_image[lesion_overlay] = overlay_image[lesion_overlay] * 0.7 + np.array([100, 0, 0]) * 0.3
                            
                            overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)
                            overlay_pil = Image.fromarray(overlay_image)
                            
                            # Cleanup intermediate arrays
                            del healthy_overlay, lesion_overlay
                            if display_scale != 1.0:
                                del display_total_mask, display_lesion_mask
                            gc.collect()
                        except Exception as e:
                            st.error(f"❌ Overlay creation failed: {str(e)}")
                            overlay_pil = adjusted_display_image  # Fallback to original image
                        
                        # Optionally, skip overlay creation for very large images or when not needed
                        overlay_skip_threshold = 800000  # Further reduced threshold for faster switching
                        create_overlay = (overlay_image.size <= overlay_skip_threshold) and (canvas_mode != "transform")
                        if not create_overlay:
                            overlay_pil = adjusted_display_image
                        else:
                            try:
                                overlay_image = np.array(adjusted_display_image)
                                if display_scale != 1.0:
                                    display_total_mask = cv2.resize(total_mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
                                    display_lesion_mask = cv2.resize(lesion_mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
                                else:
                                    display_total_mask = total_mask
                                    display_lesion_mask = lesion_mask

                                # Add colored overlays safely
                                healthy_overlay = display_total_mask > 0
                                lesion_overlay = display_lesion_mask > 0

                                overlay_image[healthy_overlay & ~lesion_overlay] = overlay_image[healthy_overlay & ~lesion_overlay] * 0.7 + np.array([0, 100, 0]) * 0.3
                                overlay_image[lesion_overlay] = overlay_image[lesion_overlay] * 0.7 + np.array([100, 0, 0]) * 0.3

                                overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)
                                overlay_pil = Image.fromarray(overlay_image)

                                # Cleanup intermediate arrays
                                del healthy_overlay, lesion_overlay, overlay_image
                                if display_scale != 1.0:
                                    del display_total_mask, display_lesion_mask
                                gc.collect()
                            except Exception as e:
                                st.error(f"❌ Overlay creation failed: {str(e)}")
                                overlay_pil = adjusted_display_image  # Fallback to original image

                        # Use a unique key for each correction mode to avoid slow redraws
                        correction_canvas_key = f"correction_canvas_{correction_mode}_{pen_size}"

                        # Enhanced correction canvas with better usability and faster switching
                        correction_canvas = st_canvas(
                            fill_color="rgba(0,0,0,0)",
                            stroke_width=pen_size,
                            stroke_color=stroke_color,
                            background_image=overlay_pil,
                            update_streamlit=True,
                            height=display_h,
                            width=display_w,
                            drawing_mode=canvas_mode,
                            key=correction_canvas_key,
                        )

                        st.caption("🟢 Green tint = Healthy areas | 🔴 Red tint = Detected lesions | Use pens to correct misclassifications")
                        
                        # Apply corrections with enhanced error handling
                        if st.session_state.get('apply_corrections', False):
                            st.session_state.apply_corrections = False
                            
                            try:
                                if correction_canvas.image_data is not None:
                                    # Get correction masks safely with memory monitoring
                                    try:
                                        correction_data = correction_canvas.image_data
                                        
                                        # Memory check before processing
                                        if correction_data.size > 1000000:  # Reduced threshold
                                            st.info("🔧 Processing corrections (large canvas)...")
                                        
                                        # Extract yellow and black pen strokes safely
                                        yellow_correction = np.zeros((correction_data.shape[0], correction_data.shape[1]), dtype=np.uint8)
                                        black_correction = np.zeros((correction_data.shape[0], correction_data.shape[1]), dtype=np.uint8)
                                        
                                        # Detect yellow strokes (high R and G, low B)
                                        yellow_pixels = (correction_data[:,:,0] > 200) & (correction_data[:,:,1] > 200) & (correction_data[:,:,2] < 100) & (correction_data[:,:,3] > 0)
                                        yellow_correction[yellow_pixels] = 255
                                        
                                        # Detect black strokes (low R, G, B)
                                        black_pixels = (correction_data[:,:,0] < 50) & (correction_data[:,:,1] < 50) & (correction_data[:,:,2] < 50) & (correction_data[:,:,3] > 0)
                                        black_correction[black_pixels] = 255
                                        
                                        # Cleanup pixel arrays
                                        del yellow_pixels, black_pixels
                                        gc.collect()
                                        
                                        # Scale corrections to match processing image safely
                                        if display_scale != 1.0:
                                            yellow_correction = cv2.resize(yellow_correction, (w, h), interpolation=cv2.INTER_NEAREST)
                                            black_correction = cv2.resize(black_correction, (w, h), interpolation=cv2.INTER_NEAREST)
                                        
                                        # Apply corrections to masks safely
                                        corrected_lesion_mask = lesion_mask.copy()
                                        corrected_total_mask = total_mask.copy()
                                        
                                        # Yellow pen: remove from lesions, add to healthy
                                        corrected_lesion_mask[yellow_correction > 0] = 0
                                        corrected_total_mask[yellow_correction > 0] = 255
                                        
                                        # Black pen: add to lesions
                                        corrected_lesion_mask[black_correction > 0] = 255
                                        
                                        # Cleanup correction arrays
                                        del yellow_correction, black_correction
                                        gc.collect()
                                    except Exception as e:
                                        st.error(f"❌ Correction processing failed: {str(e)}")
                                        # Fallback to original masks
                                        corrected_lesion_mask = lesion_mask.copy()
                                        corrected_total_mask = total_mask.copy()
                                
                                # Recalculate areas with corrections
                                corrected_mango_area_px = np.count_nonzero(corrected_total_mask)
                                corrected_lesion_area_px = np.count_nonzero(corrected_lesion_mask)
                                
                                if corrected_mango_area_px > 0:
                                    mm_per_px_sq = st.session_state.mm_per_px * st.session_state.mm_per_px
                                    corrected_mango_area_mm2 = corrected_mango_area_px * mm_per_px_sq
                                    corrected_lesion_area_mm2 = corrected_lesion_area_px * mm_per_px_sq
                                    corrected_lesion_percent = (corrected_lesion_area_mm2 / corrected_mango_area_mm2 * 100) if corrected_mango_area_mm2 > 0 else 0
                                    
                                    # Display corrected results
                                    st.markdown("### 📊 Corrected Results")
                                    corr_col1, corr_col2, corr_col3 = st.columns([1, 1, 1])
                                    
                                    with corr_col1:
                                        st.image(corrected_total_mask, caption="Corrected Mango Area", width=150)
                                    with corr_col2:
                                        st.image(corrected_lesion_mask, caption="Corrected Lesions", width=150)
                                    with corr_col3:
                                        # Show change indicators
                                        area_change = corrected_mango_area_mm2 - mango_area_mm2
                                        lesion_change = corrected_lesion_percent - lesion_percent
                                        
                                        st.metric("Corrected Area", f"{corrected_mango_area_mm2:.1f} mm²", delta=f"{area_change:+.1f} mm²")
                                        st.metric("Corrected Lesions", f"{corrected_lesion_area_mm2:.1f} mm²")
                                        st.metric("Corrected %", f"{corrected_lesion_percent:.1f}%", delta=f"{lesion_change:+.1f}%")
                                    
                                    # Update result with corrections
                                    result = {
                                        "Sample": len(st.session_state.samples) + 1,
                                        "Area (mm²)": round(corrected_mango_area_mm2, 1),
                                        "Lesions (mm²)": round(corrected_lesion_area_mm2, 1),
                                        "Lesion %": round(corrected_lesion_percent, 1)
                                    }
                                    
                                    st.dataframe(pd.DataFrame([result]), use_container_width=True)
                                    st.success("✅ Corrections applied! Updated measurements shown above.")
                            except Exception as e:
                                st.error(f"❌ Correction error: {str(e)}")
                                aggressive_cleanup()
                        
                        if st.button("✅ Add Sample", type="primary", help="Save this analysis to your sample collection"):
                            try:
                                # Use corrected result if available, otherwise use original
                                final_result = result
                                if safe_add_sample(final_result):
                                    st.success(f"✅ Sample {len(st.session_state.samples)} added successfully!")
                                    # Clean up memory safely
                                    try:
                                        del total_mask, lesion_mask, mask
                                    except:
                                        pass
                                    aggressive_cleanup()
                                    safe_rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to add sample: {str(e)}")
                                aggressive_cleanup()
                    else:
                        st.warning("⚠️ No mango detected. Try adjusting your selection or brightness.")
                        st.info("💡 **Tips:**\n- Make sure your shape covers the entire mango\n- Adjust brightness if the image is too dark/bright\n- Try a different drawing mode (circle, rectangle, polygon)")
                        
                except Exception as e:
                    aggressive_cleanup()
                    st.error(f"❌ Analysis error: {str(e)}")
                    st.info("🔄 Please try drawing your selection again")
            
            # Enhanced sample management with better error handling
            if st.session_state.samples:
                st.markdown("### 📊 All Samples")
                try:
                    df = pd.DataFrame(st.session_state.samples)
                    st.dataframe(df, use_container_width=True)
                    
                    # Enhanced summary statistics
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
                except Exception as e:
                    st.error(f"❌ Display error: {str(e)}")
                
                # Enhanced management controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🗑️ Clear All", help="Remove all samples", use_container_width=True):
                        try:
                            st.session_state.samples = []
                            aggressive_cleanup()
                            st.success("✅ All samples cleared!")
                            safe_rerun()
                        except Exception as e:
                            st.error(f"❌ Clear error: {str(e)}")
                        
                with col2:
                    # Enhanced delete specific sample
                    if st.session_state.samples:
                        try:
                            sample_to_delete = st.selectbox(
                                "🗑️ Delete specific sample:",
                                options=[f"Sample {i}" for i in range(1, len(st.session_state.samples) + 1)],
                                key="delete_sample_select"
                            )
                            if st.button("Delete Selected", use_container_width=True):
                                sample_idx = int(sample_to_delete.split()[1]) - 1
                                deleted_sample = st.session_state.samples.pop(sample_idx)
                                # Renumber remaining samples
                                for i, sample in enumerate(st.session_state.samples):
                                    sample["Sample"] = i + 1
                                st.success(f"✅ Deleted: Sample {sample_idx + 1}")
                                aggressive_cleanup()
                                safe_rerun()
                        except Exception as e:
                            st.error(f"❌ Delete error: {str(e)}")
                
                with col3:
                    # Enhanced download with better error handling
                    try:
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
                            
                            try:
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download CSV",
                                    csv,
                                    f"{safe_filename}.csv",
                                    "text/csv",
                                    help="Download analysis results",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"❌ Export error: {str(e)}")
                        else:
                            st.info("💾 Enter filename to enable download")
                    except Exception as e:
                        st.error(f"❌ Filename error: {str(e)}")
        else:
            st.info("👆 **Step 1:** Draw a line on the scale bar and set its real length to continue")
            st.markdown("**💡 Tips for scale setting:**\n- Use a ruler or known measurement in the image\n- Draw a straight line across the full length\n- Enter the exact measurement in millimeters")
            
    except MemoryError:
        # Specific handling for memory errors with enhanced cleanup
        aggressive_cleanup()
        # Emergency cleanup of all temporary session state
        temp_keys = [k for k in st.session_state.keys() if 'temp' in k.lower() or 'canvas' in k.lower()]
        for key in temp_keys:
            try:
                del st.session_state[key]
            except:
                pass
        gc.collect()
        st.error("❌ Out of memory! Please:")
        st.info("💡 **Memory Solutions:**\n- Use smaller images (< 1MB recommended)\n- Clear all samples if you have many\n- Refresh the page (F5)\n- Close other browser tabs")
        if st.button("🧹 Emergency Cleanup & Restart"):
            try:
                for key in list(st.session_state.keys()):
                    if key not in ['samples']:  # Only preserve samples
                        try:
                            del st.session_state[key]
                        except:
                            pass
                aggressive_cleanup()
                safe_rerun()
            except:
                st.info("Please refresh the page manually (F5)")
    except Exception as e:
        aggressive_cleanup()
        error_message = str(e)
        st.error(f"❌ Application error: {error_message}")
        
        # Provide specific help based on error type
        if "size" in error_message.lower() or "memory" in error_message.lower():
            st.info("💡 **Memory Issue:** Use a smaller image (< 1MB) or clear samples")
        elif "format" in error_message.lower() or "decode" in error_message.lower():
            st.info("📸 **Format Issue:** Try a different image format (JPG recommended)")
        elif "canvas" in error_message.lower():
            st.info("💡 **Canvas Issue:** Try refreshing the page or drawing again")
        else:
            st.info("💡 **General Solution:** Refresh the page (F5) or try a smaller image")
        
        if st.button("🔄 Restart Application"):
            safe_rerun()
    finally:
        # Safe final cleanup
        try:
            aggressive_cleanup()
        except:
            pass

# Enhanced cloud resource monitoring
if st.sidebar.button("💾 Memory Status"):
    try:
        import sys
        st.sidebar.info(f"🐍 Python: {sys.version}")
        st.sidebar.info(f"📊 Samples: {len(st.session_state.samples)}")
        st.sidebar.info(f"🕒 Last cleanup: {int(time.time() - st.session_state.last_cleanup)}s ago")
        if st.sidebar.button("🧹 Force Cleanup"):
            aggressive_cleanup()
            st.sidebar.success("✅ Cleanup completed")
    except Exception:
        st.sidebar.warning("⚠️ Memory info unavailable")

st.markdown("---")
st.markdown("🔬 **Plant Pathology Lab, UPLB** | Contact: jsmendoza5@up.edu.ph")