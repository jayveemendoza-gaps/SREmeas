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

# Initialize session state efficiently with cloud optimization
for key, default in [
    ("samples", []), 
    ("mm_per_px", None), 
    ("polygon_drawing", False),
    ("last_cleanup", time.time()),
    ("last_cache_clear", 0)  # Added for cache management
]:
    # Extra safety for cloud stability
    try:
        if key not in st.session_state:
            st.session_state[key] = default
    except Exception:
        # Session state might be corrupted, reinitialize safely
        try:
            st.session_state.clear()
            st.session_state[key] = default
        except Exception:
            pass  # Ultimate fallback - continue without session state

# Early cleanup check for cloud performance - less frequent
if time.time() - st.session_state.get('last_cleanup', 0) > 60:  # Every 60 seconds instead of 30
    gc.collect()
    st.session_state.last_cleanup = time.time()

# Ultra-optimized settings for maximum cloud stability
MAX_IMAGE_SIZE = 320      # Optimized for cloud speed
CANVAS_SIZE = 900         # Increased canvas size for more comfortable drawing
MAX_FILE_SIZE_MB = 8      # Reduced for faster processing
MEMORY_CLEANUP_INTERVAL = 5   # More frequent cleanup
MAX_SAMPLES = 25          # Optimized memory limit
CACHE_CLEANUP_INTERVAL = 60   # Clear cache every minute

def aggressive_cleanup():
    """Ultra-optimized memory cleanup for cloud stability"""
    gc.collect()
    try:
        cv2.setUseOptimized(True)
        # Clear specific large objects from session state safely
        cleanup_keys = ['temp_canvas_data', 'temp_masks', 'large_arrays', 'canvas_cache']
        for key in cleanup_keys:
            try:
                if key in st.session_state:
                    del st.session_state[key]
            except Exception:
                pass
    except Exception:
        pass
    
    # Safely update last_cleanup time
    try:
        st.session_state.last_cleanup = time.time()
    except Exception:
        pass
    
    # Optimize temp image cleanup safely
    try:
        if hasattr(st.session_state, 'temp_images'):
            if len(getattr(st.session_state, 'temp_images', [])) > 0:  # Clear immediately
                del st.session_state.temp_images
    except Exception:
        pass
    
    # Clear Streamlit cache periodically for cloud performance
    try:
        if (time.time() - getattr(st.session_state, 'last_cache_clear', 0)) > CACHE_CLEANUP_INTERVAL:
            try:
                st.cache_data.clear()
                st.session_state.last_cache_clear = time.time()
            except Exception:
                pass
    except Exception:
        pass

def check_memory_limit():
    """Optimized memory monitoring for cloud performance"""
    # Faster memory checks
    if len(st.session_state.samples) > MAX_SAMPLES:
        st.error(f"💾 Maximum samples reached ({MAX_SAMPLES}). Please download and clear samples.")
        return True
    if len(st.session_state.samples) > 12:  # Earlier warning for cloud
        st.warning("⚠️ Many samples stored. Consider downloading results soon.")
        return True
    # Simplified temp image check
    if hasattr(st.session_state, 'temp_images') and len(getattr(st.session_state, 'temp_images', [])) > 1:
        st.warning("⚠️ Multiple images in memory. Consider clearing for better performance.")
        return True
    return False

def should_cleanup():
    """Check if cleanup is needed - less aggressive for cloud stability"""
    return time.time() - st.session_state.get('last_cleanup', 0) > 10  # Increased from 5 to 10 seconds

def emergency_reset():
    """Emergency reset function to recover from crashes"""
    try:
        # Clear problematic session state keys safely
        problem_keys = ['polygon_drawing', 'temp_canvas_data', 'temp_masks', 'large_arrays', 'temp_images']
        for key in problem_keys:
            try:
                if key in st.session_state:
                    del st.session_state[key]
            except Exception:
                pass
        
        # Safer canvas key tracking for cloud stability
        try:
            canvas_keys = []
            for k in list(st.session_state.keys()):  # Convert to list for safety
                if 'canvas' in k.lower():
                    canvas_keys.append(k)
            
            # Clear all canvas-related session state
            for key in canvas_keys:
                try:
                    if key in st.session_state:
                        del st.session_state[key]
                except Exception:
                    pass
        except Exception:
            pass  # Session state might be corrupted, continue safely
        
        aggressive_cleanup()
        return True
    except Exception as e:
        try:
            st.error(f"❌ Reset failed: {str(e)}")
        except Exception:
            pass  # Even error display might fail in corrupted state
        return False

@st.cache_data(max_entries=1, ttl=90, show_spinner=False)  # Optimized TTL for cloud
def process_uploaded_image(uploaded_file, max_dim=MAX_IMAGE_SIZE):
    """Ultra-optimized image processing for cloud stability"""
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
        
        # Optimized size limits for cloud speed
        if original_size[0] * original_size[1] > 1200000:  # Reduced for faster processing
            st.warning("Large image detected. Optimizing for cloud speed.")
            max_dim = min(max_dim, 260)  # Optimized downscaling
        
        if image.mode != 'RGB':
            image = image.convert("RGB")
        scale = min(max_dim / image.height, max_dim / image.width, 1.0)
        if scale < 1.0:
            new_size = (int(image.width * scale), int(image.height * scale))
            # Use faster resampling for cloud
            image = image.resize(new_size, Image.LANCZOS)
            st.info(f"Resized: {original_size[0]}x{original_size[1]} → {new_size[0]}x{new_size[1]}")
        
        # Optimize array conversion
        image_np = np.array(image, dtype=np.uint8)
        del image  # Immediate cleanup
        return image_np, original_size, scale
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        return None, None, None

def quick_color_analysis(image_np, mask, mm_per_px):
    """Optimized color analysis with preserved masking formula and cloud-speed enhancements"""
    if image_np is None or mask is None or mm_per_px is None:
        return 0, 0, 0, None, None
    if mask.size == 0 or np.max(mask) == 0:
        return 0, 0, 0, None, None
    if mask.shape != image_np.shape[:2]:
        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Optimized HSV conversion for cloud
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    # Combine mask operations for efficiency
    mango_mask1 = cv2.inRange(hsv, (20, 40, 60), (85, 255, 255))
    mango_mask2 = cv2.inRange(hsv, (15, 25, 40), (80, 200, 200))
    mango_mask3 = cv2.inRange(hsv, (25, 20, 100), (75, 150, 255))
    healthy_mask = cv2.bitwise_or(mango_mask1, cv2.bitwise_or(mango_mask2, mango_mask3))
    
    # Optimized lesion detection
    lesion_mask1 = cv2.inRange(hsv, (8, 50, 30), (25, 255, 140))
    lesion_mask2 = cv2.inRange(hsv, (0, 60, 25), (15, 255, 120))
    lesion_mask3 = cv2.inRange(hsv, (0, 0, 0), (180, 255, 45))
    lesion_mask4 = cv2.inRange(hsv, (0, 10, 20), (180, 80, 100))
    raw_lesion_mask = cv2.bitwise_or(cv2.bitwise_or(lesion_mask1, lesion_mask2), 
                                     cv2.bitwise_or(lesion_mask3, lesion_mask4))
    
    # Cleanup intermediate masks for memory efficiency
    del mango_mask1, mango_mask2, mango_mask3, lesion_mask1, lesion_mask2, lesion_mask3, lesion_mask4, hsv
    
    healthy_mask = cv2.bitwise_and(healthy_mask, mask)
    raw_lesion_mask = cv2.bitwise_and(raw_lesion_mask, mask)
    
    # Optimized morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_lesion_mask = cv2.morphologyEx(raw_lesion_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_lesion_mask = cv2.morphologyEx(cleaned_lesion_mask, cv2.MORPH_OPEN, kernel)
    
    # Faster connected components analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_lesion_mask, connectivity=8)
    final_lesion_mask = np.zeros_like(cleaned_lesion_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 3:
            final_lesion_mask[labels == i] = 255
    
    # Cleanup intermediate arrays
    del labels, stats, cleaned_lesion_mask
    
    total_mango_mask = cv2.bitwise_or(healthy_mask, final_lesion_mask)
    
    # Optimized morphological refinement
    base_kernel_size = max(2, int(np.sqrt(np.count_nonzero(mask)) / 50))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_kernel_size, base_kernel_size))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_kernel_size + 1, base_kernel_size + 1))
    
    # Combined morphological operations for speed
    final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_CLOSE, kernel_medium)
    final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_OPEN, kernel_small)
    final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_CLOSE, kernel_small)
    final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_OPEN, kernel_small)
    
    # Fast area calculations
    mango_area_px = np.count_nonzero(total_mango_mask)
    lesion_area_px = np.count_nonzero(final_lesion_mask)
    
    if mango_area_px == 0:
        return 0, 0, 0, None, None
    
    mm_per_px_sq = mm_per_px * mm_per_px
    mango_area_mm2 = mango_area_px * mm_per_px_sq
    lesion_area_mm2 = lesion_area_px * mm_per_px_sq
    lesion_percent = (lesion_area_mm2 / mango_area_mm2 * 100) if mango_area_mm2 > 0 else 0
    
    # Memory cleanup before return
    gc.collect()
    
    return mango_area_mm2, lesion_area_mm2, lesion_percent, total_mango_mask, final_lesion_mask

# File uploader with cloud-optimized messaging
uploaded_file = st.file_uploader(
    "Upload mango image (JPG/PNG)", 
    type=["png", "jpg", "jpeg"],
    help=f"Max size: {MAX_FILE_SIZE_MB}MB. Cloud-optimized: Use JPG format, <2MB recommended for fastest processing"
)

def safe_rerun():
    """Cloud-optimized rerun function with safer fallbacks"""
    aggressive_cleanup()
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            try:
                # Safer session state cleanup for cloud
                keys_to_preserve = ['samples', 'mm_per_px', 'last_cleanup']
                keys_to_delete = []
                
                for key in list(st.session_state.keys()):
                    if key not in keys_to_preserve:
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    try:
                        if key in st.session_state:
                            del st.session_state[key]
                    except Exception:
                        pass
                
                st.info("🔄 App optimized. Continue with your analysis.")
            except Exception:
                st.write("🔄 Please refresh the page (F5) for best performance")

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
        
        # Optimized file size feedback for cloud performance
        if file_size > 4:
            st.warning(f"⚠️ Large file: {file_size:.1f}MB - expect slower cloud processing")
        elif file_size > 2:
            st.info(f"📏 File size: {file_size:.1f}MB - good for cloud processing")
        else:
            st.success(f"✅ File size: {file_size:.1f}MB - optimal cloud performance")
        
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
        
        # Store original image for analysis
        original_image_np = image_np.copy()  # Keep original for analysis
        
        h, w = image_np.shape[:2]
        st.success(f"✅ Image loaded: {w}x{h} pixels (scale: {scale:.2f})")
        
        # Calculate optimal display size with cloud performance limits
        display_scale = min(CANVAS_SIZE / w, CANVAS_SIZE / h, 1.0)
        display_w = int(w * display_scale)
        display_h = int(h * display_scale)
        
        # Validate calculated dimensions
        if display_w <= 0 or display_h <= 0:
            st.error(f"❌ Invalid display dimensions calculated: {display_w}x{display_h}")
            st.stop()
        
        # Optimized minimum size for better usability - ensure at least 300px on smaller dimension
        MIN_CANVAS_SIZE = 300  # Minimum size for comfortable drawing
        if display_w < MIN_CANVAS_SIZE or display_h < MIN_CANVAS_SIZE:
            # Calculate scale needed to make smaller dimension at least 300px
            min_scale = max(MIN_CANVAS_SIZE / w, MIN_CANVAS_SIZE / h)
            if min_scale <= 3.0:  # Allow more upscaling for better usability
                display_w = int(w * min_scale)
                display_h = int(h * min_scale)
                display_scale = min_scale
                st.info(f"🔧 Canvas enlarged for better usability: {display_w}x{display_h} (scale: {display_scale:.2f})")
        
        # Additional validation after min size adjustment
        if display_w <= 0 or display_h <= 0:
            st.error(f"❌ Invalid display dimensions after adjustment: {display_w}x{display_h}")
            st.stop()
            
        # Cloud-optimized maximum limits but maintain usability
        MAX_DISPLAY_DIM = 2200  # Increased for bigger, more comfortable canvas
        if display_w > MAX_DISPLAY_DIM or display_h > MAX_DISPLAY_DIM:
            st.warning(f"⚠️ Display size optimized for cloud ({display_w}x{display_h} → smaller).")
            scale_factor = min(MAX_DISPLAY_DIM / display_w, MAX_DISPLAY_DIM / display_h)
            display_w = int(display_w * scale_factor)
            display_h = int(display_h * scale_factor)
            display_scale *= scale_factor
        
        # Cloud-optimized memory check with usability balance
        if display_w * display_h * 3 > 2000000:  # Increased for bigger canvas support
            st.warning("⚠️ Optimizing display size for cloud performance.")
            display_scale = min(display_scale, 0.8)  # Less aggressive scaling for bigger canvas
            display_w = int(w * display_scale)
            display_h = int(h * display_scale)
            
            # Ensure minimum size is still maintained after scaling
            MIN_CANVAS_SIZE = 300
            if display_w < MIN_CANVAS_SIZE or display_h < MIN_CANVAS_SIZE:
                min_scale = max(MIN_CANVAS_SIZE / w, MIN_CANVAS_SIZE / h)
                display_w = int(w * min_scale)
                display_h = int(h * min_scale)
                display_scale = min_scale
            
            # Final validation
            if display_w <= 0 or display_h <= 0:
                st.error("❌ Display scaling resulted in invalid dimensions")
                st.stop()
        
        # Create display image with optimized error handling
        try:
            display_image = Image.fromarray(image_np)
            if display_scale != 1.0:
                # Use faster resampling for cloud
                display_image = display_image.resize((display_w, display_h), Image.LANCZOS)
            # Keep image_np for analysis - don't delete yet
            gc.collect()
        except Exception as e:
            st.error(f"❌ Display image creation failed: {str(e)}")
            aggressive_cleanup()
            st.stop()
        
        # --- Step 1: Scale Setting ---
        st.markdown("## 1️⃣ Set Scale")
        st.info("📏 Draw a line on a known measurement (ruler/scale bar)")
        st.success(f"✅ Canvas size optimized: {display_w}x{display_h} pixels for comfortable scale setting")
        
        # Enhanced scale canvas with better usability and error handling
        try:
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
        except Exception as e:
            st.error(f"❌ Scale canvas creation failed: {str(e)}")
            st.error("Please try refreshing the page or using a smaller image.")
            
            # Offer immediate recovery
            if st.button("🔄 Try Again", key="scale_canvas_retry"):
                emergency_reset()
                st.rerun()

        scale_length_mm = st.number_input(
            "Real length of drawn line (mm):",
            min_value=0.1,
            value=10.0,
            step=1.0
        )
        
        # Safe scale calculation with enhanced error handling
        scale_px = None
        if scale_canvas and hasattr(scale_canvas, 'json_data') and scale_canvas.json_data:
            try:
                objects = scale_canvas.json_data.get("objects", [])
                if objects and isinstance(objects, list):
                    # Get the last drawn object
                    obj = objects[-1]
                    if obj and isinstance(obj, dict) and obj.get("type") == "line":
                        # Safe coordinate extraction
                        try:
                            x1 = float(obj.get("x1", 0))
                            y1 = float(obj.get("y1", 0))
                            x2 = float(obj.get("x2", 0))
                            y2 = float(obj.get("y2", 0))
                            
                            dx = x2 - x1
                            dy = y2 - y1
                            scale_px = np.sqrt(dx*dx + dy*dy)
                            
                            if scale_px <= 0:
                                st.warning("⚠️ Please draw a line with some length")
                                scale_px = None
                            else:
                                if display_scale != 1.0 and display_scale > 0:
                                    scale_px /= display_scale
                                st.success(f"📏 Line drawn: {scale_px:.1f} pixels")
                        except (KeyError, ValueError, TypeError) as coord_e:
                            st.warning(f"⚠️ Error reading line coordinates: {str(coord_e)}")
                            scale_px = None
                    else:
                        st.info("📏 Please draw a line on the scale")
            except Exception as e:
                st.warning(f"⚠️ Scale calculation error: {str(e)}")
                scale_px = None
        
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
            
            # Optimized brightness adjustment
            try:
                if brightness != 1.0:
                    display_image_array = np.array(display_image, dtype=np.float32)
                    if display_image_array.size > 800000:
                        st.info("🔧 Optimizing brightness for cloud...")
                    display_image_array = np.clip(display_image_array * brightness, 0, 255).astype(np.uint8)
                    adjusted_display_image = Image.fromarray(display_image_array)
                    del display_image_array
                    gc.collect()
                else:
                    adjusted_display_image = display_image
            except (MemoryError, Exception) as e:
                st.warning(f"⚠️ Using original image for stability: {str(e)}")
                adjusted_display_image = display_image
                if isinstance(e, MemoryError):
                    aggressive_cleanup()
            
            # Mode-specific instructions
            if drawing_mode == "transform":
                st.info("🔄 **Transform Mode**: Click and drag shapes to move, drag corners/edges to resize.")
            elif drawing_mode == "polygon":
                st.info("📐 **Polygon Mode**: Click to place points, double-click to close polygon.")
            else:
                st.info(f"✏️ **{drawing_mode.title()} Mode**: Draw a new {drawing_mode} around the mango.")
            
            # Canvas stability check
            canvas_attempts = st.session_state.get('canvas_attempts', 0)
            if canvas_attempts > 3:
                st.warning("⚠️ Multiple canvas errors detected. Consider refreshing the page.")

            # Enhanced canvas with better usability and error handling
            try:
                # Reset error counter on successful canvas creation
                if 'canvas_attempts' in st.session_state:
                    st.session_state.canvas_attempts = 0
                    
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
            except Exception as e:
                # Track canvas creation errors
                st.session_state.canvas_attempts = st.session_state.get('canvas_attempts', 0) + 1
                
                st.error(f"❌ Canvas creation failed: {str(e)}")
                st.error("Please try refreshing the page or using a smaller image.")
                
                # Offer immediate recovery
                if st.button("🔄 Try Again", key="mango_canvas_retry"):
                    emergency_reset()
                    st.rerun()

            # Safe analysis processing with enhanced error handling
            process_analysis = False
            
            # Comprehensive safety checks for canvas data
            try:
                if (canvas_result is not None and 
                    hasattr(canvas_result, 'image_data') and 
                    canvas_result.image_data is not None):
                    
                    # Additional safety check for image data shape
                    if (len(canvas_result.image_data.shape) >= 3 and 
                        canvas_result.image_data.shape[2] >= 4 and
                        canvas_result.image_data.size > 0):
                        
                        # Safe alpha channel check
                        try:
                            alpha_channel = canvas_result.image_data[:,:,3]
                            has_drawing = np.any(alpha_channel > 0)
                        except (IndexError, ValueError) as e:
                            st.warning(f"⚠️ Canvas data format error: {str(e)}")
                            has_drawing = False
                        
                        if has_drawing:
                            if drawing_mode == "polygon":
                                try:
                                    if (canvas_result.json_data and 
                                        isinstance(canvas_result.json_data, dict) and
                                        canvas_result.json_data.get("objects")):
                                        
                                        objects = canvas_result.json_data["objects"]
                                        if isinstance(objects, list):
                                            polygon_objects = [obj for obj in objects if obj and obj.get("type") == "polygon"]
                                            path_objects = [obj for obj in objects if obj and obj.get("type") == "path"]
                                            
                                            if polygon_objects:
                                                if not st.session_state.get("polygon_drawing", False):
                                                    st.success("✅ Polygon completed! Computing analysis...")
                                                    st.session_state.polygon_drawing = True
                                                process_analysis = True
                                            elif path_objects:
                                                st.info("🔄 Drawing polygon... Double-click to close")
                                                st.session_state.polygon_drawing = False
                                except Exception as e:
                                    st.warning(f"⚠️ Polygon processing error: {str(e)}")
                                    st.session_state.polygon_drawing = False
                            else:
                                process_analysis = True
            except Exception as e:
                st.error(f"❌ Canvas processing error: {str(e)}")
                process_analysis = False
            
            if process_analysis:
                try:
                    # Memory check before analysis
                    if check_memory_limit():
                        st.error("❌ Memory limit reached. Please try a smaller image or clear samples.")
                        st.stop()
                    
                    # Create mask safely with comprehensive error handling
                    try:
                        # Validate canvas result structure
                        if (canvas_result is None or 
                            not hasattr(canvas_result, 'image_data') or 
                            canvas_result.image_data is None):
                            st.error("❌ Invalid canvas data")
                            st.stop()
                        
                        # Validate image data dimensions
                        image_data = canvas_result.image_data
                        if len(image_data.shape) < 3:
                            st.error("❌ Invalid canvas image format - missing channels")
                            st.stop()
                        
                        if image_data.shape[2] < 4:
                            st.error("❌ Invalid canvas image format - missing alpha channel")
                            st.stop()
                        
                        # Safe mask creation with bounds checking
                        try:
                            alpha_channel = image_data[:,:,3]
                            mask = (alpha_channel > 0).astype(np.uint8) * 255
                        except IndexError as e:
                            st.error(f"❌ Alpha channel access error: {str(e)}")
                            st.stop()
                        
                        # Validate mask
                        if mask.size == 0:
                            st.error("❌ Empty mask generated")
                            st.stop()
                        
                        if np.max(mask) == 0:
                            st.warning("⚠️ No selection detected. Please draw on the image.")
                            st.stop()
                        
                        # Validate mask dimensions match expected canvas size
                        expected_h, expected_w = display_h, display_w
                        if mask.shape[0] != expected_h or mask.shape[1] != expected_w:
                            st.warning(f"⚠️ Mask dimension mismatch. Expected: {expected_w}x{expected_h}, Got: {mask.shape[1]}x{mask.shape[0]}")
                            # Try to resize mask to expected dimensions
                            try:
                                mask = cv2.resize(mask, (expected_w, expected_h), interpolation=cv2.INTER_NEAREST)
                            except Exception as resize_e:
                                st.error(f"❌ Failed to fix mask dimensions: {str(resize_e)}")
                                st.stop()
                        
                    except Exception as e:
                        st.error(f"❌ Mask creation failed: {str(e)}")
                        st.error("Please try drawing your shape again, or refresh the page if the problem persists.")
                        st.stop()
                    
                    # Scale mask to match processing image safely
                    try:
                        # Validate original image dimensions
                        if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
                            st.error(f"❌ Invalid image dimensions: {w}x{h}")
                            st.stop()
                        
                        if display_scale != 1.0:
                            # Validate scaling parameters
                            if display_scale <= 0:
                                st.error(f"❌ Invalid display scale: {display_scale}")
                                st.stop()
                            
                            # Safe mask resizing with validation
                            try:
                                original_mask_shape = mask.shape
                                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                                
                                # Validate resize result
                                if mask.size == 0:
                                    st.error("❌ Mask scaling failed - empty result")
                                    st.stop()
                                
                                if mask.shape[0] != h or mask.shape[1] != w:
                                    st.error(f"❌ Mask scaling failed - wrong dimensions. Expected: {w}x{h}, Got: {mask.shape[1]}x{mask.shape[0]}")
                                    st.stop()
                                
                                st.info(f"🔧 Mask scaled: {original_mask_shape} → {mask.shape}")
                                
                            except cv2.error as cv_e:
                                st.error(f"❌ OpenCV resize error: {str(cv_e)}")
                                st.stop()
                            except Exception as resize_e:
                                st.error(f"❌ Mask resize error: {str(resize_e)}")
                                st.stop()
                        
                        # Final mask validation
                        if mask.shape[0] != h or mask.shape[1] != w:
                            st.error(f"❌ Final mask dimension check failed. Expected: {w}x{h}, Got: {mask.shape[1]}x{mask.shape[0]}")
                            st.stop()
                            
                    except Exception as e:
                        st.error(f"❌ Mask scaling error: {str(e)}")
                        st.error("Please try with a smaller image or refresh the page.")
                        aggressive_cleanup()
                        st.stop()
                    
                    # Analyze with enhanced error handling and progress
                    with st.spinner("🔬 Analyzing mango (cloud processing may take longer)..."):
                        try:
                            # Use original image for analysis
                            mango_area_mm2, lesion_area_mm2, lesion_percent, total_mask, lesion_mask = quick_color_analysis(
                                original_image_np, mask, st.session_state.mm_per_px
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
                                ["🟡 Yellow Pen (Mark as Healthy)", "⚫ Black Pen (Mark as Lesion)"],
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

                        # Create an overlay image showing current classification
                        try:
                            overlay_image = np.array(adjusted_display_image)
                            
                            # Check if we should create overlay for performance
                            if overlay_image.size > 600000:  # Optimized threshold
                                overlay_pil = adjusted_display_image
                                st.info("🚀 Using optimized display for faster cloud performance")
                            else:
                                # Resize masks to display dimensions
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

                                # Cleanup
                                del healthy_overlay, lesion_overlay, overlay_image
                                if display_scale != 1.0:
                                    del display_total_mask, display_lesion_mask
                                gc.collect()
                        except Exception as e:
                            st.error(f"❌ Overlay creation failed: {str(e)}")
                            overlay_pil = adjusted_display_image

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
                                    # Clean up memory safely including original image
                                    try:
                                        del total_mask, lesion_mask, mask, original_image_np
                                    except Exception:
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
        # Emergency cleanup of all temporary session state with safer iteration
        try:
            temp_keys = []
            for k in list(st.session_state.keys()):  # Convert to list to avoid iteration issues
                if 'temp' in k.lower() or 'canvas' in k.lower():
                    temp_keys.append(k)
            
            for key in temp_keys:
                try:
                    if key in st.session_state:  # Double-check existence
                        del st.session_state[key]
                except Exception:
                    pass
        except Exception:
            pass  # Session state might be corrupted
        
        gc.collect()
        st.error("❌ Cloud memory limit reached!")
        st.info("💡 Use smaller images (< 2MB) for best performance")
        if st.button("🧹 Clear Memory"):
            try:
                # Safer session state cleanup
                keys_to_preserve = ['samples']
                keys_to_delete = []
                
                for key in list(st.session_state.keys()):
                    if key not in keys_to_preserve:
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    try:
                        if key in st.session_state:
                            del st.session_state[key]
                    except Exception:
                        pass
                
                aggressive_cleanup()
                safe_rerun()
            except Exception:
                st.info("Please refresh the page (F5)")

    except Exception as e:
        aggressive_cleanup()
        error_message = str(e)
        st.error(f"❌ Processing error: {error_message}")
        
        if st.button("🔄 Restart Application"):
            safe_rerun()

# Add simple emergency controls in sidebar
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