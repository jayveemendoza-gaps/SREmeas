import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import gc
from io import BytesIO
import weakref
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

# Cloud-optimized settings with aggressive memory management
MAX_IMAGE_SIZE = 300      # Reduced for cloud memory limits
CANVAS_SIZE = 500         # Reduced canvas size for cloud
MAX_FILE_SIZE_MB = 5      # Reduced file size limit
MEMORY_CLEANUP_INTERVAL = 30  # Cleanup every 30 seconds

# Memory management utilities
def aggressive_cleanup():
    """Aggressive memory cleanup for cloud deployment"""
    gc.collect()
    # Clear OpenCV cache
    cv2.setUseOptimized(True)
    # Update cleanup timestamp
    st.session_state.last_cleanup = time.time()

def should_cleanup():
    """Check if cleanup is needed"""
    return time.time() - st.session_state.last_cleanup > MEMORY_CLEANUP_INTERVAL

@st.cache_data(max_entries=1, ttl=300, show_spinner=False)  # Reduced cache for cloud
def process_uploaded_image(uploaded_file, max_dim=MAX_IMAGE_SIZE):
    """Cloud-optimized image processing with aggressive memory management"""
    try:
        if not uploaded_file or len(uploaded_file) == 0:
            return None, None, None
            
        file_size_mb = len(uploaded_file) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File too large: {file_size_mb:.1f}MB. Use files under {MAX_FILE_SIZE_MB}MB.")
            return None, None, None
        
        # Process image in chunks to avoid memory spikes
        image = Image.open(BytesIO(uploaded_file))
        original_size = image.size
        
        # Convert to RGB immediately and delete original
        if image.mode != 'RGB':
            rgb_image = image.convert("RGB")
            del image
            image = rgb_image
        
        # Aggressive downscaling for cloud
        scale = min(max_dim / image.height, max_dim / image.width, 1.0)
        if scale < 1.0:
            new_size = (int(image.width * scale), int(image.height * scale))
            # Use NEAREST for speed on cloud
            resized_image = image.resize(new_size, Image.NEAREST)
            del image
            image = resized_image
            st.info(f"Resized from {original_size[0]}x{original_size[1]} to {new_size[0]}x{new_size[1]} pixels")
        
        # Convert to numpy with minimal memory usage
        image_np = np.array(image, dtype=np.uint8)
        del image
        
        # Immediate cleanup
        aggressive_cleanup()
        
        return image_np, original_size, scale
        
    except MemoryError:
        aggressive_cleanup()
        st.error("Out of memory. Please use a smaller image (< 2MB recommended).")
        return None, None, None
    except Exception as e:
        aggressive_cleanup()
        st.error(f"Image processing failed: {str(e)}")
        return None, None, None

def quick_color_analysis(image_np, mask, mm_per_px):
    """Cloud-optimized color analysis - MASKING FORMULA PRESERVED"""
    try:
        if image_np is None or mask is None or mm_per_px is None:
            return 0, 0, 0, None, None
        
        if mask.size == 0 or np.max(mask) == 0:
            return 0, 0, 0, None, None
        
        # Ensure mask matches image dimensions
        if mask.shape != image_np.shape[:2]:
            mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # ORIGINAL MASKING FORMULA - PRESERVED EXACTLY
        # Efficient HSV conversion
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Enhanced color range detection for mango areas
        # Healthy mango: yellows, greens, and lighter variants
        # Range 1: Bright yellow-green (healthy mango)
        mango_mask1 = cv2.inRange(hsv, (20, 40, 60), (85, 255, 255))
        # Range 2: Medium yellow-green 
        mango_mask2 = cv2.inRange(hsv, (15, 25, 40), (80, 200, 200))
        # Combine healthy mango masks
        healthy_mask = cv2.bitwise_or(mango_mask1, mango_mask2)
        
        # Shadow detection (darker but still colored areas)
        # Shadows typically have lower saturation and value but retain some hue
        shadow_mask1 = cv2.inRange(hsv, (10, 15, 20), (90, 120, 100))  # Dark but colored
        shadow_mask2 = cv2.inRange(hsv, (15, 10, 15), (85, 80, 80))    # Very dark shadows
        shadow_mask = cv2.bitwise_or(shadow_mask1, shadow_mask2)
        
        # True lesion detection (browns, dark spots, diseased areas)
        # Range 1: Brown lesions
        lesion_mask1 = cv2.inRange(hsv, (5, 60, 30), (25, 255, 150))
        # Range 2: Dark brown to black lesions
        lesion_mask2 = cv2.inRange(hsv, (0, 30, 10), (20, 255, 100))
        # Range 3: Very dark lesions (black spots)
        lesion_mask3 = cv2.inRange(hsv, (0, 0, 0), (180, 255, 40))
        # Combine lesion masks
        raw_lesion_mask = cv2.bitwise_or(lesion_mask1, cv2.bitwise_or(lesion_mask2, lesion_mask3))
        
        # Apply ROI mask to all
        healthy_mask = cv2.bitwise_and(healthy_mask, mask)
        shadow_mask = cv2.bitwise_and(shadow_mask, mask)
        raw_lesion_mask = cv2.bitwise_and(raw_lesion_mask, mask)
        
        # Morphological operations to separate border shadows from interior lesions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Create border and interior regions
        # Erode mask to get interior region
        interior_mask = cv2.erode(mask, kernel, iterations=3)
        # Border region = original - interior
        border_mask = cv2.subtract(mask, interior_mask)
        
        # Distance transform to identify regions far from edges
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        # Normalize distance transform
        if np.max(dist_transform) > 0:
            dist_transform = dist_transform / np.max(dist_transform)
        
        # Create weight masks based on distance from border
        # Areas closer to center get higher weight for lesion classification
        center_weight = (dist_transform > 0.3).astype(np.float32)  # Inner 70% of shape
        border_weight = (dist_transform <= 0.3).astype(np.float32)  # Outer 30% of shape
        
        # Separate shadows from lesions using position and morphology
        # Shadows in border regions are likely actual shadows
        border_shadows = cv2.bitwise_and(shadow_mask, border_mask)
        
        # Lesions in interior regions are likely true lesions
        interior_lesions = cv2.bitwise_and(raw_lesion_mask, interior_mask)
        
        # Handle ambiguous dark areas using distance weighting
        ambiguous_dark = cv2.bitwise_and(shadow_mask, interior_mask)
        
        # Apply distance-based classification to ambiguous areas
        # Areas far from border are more likely lesions
        ambiguous_lesions = cv2.bitwise_and(
            ambiguous_dark, 
            (center_weight > 0.7).astype(np.uint8) * 255
        )
        
        # Combine final lesion mask
        final_lesion_mask = cv2.bitwise_or(interior_lesions, ambiguous_lesions)
        
        # Create final masks
        total_mango_mask = cv2.bitwise_or(healthy_mask, cv2.bitwise_or(border_shadows, final_lesion_mask))
        
        # Clean up lesion mask with morphological operations
        # Remove small noise and fill small gaps
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_CLOSE, kernel_small)
        final_lesion_mask = cv2.morphologyEx(final_lesion_mask, cv2.MORPH_OPEN, kernel_small)
        # END OF ORIGINAL MASKING FORMULA
        
        # Fast pixel counting
        mango_area_px = np.count_nonzero(total_mango_mask)
        lesion_area_px = np.count_nonzero(final_lesion_mask)
        
        if mango_area_px == 0:
            return 0, 0, 0, None, None
        
        # Calculate areas
        mm_per_px_sq = mm_per_px * mm_per_px
        mango_area_mm2 = mango_area_px * mm_per_px_sq
        lesion_area_mm2 = lesion_area_px * mm_per_px_sq
        lesion_percent = (lesion_area_mm2 / mango_area_mm2 * 100) if mango_area_mm2 > 0 else 0
        
        # Aggressive cleanup for cloud
        del hsv, mango_mask1, mango_mask2, healthy_mask, shadow_mask1, shadow_mask2, shadow_mask
        del lesion_mask1, lesion_mask2, lesion_mask3, raw_lesion_mask, border_shadows, interior_lesions
        del ambiguous_dark, ambiguous_lesions, dist_transform, center_weight, border_weight
        del interior_mask, border_mask, kernel, kernel_small
        
        # Return smaller masks for cloud memory efficiency
        return mango_area_mm2, lesion_area_mm2, lesion_percent, total_mango_mask, final_lesion_mask
        
    except Exception as e:
        aggressive_cleanup()
        st.error(f"Analysis failed: {str(e)}")
        return 0, 0, 0, None, None

# File uploader with cloud optimization
uploaded_file = st.file_uploader(
    "Upload mango image (JPG/PNG)", 
    type=["png", "jpg", "jpeg"],
    help=f"Max size: {MAX_FILE_SIZE_MB}MB. Smaller files process faster on cloud."
)

def safe_rerun():
    """Cloud-optimized rerun function with cleanup"""
    aggressive_cleanup()
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            st.write("Please refresh the page manually")

if uploaded_file:
    try:
        # Periodic cleanup check
        if should_cleanup():
            aggressive_cleanup()
        
        # File size check
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size > MAX_FILE_SIZE_MB:
            st.error(f"File too large: {file_size:.1f}MB")
            st.stop()
        
        if file_size > 2:
            st.warning(f"File size: {file_size:.1f}MB - consider using smaller images for faster processing")
        
        # Process image with cloud optimization
        with st.spinner("Processing image..."):
            image_np, original_size, scale = process_uploaded_image(uploaded_file.getvalue())
        
        if image_np is None:
            st.error("Failed to process image. Try a smaller file (< 2MB recommended).")
            st.stop()
        
        h, w = image_np.shape[:2]
        st.info(f"Processing: {w}x{h} pixels (scale: {scale:.2f})")
        
        # Calculate display size with cloud optimization
        display_scale = min(CANVAS_SIZE / w, CANVAS_SIZE / h, 1.0)
        display_w = int(w * display_scale)
        display_h = int(h * display_scale)
        
        # Create display image efficiently
        display_image = Image.fromarray(image_np)
        if display_scale < 1.0:
            display_image = display_image.resize((display_w, display_h), Image.NEAREST)  # Faster for cloud
        
        # --- Step 1: Scale setting ---
        st.markdown("## 1️⃣ Set Scale")
        st.info("Draw a line on a known measurement (ruler/scale bar)")
        
        scale_canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=2,  # Reduced for cloud
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
        
        # Calculate scale efficiently
        scale_px = None
        if scale_canvas.json_data and scale_canvas.json_data.get("objects"):
            try:
                obj = scale_canvas.json_data["objects"][-1]
                if obj.get("type") == "line":
                    dx = obj["x2"] - obj["x1"]
                    dy = obj["y2"] - obj["y1"]
                    scale_px = np.sqrt(dx*dx + dy*dy)
                    if display_scale < 1.0:
                        scale_px /= display_scale
                    st.info(f"Line: {scale_px:.1f} pixels")
            except:
                st.warning("Draw a clear line on the scale")
        
        if scale_px and scale_length_mm > 0:
            st.session_state.mm_per_px = scale_length_mm / scale_px
            st.success(f"Scale: {st.session_state.mm_per_px:.4f} mm/pixel")
            
            # --- Step 2: Mango analysis ---
            st.markdown("## 2️⃣ Analyze Mango")
            st.info("Draw around one mango at a time")
            
            # Drawing mode selection
            drawing_mode = st.radio(
                "Drawing mode:",
                ["circle", "rect", "polygon", "transform"],
                horizontal=True,
                help="Circle for round mangoes, polygon for irregular shapes, transform to adjust existing shapes"
            )
            
            # Brightness adjustment (cloud-optimized)
            brightness = st.slider(
                "🔆 Brightness:",
                min_value=0.7,
                max_value=1.5,
                value=1.0,
                step=0.1,
                help="Adjust brightness for better visibility"
            )
            
            # Apply brightness adjustment efficiently
            if brightness != 1.0:
                display_image_array = np.array(display_image, dtype=np.float32)
                display_image_array = np.clip(display_image_array * brightness, 0, 255).astype(np.uint8)
                adjusted_display_image = Image.fromarray(display_image_array)
                del display_image_array
                aggressive_cleanup()
            else:
                adjusted_display_image = display_image
            
            # Mode-specific instructions
            if drawing_mode == "transform":
                st.info("🔄 **Transform Mode**: Click and drag shapes to move, drag corners/edges to resize.")
            elif drawing_mode == "polygon":
                st.info("📐 **Polygon Mode**: Click to place points, double-click to close polygon.")
            else:
                st.info(f"✏️ **{drawing_mode.title()} Mode**: Draw a new {drawing_mode} around the mango.")
            
            canvas_result = st_canvas(
                fill_color="rgba(255,165,0,0.15)",  # Lighter fill for cloud
                stroke_width=2,
                stroke_color="rgba(255,165,0,1)",
                background_image=adjusted_display_image,
                update_streamlit=True,
                height=display_h,
                width=display_w,
                drawing_mode=drawing_mode,
                key="mango_canvas",
            )

            # Process analysis with cloud optimization
            process_analysis = False
            
            if canvas_result.image_data is not None and np.any(canvas_result.image_data[:,:,3] > 0):
                if drawing_mode == "polygon":
                    if canvas_result.json_data and canvas_result.json_data.get("objects"):
                        polygon_objects = [obj for obj in canvas_result.json_data["objects"] if obj.get("type") == "polygon"]
                        path_objects = [obj for obj in canvas_result.json_data["objects"] if obj.get("type") == "path"]
                        
                        if polygon_objects:
                            if not st.session_state.polygon_drawing:
                                st.success("✅ Polygon closed! Computing analysis...")
                                st.session_state.polygon_drawing = True
                            process_analysis = True
                        elif path_objects:
                            st.info("🔄 Drawing polygon... Double-click to close")
                            st.session_state.polygon_drawing = False
                else:
                    process_analysis = True
            
            if process_analysis:
                try:
                    # Create mask efficiently
                    mask = (canvas_result.image_data[:,:,3] > 0).astype(np.uint8) * 255
                    
                    # Scale mask to match processing image
                    if display_scale < 1.0:
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Analyze with cloud optimization
                    with st.spinner("Analyzing mango..."):
                        mango_area_mm2, lesion_area_mm2, lesion_percent, total_mask, lesion_mask = quick_color_analysis(
                            image_np, mask, st.session_state.mm_per_px
                        )
                    
                    if mango_area_mm2 > 0:
                        # Display results efficiently
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            st.image(total_mask, caption="Total Mango Area", width=150)  # Smaller for cloud
                        with col2:
                            st.image(lesion_mask, caption="Lesion Areas", width=150)
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
                        
                        if st.button("✅ Add Sample", type="primary"):
                            st.session_state.samples.append(result)
                            st.success("Sample added!")
                            # Clean up memory aggressively
                            del total_mask, lesion_mask, mask
                            aggressive_cleanup()
                            safe_rerun()
                    else:
                        st.warning("No mango detected. Adjust your selection.")
                        
                except Exception as e:
                    aggressive_cleanup()
                    st.error(f"Analysis error: {str(e)}")
            
            # Sample management - ALL FEATURES PRESERVED
            if st.session_state.samples:
                st.markdown("### 📊 All Samples")
                df = pd.DataFrame(st.session_state.samples)
                st.dataframe(df, use_container_width=True)
                
                # Summary statistics
                if len(st.session_state.samples) > 1:
                    avg_lesion = df["Lesion %"].mean()
                    st.metric("Average Lesion %", f"{avg_lesion:.1f}%")
                
                # Management controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Clear All", help="Remove all samples", use_container_width=True):
                        st.session_state.samples = []
                        aggressive_cleanup()
                        safe_rerun()
                        
                with col2:
                    # Delete specific sample
                    if st.session_state.samples:
                        sample_to_delete = st.selectbox(
                            "Delete specific sample:",
                            options=[f"Sample {i}" for i in range(1, len(st.session_state.samples) + 1)],
                            key="delete_sample_select"
                        )
                        if st.button("Delete Selected", use_container_width=True):
                            sample_idx = int(sample_to_delete.split()[1]) - 1
                            st.session_state.samples.pop(sample_idx)
                            # Renumber remaining samples
                            for i, sample in enumerate(st.session_state.samples):
                                sample["Sample"] = i + 1
                            aggressive_cleanup()
                            safe_rerun()
                
                with col3:
                    # Download results with custom filename - FEATURE PRESERVED
                    custom_filename = st.text_input(
                        "Filename:",
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
                            "Download CSV",
                            csv,
                            f"{safe_filename}.csv",
                            "text/csv",
                            help="Download analysis results",
                            use_container_width=True
                        )
                    else:
                        st.info("Enter filename")
        else:
            st.info("👆 Draw a line on the scale bar and set its real length")
            
    except Exception as e:
        aggressive_cleanup()
        st.error(f"Error: {str(e)}")
        st.info("Try refreshing the page or use a smaller image (< 2MB)")
    finally:
        # Final cleanup
        aggressive_cleanup()

# Cloud resource monitoring
if st.sidebar.button("💾 Memory Status"):
    try:
        import sys
        st.sidebar.info(f"Python: {sys.version}")
        st.sidebar.info(f"Samples: {len(st.session_state.samples)}")
        st.sidebar.info(f"Last cleanup: {int(time.time() - st.session_state.last_cleanup)}s ago")
        if st.sidebar.button("🧹 Force Cleanup"):
            aggressive_cleanup()
            st.sidebar.success("Cleanup completed")
    except Exception:
        st.sidebar.info("Memory info unavailable")

st.markdown("---")
st.markdown("🔬 **Plant Pathology Lab, UPLB** | Contact: jsmendoza5@up.edu.ph")