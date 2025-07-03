import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import gc
from io import BytesIO

# Optimized page config
st.set_page_config(
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_title="Mango Lesion Analyzer",
    page_icon="🥭"
)

st.title("🥭 Mango Lesion Analyzer")

# Initialize session state efficiently
for key, default in [
    ("samples", []), 
    ("mm_per_px", None), 
    ("polygon_drawing", False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Cloud-optimized settings - keep canvas sizes but optimize processing
MAX_IMAGE_SIZE = 350  # Slightly reduced for speed
CANVAS_SIZE = 600     # Keep original canvas size
MAX_FILE_SIZE_MB = 8  # Reasonable file size limit

@st.cache_data(max_entries=2, ttl=600)  # Reduced cache with TTL
def process_uploaded_image(uploaded_file, max_dim=MAX_IMAGE_SIZE):
    """Optimized image processing for cloud deployment"""
    try:
        if not uploaded_file or len(uploaded_file) == 0:
            return None, None, None
            
        file_size_mb = len(uploaded_file) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File too large: {file_size_mb:.1f}MB. Use files under {MAX_FILE_SIZE_MB}MB.")
            return None, None, None
        
        # Efficient image processing
        image = Image.open(BytesIO(uploaded_file)).convert("RGB")
        original_size = image.size
        
        # Smart downscaling
        scale = min(max_dim / image.height, max_dim / image.width, 1.0)
        if scale < 1.0:
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)
            st.info(f"Resized from {original_size[0]}x{original_size[1]} to {new_size[0]}x{new_size[1]} pixels")
        
        image_np = np.array(image, dtype=np.uint8)
        del image
        gc.collect()
        
        return image_np, original_size, scale
        
    except MemoryError:
        st.error("Out of memory. Please use a smaller image.")
        return None, None, None
    except Exception as e:
        st.error(f"Image processing failed: {str(e)}")
        return None, None, None

def quick_color_analysis(image_np, mask, mm_per_px):
    """Optimized color analysis with error handling"""
    try:
        if image_np is None or mask is None or mm_per_px is None:
            return 0, 0, 0, None, None
        
        if mask.size == 0 or np.max(mask) == 0:
            return 0, 0, 0, None, None
        
        # Ensure mask matches image dimensions
        if mask.shape != image_np.shape[:2]:
            mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Efficient HSV conversion
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Optimized color range detection
        healthy_mask = cv2.inRange(hsv, (20, 30, 30), (80, 255, 255))
        lesion_mask = cv2.inRange(hsv, (0, 0, 0), (30, 255, 100))
        
        # Apply ROI mask efficiently
        healthy_mask = cv2.bitwise_and(healthy_mask, mask)
        lesion_mask = cv2.bitwise_and(lesion_mask, mask)
        total_mango_mask = cv2.bitwise_or(healthy_mask, lesion_mask)
        
        # Fast pixel counting
        mango_area_px = np.count_nonzero(total_mango_mask)
        lesion_area_px = np.count_nonzero(lesion_mask)
        
        if mango_area_px == 0:
            return 0, 0, 0, None, None
        
        # Calculate areas
        mm_per_px_sq = mm_per_px * mm_per_px
        mango_area_mm2 = mango_area_px * mm_per_px_sq
        lesion_area_mm2 = lesion_area_px * mm_per_px_sq
        lesion_percent = (lesion_area_mm2 / mango_area_mm2 * 100) if mango_area_mm2 > 0 else 0
        
        # Cleanup
        del hsv, healthy_mask
        gc.collect()
        
        return mango_area_mm2, lesion_area_mm2, lesion_percent, total_mango_mask, lesion_mask
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return 0, 0, 0, None, None

# File uploader
uploaded_file = st.file_uploader(
    "Upload mango image (JPG/PNG)", 
    type=["png", "jpg", "jpeg"],
    help=f"Max size: {MAX_FILE_SIZE_MB}MB. For best performance, use images under 5MB."
)

def safe_rerun():
    """Memory-safe rerun function"""
    gc.collect()
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            st.write("Please refresh the page manually")

if uploaded_file:
    try:
        # File size check
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size > MAX_FILE_SIZE_MB:
            st.error(f"File too large: {file_size:.1f}MB")
            st.stop()
        
        if file_size > 5:
            st.warning(f"Large file ({file_size:.1f}MB) - processing may be slower")
        
        # Process image
        with st.spinner("Processing image..."):
            image_np, original_size, scale = process_uploaded_image(uploaded_file.getvalue())
        
        if image_np is None:
            st.error("Failed to process image. Try a smaller file.")
            st.stop()
        
        h, w = image_np.shape[:2]
        st.info(f"Processing: {w}x{h} pixels (scale: {scale:.2f})")
        
        # Calculate display size - keep original canvas size
        display_scale = min(CANVAS_SIZE / w, CANVAS_SIZE / h, 1.0)
        display_w = int(w * display_scale)
        display_h = int(h * display_scale)
        
        # Create display image
        display_image = Image.fromarray(image_np)
        if display_scale < 1.0:
            display_image = display_image.resize((display_w, display_h), Image.LANCZOS)
        
        # --- Step 1: Scale setting ---
        st.markdown("## 1️⃣ Set Scale")
        st.info("Draw a line on a known measurement (ruler/scale bar)")
        
        scale_canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=3,
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
            
            # Drawing mode selection with all options
            drawing_mode = st.radio(
                "Drawing mode:",
                ["circle", "rect", "polygon", "transform"],
                horizontal=True,
                help="Circle for round mangoes, polygon for irregular shapes, transform to adjust existing shapes"
            )
            
            # Brightness adjustment
            brightness = st.slider(
                "🔆 Brightness adjustment:",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust brightness for better visibility"
            )
            
            # Apply brightness adjustment efficiently
            if brightness != 1.0:
                display_image_array = np.array(display_image)
                display_image_array = np.clip(display_image_array * brightness, 0, 255).astype(np.uint8)
                adjusted_display_image = Image.fromarray(display_image_array)
                del display_image_array
            else:
                adjusted_display_image = display_image
            
            # Mode-specific instructions
            if drawing_mode == "transform":
                st.info("🔄 **Transform Mode**: Click and drag shapes to move, drag corners/edges to resize. Analysis updates automatically.")
            elif drawing_mode == "polygon":
                st.info("📐 **Polygon Mode**: Click to place points, double-click to close polygon. Perfect for irregular mango shapes. Press transform to adjust and process image")
            else:
                st.info(f"✏️ **{drawing_mode.title()} Mode**: Draw a new {drawing_mode} around the mango.")
            
            canvas_result = st_canvas(
                fill_color="rgba(255,165,0,0.2)",
                stroke_width=2,
                stroke_color="rgba(255,165,0,1)",
                background_image=adjusted_display_image,
                update_streamlit=True,
                height=display_h,
                width=display_w,
                drawing_mode=drawing_mode,
                key="mango_canvas",
            )

            # Smart polygon detection with improved logic
            process_analysis = False
            
            if canvas_result.image_data is not None and np.any(canvas_result.image_data[:,:,3] > 0):
                if drawing_mode == "polygon":
                    # Check for closed polygon
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
                    # For other modes, process immediately
                    process_analysis = True
            
            if process_analysis:
                try:
                    # Create mask efficiently
                    mask = (canvas_result.image_data[:,:,3] > 0).astype(np.uint8) * 255
                    
                    # Scale mask to match processing image
                    if display_scale < 1.0:
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Analyze with progress indicator
                    with st.spinner("Analyzing mango..."):
                        mango_area_mm2, lesion_area_mm2, lesion_percent, total_mask, lesion_mask = quick_color_analysis(
                            image_np, mask, st.session_state.mm_per_px
                        )
                    
                    if mango_area_mm2 > 0:
                        # Display results - masks side by side with data
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            st.image(total_mask, caption="Total Mango Area", width=200)
                        with col2:
                            st.image(lesion_mask, caption="Lesion Areas", width=200)
                        with col3:
                            st.metric("Total Area", f"{mango_area_mm2:.1f} mm²")
                            st.metric("Lesion Area", f"{lesion_area_mm2:.1f} mm²")
                            st.metric("Lesion %", f"{lesion_percent:.1f}%")
                        
                        # Results table below
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
                            # Clean up memory
                            del total_mask, lesion_mask
                            gc.collect()
                            safe_rerun()
                    else:
                        st.warning("No mango detected. Adjust your selection.")
                        
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
            
            # Sample management with all features
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
                            safe_rerun()
                
                with col3:
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "mango_analysis.csv",
                        "text/csv",
                        help="Download analysis results",
                        use_container_width=True
                    )
        else:
            st.info("👆 Draw a line on the scale bar and set its real length")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Try refreshing the page or use a smaller image")
    finally:
        # Memory cleanup
        gc.collect()

# Resource monitoring (optional)
if st.sidebar.button("💾 Memory Info"):
    try:
        import sys
        import os
        # Simple memory info without psutil
        st.sidebar.info("Memory monitoring available")
        st.sidebar.info(f"Python version: {sys.version}")
    except Exception:
        st.sidebar.info("Memory info unavailable")

st.markdown("---")
st.markdown("🔬 **Plant Pathology Lab, UPLB** | Contact: jsmendoza5@up.edu.ph")
