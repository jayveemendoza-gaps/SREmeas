import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import gc
import sys
from io import BytesIO

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("🥭 Mango Lesion Analyzer")

# Initialize session state
if "samples" not in st.session_state:
    st.session_state.samples = []
if "mm_per_px" not in st.session_state:
    st.session_state.mm_per_px = None

# Streamlit Cloud optimized settings
MAX_IMAGE_SIZE = 400  # Smaller for cloud deployment
CANVAS_SIZE = 600     # Smaller canvas

@st.cache_data(max_entries=3)  # Limit cache size
def process_uploaded_image(uploaded_file, max_dim=MAX_IMAGE_SIZE):
    """Process uploaded image with aggressive optimization for cloud deployment"""
    try:
        # Create BytesIO object from uploaded file
        image_bytes = BytesIO(uploaded_file)
        image = Image.open(image_bytes).convert("RGB")
        original_size = image.size
        
        # Aggressive downscaling for cloud
        scale = min(max_dim / image.height, max_dim / image.width, 1.0)
        if scale < 1.0:
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)
        
        # Convert to numpy only when needed
        image_np = np.array(image)
        
        # Clear PIL image from memory
        del image
        gc.collect()
        
        return image_np, original_size, scale
    except Exception as e:
        st.error(f"Image processing failed: {str(e)}")
        return None, None, None

def quick_color_analysis(image_np, mask, mm_per_px):
    """Simplified color analysis for cloud deployment"""
    try:
        if image_np is None or mask is None or mm_per_px is None:
            return 0, 0, 0, None, None
        
        # Quick mask validation
        if mask.size == 0 or np.max(mask) == 0:
            return 0, 0, 0, None, None
        
        # Resize mask to match image if needed
        if mask.shape != image_np.shape[:2]:
            mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Simplified HSV conversion
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Simplified color ranges for faster processing
        healthy_mask = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([80, 255, 255]))
        lesion_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([30, 255, 100]))
        
        # Apply ROI mask
        healthy_mask = cv2.bitwise_and(healthy_mask, mask)
        lesion_mask = cv2.bitwise_and(lesion_mask, mask)
        total_mango_mask = cv2.bitwise_or(healthy_mask, lesion_mask)
        
        # Quick calculations
        mango_area_px = np.sum(total_mango_mask > 0)
        lesion_area_px = np.sum(lesion_mask > 0)
        
        if mango_area_px == 0:
            return 0, 0, 0, None, None
        
        mango_area_mm2 = mango_area_px * (mm_per_px ** 2)
        lesion_area_mm2 = lesion_area_px * (mm_per_px ** 2)
        lesion_percent = (lesion_area_mm2 / mango_area_mm2 * 100) if mango_area_mm2 > 0 else 0
        
        # Clean up
        del hsv, healthy_mask
        gc.collect()
        
        return mango_area_mm2, lesion_area_mm2, lesion_percent, total_mango_mask, lesion_mask
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return 0, 0, 0, None, None

# File uploader with size limit warning
uploaded_file = st.file_uploader(
    "Upload mango image (JPG/PNG)", 
    type=["png", "jpg", "jpeg"],
    help="For best performance, use images smaller than 2MB"
)

# Add compatibility function for rerun
def safe_rerun():
    """Safe rerun function that works with different Streamlit versions"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            st.write("Please refresh the page manually")

if uploaded_file:
    # Check file size
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
    if file_size > 5:
        st.warning(f"Large file ({file_size:.1f}MB) - processing may be slow on cloud deployment")
    
    try:
        # Process image - pass the file bytes correctly
        with st.spinner("Processing image..."):
            image_np, original_size, scale = process_uploaded_image(uploaded_file.getvalue())
        
        if image_np is None:
            st.error("Failed to process image. Try a smaller file.")
            st.stop()
        
        h, w = image_np.shape[:2]
        st.info(f"Image processed: {w}x{h} pixels (scale: {scale:.2f})")
        
        # Calculate display size
        display_scale = min(CANVAS_SIZE / w, CANVAS_SIZE / h, 1.0)
        display_w = int(w * display_scale)
        display_h = int(h * display_scale)
        
        # Convert to PIL for display
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
        
        scale_px = None
        if scale_canvas.json_data and scale_canvas.json_data.get("objects"):
            try:
                obj = scale_canvas.json_data["objects"][-1]
                if obj.get("type") == "line":
                    dx = obj["x2"] - obj["x1"]
                    dy = obj["y2"] - obj["y1"]
                    scale_px = np.sqrt(dx*dx + dy*dy)
                    # Adjust for display scaling
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
            
            drawing_mode = st.radio(
                "Drawing mode:",
                ["circle", "rect", "transform"],
                horizontal=True,
                help="Circle recommended for mangoes. Use transform to adjust existing shapes."
            )
            
            if drawing_mode == "transform":
                st.info("🔄 **Transform Mode**: Click and drag existing shapes to move them, or drag corners/edges to resize.")
            else:
                st.info(f"✏️ **{drawing_mode.title()} Mode**: Draw a new {drawing_mode}. Switch to 'transform' mode to adjust existing shapes.")
            
            canvas_result = st_canvas(
                fill_color="rgba(255,165,0,0.2)",
                stroke_width=2,
                stroke_color="rgba(255,165,0,1)",
                background_image=display_image,
                update_streamlit=True,
                height=display_h,
                width=display_w,
                drawing_mode=drawing_mode,
                key="mango_canvas",
            )
            
            # Process selection
            if canvas_result.image_data is not None and np.any(canvas_result.image_data[:,:,3] > 0):
                try:
                    # Create mask
                    mask = canvas_result.image_data[:,:,3] > 0
                    mask = mask.astype(np.uint8) * 255
                    
                    # Scale mask to match processing image
                    if display_scale < 1.0:
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Analyze
                    with st.spinner("Analyzing..."):
                        mango_area_mm2, lesion_area_mm2, lesion_percent, total_mask, lesion_mask = quick_color_analysis(
                            image_np, mask, st.session_state.mm_per_px
                        )
                    
                    if mango_area_mm2 > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(total_mask, caption="Total Mango", width=200)
                        with col2:
                            st.image(lesion_mask, caption="Lesions", width=200)
                        
                        # Results
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
                            # Clean up masks
                            del total_mask, lesion_mask
                            gc.collect()
                            safe_rerun()
                    else:
                        st.warning("No mango detected. Adjust selection.")
                        
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
            
            # Show all samples
            if st.session_state.samples:
                st.markdown("### 📊 All Samples")
                df = pd.DataFrame(st.session_state.samples)
                st.dataframe(df, use_container_width=True)
                
                # Summary stats
                if len(st.session_state.samples) > 1:
                    avg_lesion = df["Lesion %"].mean()
                    st.metric("Average Lesion %", f"{avg_lesion:.1f}%")
                
                # Management
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Clear All"):
                        st.session_state.samples = []
                        safe_rerun()
                with col2:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        "mango_analysis.csv",
                        "text/csv"
                    )
        else:
            st.info("👆 Draw a line on the scale bar and set its real length")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Try refreshing or use a smaller image")
    finally:
        # Aggressive cleanup for cloud deployment
        if 'image_np' in locals():
            del image_np
        if 'display_image' in locals():
            del display_image
        gc.collect()

# Resource info for cloud deployment
if st.sidebar.button("💾 Memory Info"):
    try:
        process = __import__('psutil').Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        st.sidebar.info(f"Memory: {memory_mb:.1f} MB")
    except:
        st.sidebar.info("Memory info unavailable")

st.markdown("---")
st.markdown("🔬 **Plant Pathology Lab, UPLB** | Contact: jsmendoza5@up.edu.ph")