# 🥭 Mango Lesion Analyzer

A Streamlit-based application for analyzing and measuring mango fruit lesions. This tool provides an interactive interface for drawing lesion boundaries, measuring lesion characteristics, and generating detailed analysis reports.

## Features

- **Image Upload**: Upload mango fruit images for analysis
- **Interactive Canvas**: Draw and edit lesion boundaries with a drawable canvas
- **Measurement Tools**: Measure lesion dimensions and area
- **Calibration**: Set scale/calibration for accurate measurements
- **Multiple Lesions**: Analyze multiple lesion samples in a single session
- **Export Results**: Download analysis data as CSV
- **Memory Optimized**: Cloud-optimized for stable performance on Streamlit Cloud
- **Real-time Processing**: Quick feedback with concurrent processing

## Installation

### Local Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/streamlit-mango-analyzer.git
cd streamlit-mango-analyzer
```

2. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Streamlit Cloud Deployment

1. Fork this repository on GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and select your forked repository
4. Configure the deployment settings
5. The app will be live at `https://yourusername-streamlit-mango-analyzer.streamlit.app`

## Usage

### Run Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Upload Image**: Start by uploading a mango image using the file uploader
2. **Set Calibration**: Use the calibration tools to establish accurate measurements
3. **Draw Lesions**: Use the canvas to mark and outline lesions
4. **Analyze**: The app will automatically calculate lesion metrics
5. **Export**: Download results as CSV for further analysis

## Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`

Core packages:
- **streamlit** - Web app framework
- **streamlit-drawable-canvas** - Interactive drawing tool
- **opencv-python-headless** - Image processing
- **numpy** - Numerical computations
- **pandas** - Data handling
- **Pillow** - Image manipulation
- **rembg** - Background removal

## System Requirements

### Minimum (Local)
- 4GB RAM
- 500MB disk space
- Python 3.8+

### Recommended (Streamlit Cloud)
- Standard Streamlit Cloud free tier is sufficient
- Note: Large images (>4MB) may be reduced in size automatically

## Performance Optimization

This application is optimized for Streamlit Cloud with:
- Reduced image resolution handling
- Memory cleanup intervals
- Session state management
- Canvas error recovery
- Throttled updates for stable performance

## Resources

📚 **Learning Materials & Data**
- Access shared resources: [Google Drive Resources](https://drive.google.com/drive/folders/15NnfIcOXAoV7mg3cevWrUFra0-Kj5DH-)

## Support & Contact

📧 **For inquiries and technical support:**
- Email: jsmendoza5@up.edu.ph

## Troubleshooting

### Common Issues

**"ModuleNotFoundError" when running app**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

**Slow performance or crashes**
- Reduce image size before uploading
- Limit to 12 or fewer samples per session
- Close other applications to free up memory

**Canvas not responding**
- Refresh the page in your browser
- Clear browser cache
- Try with a smaller image

## Development

To modify or extend the application:

1. Make changes to `app.py`
2. Test locally with `streamlit run app.py`
3. Commit and push to trigger Streamlit Cloud redeploy (if deployed)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Provide details about your environment and the issue

## Author

Created for agricultural research and crop quality assessment.

## Version History

- **1.0** - Initial Streamlit Cloud optimized release

## Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Drawing canvas powered by [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas)
- Image processing with [OpenCV](https://opencv.org/)
