import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# =========================
# CONFIG & STYLING
# =========================
API_URL_DISPLAY = "http://localhost:8000/docs"  # URL shown in UI
API_URL = "http://localhost:8000/predict"       # Actual API endpoint

st.set_page_config(
    page_title="Rice Grain Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "🌾 Rice Grain Variety Classification System",
        "Get Help": "https://github.com/your-repo",
        "Report a bug": "https://github.com/your-repo/issues",
    }
)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        background: linear-gradient(135deg, #1f7a1f 0%, #2d9e3d 100%);
        color: white;
        padding: 30px 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-title h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: bold;
    }
    
    .main-title p {
        margin: 10px 0 0 0;
        font-size: 1.1em;
        opacity: 0.9;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1f7a1f;
    }
    
    /* Prediction card styling */
    .prediction-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-top: 3px solid #2d9e3d;
        transition: transform 0.2s;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
    }
    
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
    }
    
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Section headers */
    .section-header {
        color: #1f7a1f;
        border-bottom: 2px solid #1f7a1f;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    /* Upload area */
    .upload-section {
        background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%);
        padding: 30px;
        border-radius: 10px;
        border: 2px dashed #1f7a1f;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Success message */
    .success-banner {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 15px 20px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 20px;
    }
    
    /* Sidebar header styling */
    .sidebar-header {
        font-size: 1.25em;
        font-weight: bold;
        text-decoration: underline;
        color: #1f7a1f;
        margin-bottom: 0.3em;
        line-height: 1.1;
    }
    
    /* Pagination styling */
    .pagination-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        margin: 15px 0;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 8px;
    }
    
    /* Green button styling */
    .stButton button {
        background: linear-gradient(135deg, #1f7a1f 0%, #2d9e3d 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #2d9e3d 0%, #1f7a1f 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
    }
    
    .stButton button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    api_url = st.text_input("API URL", value=API_URL_DISPLAY)
    if api_url != API_URL_DISPLAY:  # Only update if user changed it
        API_URL = api_url
    
    st.markdown("---")
    st.markdown("### <div class='sidebar-header'>About</div>", unsafe_allow_html=True)
    st.markdown("""
This application classifies rice grain varieties using:
- **YOLO v8**: Grain detection & segmentation
- **CNN Models**: Variety classification (EfficientNetV2S, MobileNetV2, ResNet18)
- **XGBoost**: Meta-model ensembling

**Supported varieties:**
- 1509
- IRRI-6
- Super White
""")
    
    st.markdown("---")
    st.markdown("### <div class='sidebar-header'>Tips</div>", unsafe_allow_html=True)
    st.markdown("""
- Use clear, well-lit images
- Multiple grains improve accuracy
- High-resolution images work best
""")

# =========================
# MAIN HEADER
# =========================
st.markdown("""
    <div class="main-title">
        <h1>🌾 Rice Grain Classification</h1>
        <p>Advanced AI-powered variety classification system</p>
    </div>
""", unsafe_allow_html=True)

# =========================
# INTRODUCTION
# =========================
intro_col1, intro_col2 = st.columns([2, 1])
with intro_col1:
    st.markdown("""
    Welcome to the **Rice Grain Variety Classifier**! This intelligent system analyzes images to:
    -  Detect individual grains using advanced computer vision
    -  Classify each grain's variety with high accuracy
    -  Provide comprehensive statistics and confidence metrics
    """)

with intro_col2:
    st.info("**V1.0** - Fast & Accurate Classification")

# =========================
# UPLOAD IMAGE SECTION
# =========================
st.markdown('<div class="section-header"><h2>Upload Your Image</h2></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a rice grain image",
    type=["jpg", "jpeg", "png", "heic"],
    help="Upload a clear image of rice grains for analysis"
)

if uploaded_file is not None:
    # Create two columns for image display and info
    img_col, info_col = st.columns([1, 1])
    
    with img_col:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", width=400)
    
    with info_col:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**📋 File Information**")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        st.write(f"**Format:** {uploaded_file.type}")
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # PROCESS IMAGE
    # =========================
    process_col1, process_col2 = st.columns([1, 4])
    
    with process_col1:
        process_button = st.button("Analyze Image", key="process", use_container_width=True)
    
    if process_button or "processed_data" in st.session_state:
        if process_button:
            with st.spinner("🔄 Processing your image... This may take a moment..."):
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                
                try:
                    response = requests.post(API_URL, files=files, timeout=60)
                    
                    if response.status_code == 200:
                        st.session_state.processed_data = response.json()
                    else:
                        st.error(f"❌ API Error: {response.text}")
                        st.stop()
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection Error: {str(e)}")
                    st.error("⚠️ Make sure the API server is running at " + API_URL)
                    st.stop()
        
        if "processed_data" in st.session_state:
            data = st.session_state.processed_data
            
            # Success message
            st.markdown("""
                <div class="success-banner">
                    <strong>✅ Analysis Complete!</strong> Results are ready below.
                </div>
            """, unsafe_allow_html=True)
            
            # =========================
            # SUMMARY SECTION
            # =========================
            st.markdown('<div class="section-header"><h2>📊 Summary Statistics</h2></div>', unsafe_allow_html=True)
            
            # Key metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="🌾 Total Grains Detected",
                    value=data["num_grains"],
                    delta=None,
                    delta_color="off"
                )
            
            with metric_col2:
                total_predictions = len(data["predictions"])
                st.metric(
                    label="Predictions Made",
                    value=total_predictions,
                    delta=None,
                    delta_color="off"
                )
            
            with metric_col3:
                confusion_rate = (1 - len(data["predictions"]) / max(data["num_grains"], 1)) * 100
                st.metric(
                    label="Classification Rate",
                    value=f"{(len(data['predictions']) / max(data['num_grains'], 1) * 100):.1f}%",
                    delta=None,
                    delta_color="off"
                )
            
            # Class distribution
            st.markdown("###Class Distribution")
            
            dist_col1, dist_col2 = st.columns([1, 1])
            
            with dist_col1:
                # Pie chart
                summary_data = data["summary"]
                fig_pie = go.Figure(data=[
                    go.Pie(
                        labels=list(summary_data.keys()),
                        values=list(summary_data.values()),
                        marker=dict(colors=['#1f7a1f', '#2d9e3d', '#4cb050']),
                        textposition='inside',
                        textinfo='label+percent'
                    )
                ])
                fig_pie.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with dist_col2:
                # Bar chart
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=list(summary_data.keys()),
                        y=list(summary_data.values()),
                        marker=dict(color='#2d9e3d'),
                        text=list(summary_data.values()),
                        textposition='outside',
                    )
                ])
                fig_bar.update_layout(
                    height=400,
                    xaxis_title="Rice Variety",
                    yaxis_title="Count",
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # =========================
            # DETAILED PREDICTIONS
            # =========================
            st.markdown('<div class="section-header"><h2>🔍 Detailed Predictions</h2></div>', unsafe_allow_html=True)
            
            # Sorting and filtering options
            filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
            
            with filter_col1:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Confidence (↓)", "Confidence (↑)", "Class (A→Z)"],
                    key="sort"
                )
            
            with filter_col2:
                min_confidence = st.slider(
                    "Min. Confidence",
                    0.0, 1.0, 0.0,
                    step=0.05,
                    key="confidence_slider"
                )
            
            # Process predictions
            predictions = data["predictions"]
            
            # Sort predictions
            if sort_by == "Confidence (↓)":
                predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
            elif sort_by == "Confidence (↑)":
                predictions = sorted(predictions, key=lambda x: x['confidence'])
            else:
                predictions = sorted(predictions, key=lambda x: x['class'])
            
            # Filter by confidence
            predictions = [p for p in predictions if p['confidence'] >= min_confidence]
            
            st.write(f"Showing {len(predictions)} prediction(s)")
            
            # Pagination logic
            items_per_page = 12  # Show 12 images per page (3 rows of 4)
            total_pages = (len(predictions) + items_per_page - 1) // items_per_page
            
            # Initialize current_page
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 1
            current_page = st.session_state.current_page
            
            if total_pages > 1:
                # Page navigation
                st.markdown('<div class="pagination-container">', unsafe_allow_html=True)
                nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 1])
                
                with nav_col1:
                    if st.button("Previous", disabled=(current_page == 1), key="prev_page"):
                        st.session_state.current_page = max(1, current_page - 1)
                        st.rerun()
                
                with nav_col2:
                    if st.button("Next", disabled=(current_page == total_pages), key="next_page"):
                        st.session_state.current_page = min(total_pages, current_page + 1)
                        st.rerun()
                
                with nav_col3:
                    selected_page = st.selectbox(
                        "Go to page",
                        range(1, total_pages + 1),
                        index=current_page - 1,
                        key="page_selector",
                        label_visibility="collapsed"
                    )
                    if selected_page != current_page:
                        st.session_state.current_page = selected_page
                        st.rerun()
                
                with nav_col4:
                    st.write(f"**{current_page}/{total_pages}**")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.session_state.current_page = 1
                current_page = 1
            
            # Calculate start and end indices for current page
            start_idx = (current_page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(predictions))
            page_predictions = predictions[start_idx:end_idx]
            
            # Display page info
            if total_pages > 1:
                st.write(f"**Page {current_page} of {total_pages}** (showing grains {start_idx + 1}-{end_idx} of {len(predictions)})")
            elif len(predictions) <= items_per_page:
                st.write("**All predictions shown below**")
            
            # Display predictions in grid
            cols = st.columns(4)
            
            for i, pred in enumerate(page_predictions):
                col = cols[i % 4]
                
                confidence = pred['confidence']
                if confidence >= 0.8:
                    conf_badge = "confidence-high"
                    conf_text = "High"
                elif confidence >= 0.6:
                    conf_badge = "confidence-medium"
                    conf_text = "Medium"
                else:
                    conf_badge = "confidence-low"
                    conf_text = "Low"
                
                with col:
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.image(pred["image"], use_container_width=True)
                    
                    # Variety
                    st.markdown(f"### {pred['class']}")
                    
                    # Confidence bar
                    st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                    
                    # Badge
                    st.markdown(
                        f'<span class="confidence-badge confidence-{conf_badge.split("-")[1]}">{conf_text} ({confidence:.2f})</span>',
                        unsafe_allow_html=True
                    )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Show empty state
    st.markdown("""
    <div style="text-align: center; padding: 40px;">
        <h3>👆 Start by uploading an image above</h3>
        <p>Supported formats: JPG, JPEG, PNG, HEIC</p>
    </div>
    """, unsafe_allow_html=True)