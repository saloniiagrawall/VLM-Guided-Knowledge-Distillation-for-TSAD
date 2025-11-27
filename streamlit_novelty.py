import streamlit as st
import json
import os
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Anomaly Detection Results",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Anomaly Detection Results</h1>', unsafe_allow_html=True)

# Configuration
SAVE_DIR = "checkpoints_multiscale_adaptive_3"

@st.cache_data
def load_results(save_dir):
    """Load results.json file"""
    results_path = os.path.join(save_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def get_available_files(save_dir):
    """Get list of processed files with their corresponding images"""
    if not os.path.exists(save_dir):
        return []
    
    files = []
    for file in os.listdir(save_dir):
        if file.endswith("_visualization.png"):
            base_name = file.replace("_visualization.png", "")
            files.append(base_name + ".txt")
    return sorted(files)

def load_image(save_dir, filename):
    """Load visualization image for a specific file"""
    img_path = os.path.join(save_dir, filename.replace(".txt", "_visualization.png"))
    if os.path.exists(img_path):
        return Image.open(img_path)
    return None

def display_metrics(precision, recall, f1, num_anomalies, threshold):
    """Display formatted metrics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Precision", f"{precision:.4f}")
    with col2:
        st.metric("Recall", f"{recall:.4f}")
    with col3:
        st.metric("F1-Score", f"{f1:.4f}")
    with col4:
        st.metric("Detected Anomalies", num_anomalies)
    with col5:
        st.metric("Best Threshold", f"{threshold}th")

# Check if results exist
if not os.path.exists(SAVE_DIR):
    st.warning(f"Results directory '{SAVE_DIR}' not found. Please run the training script first.")
    st.info("""
    ### Instructions:
    1. Run the training cells in the notebook to generate results
    2. Results will be saved to the `checkpoints_multiscale_adaptive_3` directory
    3. Refresh this page to view the results
    """)
    st.stop()

# Load results
results_data = load_results(SAVE_DIR)
available_files = get_available_files(SAVE_DIR)

if not available_files and not results_data:
    st.warning("No results found yet. Please wait for training to complete.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    if results_data and 'config' in results_data:
        config = results_data['config']
        st.subheader("Model Configuration")
        st.write(f"**Multi-scale Windows:** {config.get('multi_scale_windows', 'N/A')}")
        st.write(f"**Scale Weights:** {config.get('scale_weights', 'N/A')}")
        st.write(f"**Thresholds Tested:** {config.get('thresholds', 'N/A')}")
        st.write(f"**Training Epochs:** {config.get('num_epochs', 'N/A')}")
        st.write(f"**Num Prototypes:** {config.get('num_prototypes', 'N/A')}")
    
    st.divider()
    
    st.subheader("Files Processed")
    st.write(f"**Total Files:** {len(available_files)}")

# Main content
if available_files:
    st.header("Individual File Results")
    
    # File selector
    selected_file = st.selectbox("Select a file to view:", available_files, index=0)
    
    if selected_file:
        st.divider()
        
        # Display file name
        st.subheader(f"{selected_file}")
        
        # Find corresponding results
        file_result = None
        if results_data and 'results' in results_data:
            file_result = next((r for r in results_data['results'] if r['file'] == selected_file), None)
        
        if file_result:
            # Display metrics
            st.markdown("### Performance Metrics")
            display_metrics(
                file_result['precision'],
                file_result['recall'],
                file_result['f1'],
                file_result.get('num_anomalies', 0),
                file_result['best_threshold']
            )
            
            st.divider()
            
            # Additional details
            st.markdown("### Additional Details")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Precision (detailed):** {file_result['precision']:.6f}")
                st.write(f"**Recall (detailed):** {file_result['recall']:.6f}")
            with col2:
                st.write(f"**F1-Score (detailed):** {file_result['f1']:.6f}")
                st.write(f"**Best Threshold:** {file_result['best_threshold']}th percentile")
            
            # Model checkpoint info
            model_path = os.path.join(SAVE_DIR, selected_file.replace(".txt", "_model.pth"))
            if os.path.exists(model_path):
                st.success(f"Model checkpoint available: `{os.path.basename(model_path)}`")
            
            st.divider()
        
        # Display visualization
        st.markdown("### Time Series Visualization")
        img = load_image(SAVE_DIR, selected_file)
        if img:
            st.image(img, use_column_width=True, caption=f"Anomaly Detection - {selected_file}")
        else:
            st.warning("Visualization image not found for this file.")
else:
    st.warning("No processed files found yet.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Multi-Scale Anomaly Detection using Vision Transformers</p>
    <p>Dataset: Server Machine Dataset (SMD)</p>
</div>
""", unsafe_allow_html=True)