import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Page config
st.set_page_config(
    page_title="Cake AI Classifier",
    page_icon="🍰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .upload-box {
        border: 3px dashed #FF6B6B;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: #FFF5F5;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load MobileNetV2 model
@st.cache_resource
def load_model():
    model_path = "best_cake_model_mobilenet.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    model = tf.keras.models.load_model(model_path)
    return model

# Class names (update if needed)
CLASS_NAMES = ["cheesecake", "chocolate", "extra_cakes", "not_cake", "red_velvet"]

def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_cake(image, model):
    img = preprocess_image(image)
    preds = model.predict(img)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx]
    class_probs = dict(zip(CLASS_NAMES, preds[0]))
    return CLASS_NAMES[pred_idx], confidence, class_probs

def main():
    st.markdown('<h1 class="main-header">🍰 Cake AI Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Powered by MobileNetV2 (84.7% Accuracy)</p>', unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Cake Image</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a cake image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a cake image to classify"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 Classify Cake", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, class_probs = predict_cake(image, model)
                    st.session_state.prediction = {
                        'class': predicted_class,
                        'confidence': confidence,
                        'probabilities': class_probs
                    }
                    st.rerun()

    with col2:
        st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
        if 'prediction' in st.session_state:
            pred = st.session_state.prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🍰 Predicted Cake</h3>
                <h2 style="font-size: 2.5rem; margin: 1rem 0;">{pred['class'].replace('_', ' ').title()}</h2>
                <h3 style="font-size: 1.5rem; margin: 1rem 0;">Confidence: {pred['confidence']*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(pred['confidence']))
            st.markdown("### 📊 All Class Probabilities")
            for class_name, prob in pred['probabilities'].items():
                if class_name == pred['class']:
                    st.markdown(f"**🥇 {class_name.replace('_', ' ').title()}:** {prob*100:.1f}%")
                else:
                    st.markdown(f"**{class_name.replace('_', ' ').title()}:** {prob*100:.1f}%")
        else:
            st.markdown("""
            <div class="upload-box">
                <h3>📤 No Image Uploaded</h3>
                <p>Upload a cake image on the left to see predictions!</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🍰 Cake AI Classifier | MobileNetV2 Model | 84.7% Accuracy</p>
        <p>Built with Streamlit & TensorFlow/Keras</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# Load the trained model and components
@st.cache_resource
def load_model():
    """Load the trained Random Forest model and components"""
    try:
        models_dir = "traditional_models"
        
        # Load Random Forest model
        with open(os.path.join(models_dir, "random_forest.pkl"), "rb") as f:
            model = pickle.load(f)
        
        # Load scaler
        with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        
        # Load label encoder
        with open(os.path.join(models_dir, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Feature extraction functions (same as training)
def extract_color_features(image):
    """Extract color histogram features from PIL image"""
    try:
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Color histograms for each channel
        features = []
        for i in range(3):
            hist = cv2.calcHist([img_cv], [i], None, [256], [0, 256])
            features.extend(hist.flatten())
        
        return np.array(features, dtype=np.float32)
    except Exception as e:
        st.error(f"Error extracting color features: {e}")
        return np.zeros(768)

def extract_texture_features(image):
    """Extract texture features using Local Binary Patterns"""
    try:
        # Convert to grayscale
        img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Resize for consistency
        img_gray = cv2.resize(img_gray, (64, 64))
        
        # Local Binary Patterns
        lbp = cv2.cornerHarris(img_gray.astype(np.float32), blockSize=2, ksize=3, k=0.04)
        lbp = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX)
        
        # Histogram of LBP
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        return hist.astype(np.float32)
    except Exception as e:
        st.error(f"Error extracting texture features: {e}")
        return np.zeros(256)

def extract_shape_features(image):
    """Extract shape and edge features"""
    try:
        # Convert to grayscale
        img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Resize for consistency
        img_gray = cv2.resize(img_gray, (100, 100))
        
        # Edge detection
        edges = cv2.Canny(img_gray, 50, 150)
        
        # Contour features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Area and perimeter
            areas = [cv2.contourArea(c) for c in contours]
            perimeters = [cv2.arcLength(c, True) for c in contours]
            
            # Shape features
            max_area = max(areas) if areas else 0
            max_perimeter = max(perimeters) if perimeters else 0
            
            # Compactness
            compactness = (max_perimeter ** 2) / (4 * np.pi * max_area) if max_area > 0 else 0
            
            # Edge density
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            features = [max_area, max_perimeter, compactness, edge_density]
        else:
            features = [0, 0, 0, 0]
        
        # Pad to 100 features
        features.extend([0] * (100 - len(features)))
        return np.array(features, dtype=np.float32)
    except Exception as e:
        st.error(f"Error extracting shape features: {e}")
        return np.zeros(100)

def extract_combined_features(image):
    """Extract all features and combine them"""
    color_features = extract_color_features(image)
    texture_features = extract_texture_features(image)
    shape_features = extract_shape_features(image)
    
    # Combine all features
    combined = np.concatenate([color_features, texture_features, shape_features])
    return combined

def predict_cake(image, model, scaler, label_encoder):
    """Predict cake class and confidence"""
    try:
        # Extract features
        features = extract_combined_features(image)
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get class name and confidence
        class_name = label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        # Get all class probabilities
        all_classes = label_encoder.classes_
        class_probs = dict(zip(all_classes, probabilities))
        
        return class_name, confidence, class_probs
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🍰 Cake AI Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Powered by Random Forest (83.46% Accuracy)</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, label_encoder = load_model()
    
    if model is None:
        st.error("❌ Model could not be loaded. Please check if the model files exist.")
        return
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Model Information")
    st.sidebar.markdown(f"**Model Type:** Random Forest")
    st.sidebar.markdown(f"**Accuracy:** 83.46%")
    st.sidebar.markdown(f"**Classes:** {len(label_encoder.classes_)}")
    
    st.sidebar.markdown("## 📊 Class Distribution")
    class_counts = {
        "Cheesecake": 82,
        "Chocolate": 80, 
        "Extra Cakes": 15,
        "Red Velvet": 77
    }
    
    # Create pie chart
    fig_pie = px.pie(
        values=list(class_counts.values()),
        names=list(class_counts.keys()),
        title="Training Data Distribution"
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.sidebar.plotly_chart(fig_pie, use_container_width=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Cake Image</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a cake image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a cake image to classify"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add some spacing
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Predict button
            if st.button("🔍 Classify Cake", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    predicted_class, confidence, class_probs = predict_cake(image, model, scaler, label_encoder)
                    
                    if predicted_class is not None:
                        # Store results in session state
                        st.session_state.prediction = {
                            'class': predicted_class,
                            'confidence': confidence,
                            'probabilities': class_probs
                        }
                        st.rerun()
    
    with col2:
        st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
        
        if 'prediction' in st.session_state:
            pred = st.session_state.prediction
            
            # Main prediction box
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🍰 Predicted Cake</h3>
                <h2 style="font-size: 2.5rem; margin: 1rem 0;">{pred['class'].replace('_', ' ').title()}</h2>
                <h3 style="font-size: 1.5rem; margin: 1rem 0;">Confidence: {pred['confidence']*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.progress(pred['confidence'])
            
            # All class probabilities
            st.markdown("### 📊 All Class Probabilities")
            
            # Create bar chart
            classes = list(pred['probabilities'].keys())
            probs = list(pred['probabilities'].values())
            
            # Highlight predicted class
            colors = ['#FF6B6B' if c == pred['class'] else '#4ECDC4' for c in classes]
            
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=classes,
                    y=probs,
                    marker_color=colors,
                    text=[f'{p*100:.1f}%' for p in probs],
                    textposition='auto',
                )
            ])
            
            fig_bar.update_layout(
                title="Class Probabilities",
                xaxis_title="Cake Types",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                showlegend=False
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Detailed results
            st.markdown("### 📋 Detailed Results")
            for class_name, prob in pred['probabilities'].items():
                if class_name == pred['class']:
                    st.markdown(f"**🥇 {class_name.replace('_', ' ').title()}:** {prob*100:.1f}%")
                else:
                    st.markdown(f"**{class_name.replace('_', ' ').title()}:** {prob*100:.1f}%")
        
        else:
            st.markdown("""
            <div class="upload-box">
                <h3>📤 No Image Uploaded</h3>
                <p>Upload a cake image on the left to see predictions!</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🍰 Cake AI Classifier | Random Forest Model | 83.46% Accuracy</p>
        <p>Built with Streamlit, OpenCV, and Scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 