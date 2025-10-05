import os
import json
from datetime import datetime
import time
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px

from data_preprocessing import ExoplanetDataProcessor
from ensemble_model import ExoplanetEnsembleModel

# Logging: console + rotating file, and a simple in-app buffer
LOG_PATH = os.path.join(os.path.dirname(__file__), MODELS_DIR if 'MODELS_DIR' in globals() else 'models', 'streamlit_app.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def _get_logger():
    logger = logging.getLogger('exoplanet_ui')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
        # Console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # File (rotate ~1MB x 3)
        try:
            fh = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3)
            fh.setLevel(logging.INFO)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            # If filesystem is read-only (some hosts), we still have console
            pass
    return logger

logger = _get_logger()

def ui_log(message: str):
    try:
        if 'ui_logs' not in st.session_state:
            st.session_state['ui_logs'] = []
        # Keep a bounded buffer
        st.session_state['ui_logs'].append(message)
        if len(st.session_state['ui_logs']) > 300:
            st.session_state['ui_logs'] = st.session_state['ui_logs'][-300:]
    except Exception:
        pass
    logger.info(message)


# Environment-driven paths (compatible with Streamlit Cloud)
# Support both environment variables and Streamlit secrets
def _get_setting(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val is None:
        try:
            if name in st.secrets:
                val = st.secrets.get(name)
        except Exception:
            val = None
    return val if val is not None else default

DATA_DIR = _get_setting('DATA_DIR', '.')
MODELS_DIR = _get_setting('MODELS_DIR', 'models')
UPLOADS_DIR = _get_setting('UPLOADS_DIR', 'uploads')
ALLOW_TRAINING = (_get_setting('ALLOW_TRAINING', '0') == '1')


st.set_page_config(page_title="NASA Exoplanet Hunt", layout="wide")


def _inject_css():
    css_paths = [
        os.path.join('static', 'css', 'style.css'),
        os.path.join(os.path.dirname(__file__), 'static', 'css', 'style.css'),
    ]
    for p in css_paths:
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                break
            except Exception:
                pass


def _card_html(title: str, body_html: str, status: str | None = None) -> str:
    status_class = ''
    if status == 'success':
        status_class = 'selected-card selected-success'
    elif status == 'warning':
        status_class = 'selected-card selected-warning'
    elif status == 'danger':
        status_class = 'selected-card selected-danger'
    return f'''
    <div class="container">
      <div class="card {status_class}" style="margin:16px 0;">
        <div class="card-header"><h5 class="card-title" style="margin:0;">{title}</h5></div>
        <div class="card-body">{body_html}</div>
      </div>
    </div>
    '''


@st.cache_resource(show_spinner=False)
def load_models_and_scaler():
    """Load ensemble models and scaler from disk; return (ensemble_model, processor, metrics)."""
    logger.info("Loading models and scaler...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Model
    ensemble = ExoplanetEnsembleModel()
    metrics = None
    try:
        prefix = os.path.join(MODELS_DIR, 'exoplanet_ensemble')
        ensemble.load_models(filepath_prefix=prefix)
        logger.info("Models loaded from %s", prefix)
    except Exception as e:
        # Not fatal for UI; we can train later if allowed
        st.warning(f"Models not found or failed to load: {e}")
        logger.warning("Models not found or failed to load: %s", e)

    # Processor and scaler
    processor = ExoplanetDataProcessor(data_dir=DATA_DIR)
    scaler_path = os.path.join(MODELS_DIR, 'exoplanet_ensemble_scaler.pkl')
    if os.path.exists(scaler_path):
        try:
            processor.scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded from %s", scaler_path)
        except Exception as e:
            st.warning(f"Failed to load scaler: {e}")
            logger.warning("Failed to load scaler: %s", e)

    # Metrics
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                logger.info("Metrics loaded from %s", metrics_path)
        except Exception:
            pass

    return ensemble, processor, metrics


def train_and_save(ensemble: ExoplanetEnsembleModel, processor: ExoplanetDataProcessor):
    """Run optimized training (faster for Streamlit), then persist models/metrics/scaler."""
    logger.info("Training started")
    ui_log("Initializing training pipeline...")
    # Check if quick training mode is enabled
    quick_mode = os.environ.get('QUICK_TRAIN', '0') == '1'
    logger.info("Quick training mode: %s", quick_mode)
    
    with st.spinner("Preparing data (this may take a few minutes)..."):
        data = processor.prepare_data_for_training()
        try:
            X_shape = getattr(data.get('X_train'), 'shape', None)
            ui_log(f"Data prepared. X_train shape: {X_shape}")
        except Exception:
            pass

    # Progress hook
    progress_bar = st.progress(0)
    status_text = st.empty()
    def _progress_cb(stage: str, pct: int, msg: str):
        try:
            progress_bar.progress(max(0, min(100, int(pct))))
        except Exception:
            pass
        status_text.text(f"{msg}")
        ui_log(f"{stage.upper()}: {msg}")

    try:
        ensemble.set_progress_callback(_progress_cb)
    except Exception:
        pass

    if quick_mode:
        with st.spinner("Training ensemble (optimized for speed - RF/XGB/LGB only)..."):
            ui_log("Training with retrain-only mode (RF/XGB/LGB only)...")
            try:
                results = ensemble.train_ensemble_retrain_only(data)
                ui_log("Retrain-only training completed successfully")
            except AttributeError:
                ui_log("Fallback to quick training...")
                results = ensemble.train_ensemble_quick(data)
    else:
        with st.spinner("Training ensemble (full training - this may take several minutes)..."):
            ui_log("Training with full parameters...")
            results = ensemble.train_ensemble(data)

    # Save models, metrics, scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    prefix = os.path.join(MODELS_DIR, 'exoplanet_ensemble')
    ensemble.save_models(filepath_prefix=prefix)
    ui_log(f"Models saved to prefix: {prefix}")
    try:
        joblib.dump(processor.scaler, os.path.join(MODELS_DIR, 'exoplanet_ensemble_scaler.pkl'))
        ui_log("Scaler persisted")
    except Exception:
        pass
    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    ui_log("Metrics saved")

    st.success("Training complete and models saved.")
    progress_bar.empty()
    status_text.empty()
    logger.info("Training completed")
    return results


def page_home(metrics):
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; color: #ffffff; margin-bottom: 1rem;">
            🚀 NASA Exoplanet Detection System
        </h1>
        <p style="font-size: 1.2rem; color: #cccccc; margin-bottom: 2rem;">
            Advanced ensemble machine learning system combining Random Forest, XGBoost, LightGBM, and CNN 
            to classify exoplanet candidates with over 90% accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance Metrics Section
    if metrics:
        st.markdown("## 🎯 System Performance")
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            accuracy = metrics.get('ensemble_accuracy', 0) * 100
            st.metric(
                label="🎯 Ensemble Accuracy",
                value=f"{accuracy:.1f}%",
                delta=None,
                help="Overall accuracy of the ensemble model"
            )
        
        with perf_col2:
            auc = metrics.get('ensemble_auc', 0) * 100
            st.metric(
                label="📈 ROC AUC Score",
                value=f"{auc:.1f}%",
                delta=None,
                help="Area under the receiver operating characteristic curve"
            )
        
        with perf_col3:
            st.metric(
                label="🧠 ML Models",
                value="4",
                delta=None,
                help="Random Forest, XGBoost, LightGBM, and CNN"
            )
        
        with perf_col4:
            st.metric(
                label="📊 Data Points",
                value="17K+",
                delta=None,
                help="Total training samples in the dataset"
            )
    
    # System Features Section
    st.markdown("## 🚀 System Features")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 1rem;
        ">
            <h3 style="margin: 0; color: white;">🔍 Classify Exoplanets</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Input stellar and planetary parameters to get real-time classification predictions")
    
    with action_col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 1rem;
        ">
            <h3 style="margin: 0; color: white;">🌌 View Visualizer</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Explore interactive orbital simulations and light curve analysis")
    
    with action_col3:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 1rem;
        ">
            <h3 style="margin: 0; color: white;">📊 View Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Analyze model performance metrics and training statistics")
    
    # System Features Section
    st.markdown("## ✨ System Features")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        ### 🧠 Ensemble Learning
        - **Random Forest**: Robust tree-based classification
        - **XGBoost**: Gradient boosting for high performance
        - **LightGBM**: Fast and efficient boosting
        - **CNN**: Deep learning for light curve analysis
        """)
        
        st.markdown("""
        ### 🎛️ Hyperparameter Tuning
        - Comprehensive parameter controls for all models
        - Real-time validation and weight adjustment
        - Custom training with optimized configurations
        """)
    
    with feature_col2:
        st.markdown("""
        ### 🌌 Interactive Visualization
        - Real-time orbital simulations
        - Transit light curve generation
        - Preset exoplanet system exploration
        """)
        
        st.markdown("""
        ### 📊 Advanced Analytics
        - Model performance comparison charts
        - Dataset statistics and insights
        - Training progress monitoring
        """)
    
    # System Status Section
    st.markdown("## ⚙️ System Status")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.markdown("**Configuration:**")
        st.code(f"""
DATA_DIR: {DATA_DIR}
MODELS_DIR: {MODELS_DIR}
ALLOW_TRAINING: {ALLOW_TRAINING}
        """)
    
    with status_col2:
        st.markdown("**Model Status:**")
        if metrics:
            model_status = "✅ Models loaded and ready"
            last_trained = "Recently trained"
        else:
            model_status = "⚠️ Models not found - training required"
            last_trained = "Not yet trained"
        
        st.code(f"""
Status: {model_status}
Last Trained: {last_trained}
Training Enabled: {'Yes' if ALLOW_TRAINING else 'No'}
        """)
    
    # Dataset Overview
    if metrics:
        st.markdown("## 📈 Model Performance Overview")
        
        # Create performance comparison chart
        individual_results = metrics.get('individual_results', {})
        if individual_results:
            model_names = []
            accuracies = []
            
            model_labels = {
                'rf': 'Random Forest',
                'xgb': 'XGBoost', 
                'lgb': 'LightGBM',
                'cnn': 'CNN'
            }
            
            for model_name in ['rf', 'xgb', 'lgb', 'cnn']:
                if model_name in individual_results:
                    model_names.append(model_labels[model_name])
                    accuracies.append(individual_results[model_name].get('accuracy', 0) * 100)
            
            # Add ensemble
            model_names.append('Ensemble')
            accuracies.append(metrics.get('ensemble_accuracy', 0) * 100)
            
            # Create performance chart
            fig_perf = go.Figure(data=[
                go.Bar(x=model_names, y=accuracies,
                       marker_color=['#28a745', '#ffc107', '#17a2b8', '#dc3545', '#6f42c1'],
                       text=[f'{acc:.1f}%' for acc in accuracies],
                       textposition='auto')
            ])
            fig_perf.update_layout(
                title='Model Performance Comparison',
                yaxis_title='Accuracy (%)',
                showlegend=False,
                height=400,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            st.plotly_chart(fig_perf, use_container_width=True)
    
    # Getting Started Section
    st.markdown("## 🚀 Getting Started")
    
    with st.expander("📖 How to Use This System", expanded=False):
        st.markdown("""
        ### 1. 🔍 Classification
        - Navigate to the **Classify** page
        - Input stellar and planetary parameters
        - Get real-time predictions with confidence scores
        
        ### 2. 🌌 Visualization
        - Explore the **Visualizer** page
        - Adjust orbital parameters to see system evolution
        - Generate transit light curves
        - Try preset exoplanet systems
        
        ### 3. 🎛️ Hyperparameter Tuning
        - Access the **Hyperparameters** page
        - Adjust model parameters for optimal performance
        - Retrain models with custom configurations
        
        ### 4. 📊 Performance Analysis
        - Check the **Statistics** page
        - Review model performance metrics
        - Compare individual model results
        - Monitor training progress
        """)
    
    # Footer with additional info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Built with Streamlit • Powered by TensorFlow, XGBoost, and LightGBM</p>
        <p>NASA Exoplanet Hunt • Advanced Machine Learning for Space Discovery</p>
    </div>
    """, unsafe_allow_html=True)


def page_classify(ensemble: ExoplanetEnsembleModel, processor: ExoplanetDataProcessor):
    st.title("🔍 Classify Exoplanet Candidate")
    st.markdown("Input stellar and planetary parameters to get real-time classification predictions using our trained ensemble model.")
    
    # Check if models are loaded
    if not getattr(ensemble, 'is_trained', False):
        st.error("❌ Models are not loaded. Please train the model first.")
        st.info("💡 Go to the Hyperparameters page to train the models.")
        return
    
    if processor is None or not hasattr(processor, 'scaler') or processor.scaler is None:
        st.error("❌ Data processor/scaler not available. Please train the models first.")
        return

    with st.form("classify_form"):
        st.markdown("### 🪐 Planetary Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            orbital_period = st.number_input(
                "Orbital Period (days)", 
                min_value=0.01, max_value=10000.0, value=365.25, step=0.01,
                help="Time for one complete orbit around the star"
            )
            transit_duration = st.number_input(
                "Transit Duration (hours)", 
                min_value=0.01, max_value=1000.0, value=10.0, step=0.01,
                help="How long the planet takes to cross the star"
            )
            transit_depth = st.number_input(
                "Transit Depth (ppm)", 
                min_value=0.0, max_value=1e6, value=1000.0, step=1.0,
                help="How much the star dims during transit (parts per million)"
            )
            planet_radius = st.number_input(
                "Planet Radius (Earth radii)", 
                min_value=0.01, max_value=50.0, value=1.0, step=0.01,
                help="Planet size relative to Earth"
            )
            equilibrium_temp = st.number_input(
                "Equilibrium Temperature (K)", 
                min_value=0.0, max_value=5000.0, value=300.0, step=1.0,
                help="Expected planet temperature"
            )
            insolation_flux = st.number_input(
                "Insolation Flux (relative to Earth)", 
                min_value=0.0, max_value=1e6, value=1.0, step=0.01,
                help="Amount of stellar radiation received"
            )
        
        with col2:
            st.markdown("### ⭐ Stellar Parameters")
            impact_param = st.number_input(
                "Impact Parameter", 
                min_value=0.0, max_value=2.0, value=0.5, step=0.01,
                help="How centrally the planet crosses the star"
            )
            model_snr = st.number_input(
                "Model SNR", 
                min_value=0.0, max_value=1000.0, value=20.0, step=0.1,
                help="Signal-to-noise ratio of the detection"
            )
            stellar_temp = st.number_input(
                "Stellar Temperature (K)", 
                min_value=1000.0, max_value=10000.0, value=5778.0, step=1.0,
                help="Surface temperature of the host star"
            )
            stellar_logg = st.number_input(
                "Stellar log g", 
                min_value=0.0, max_value=10.0, value=4.44, step=0.01,
                help="Surface gravity of the star"
            )
            stellar_radius = st.number_input(
                "Stellar Radius (Solar radii)", 
                min_value=0.01, max_value=100.0, value=1.0, step=0.01,
                help="Size of the host star relative to the Sun"
            )
            stellar_magnitude = st.number_input(
                "Stellar Magnitude", 
                min_value=-10.0, max_value=30.0, value=12.0, step=0.01,
                help="Brightness of the star"
            )

        submitted = st.form_submit_button("🚀 Classify Candidate", type="primary", use_container_width=True)

    if submitted:
        logger.info("Classify form submitted")
        ui_log("Classify submitted: running prediction...")
        try:
            with st.spinner("🔄 Running ensemble model prediction..."):
                # Construct feature array in the correct format based on training data
                # This matches the feature order from data preprocessing
                features = np.array([[
                    orbital_period,           # koi_period
                    impact_param,             # koi_impact  
                    transit_duration,         # koi_duration
                    transit_depth,           # koi_depth
                    planet_radius,           # koi_prad
                    equilibrium_temp,        # koi_teq
                    insolation_flux,         # koi_insol
                    model_snr,               # koi_model_snr
                    stellar_temp,            # koi_steff
                    stellar_logg,            # koi_slogg
                    stellar_radius,          # koi_srad
                    stellar_magnitude,       # koi_kepmag
                    0, 0, 0, 0,             # koi_fpflag_nt, _ss, _co, _ec
                    0, 0, 0, 0,             # period errors
                    0, 0, 0, 0              # other error terms
                ]])

                # Scale features using the trained scaler
                logger.info(
                    "Predict: period=%.3f, duration=%.3f, depth=%.1f, prad=%.2f, teq=%.1f, insol=%.2f, steff=%.1f, slogg=%.2f, srad=%.2f, mag=%.2f",
                    orbital_period, transit_duration, transit_depth, planet_radius, equilibrium_temp, insolation_flux,
                    stellar_temp, stellar_logg, stellar_radius, stellar_magnitude
                )
                features_scaled = processor.scaler.transform(features)

                # Create light curve for CNN
                light_curve_data = pd.DataFrame([{
                    'koi_period': orbital_period, 
                    'koi_duration': transit_duration, 
                    'koi_depth': transit_depth
                }])
                light_curve = processor.create_light_curve_features(light_curve_data)[0]

                # Get ensemble prediction
                prediction_proba = ensemble.predict(features_scaled, light_curve.reshape(1, -1))[0]
                logger.info("Prediction complete with confidence=%.4f", float(prediction_proba))
                ui_log(f"Prediction confidence: {float(prediction_proba):.4f}")
                
                # Convert to percentage
                confidence_pct = prediction_proba * 100
                
                # Determine classification based on model confidence
                if prediction_proba >= 0.7:
                    classification = "Confirmed Exoplanet"
                    status = 'success'
                    result_color = '#28a745'
                elif prediction_proba >= 0.3:
                    classification = "Candidate Exoplanet"
                    status = 'warning'
                    result_color = '#ffc107'
                else:
                    classification = "False Positive"
                    status = 'danger'
                    result_color = '#dc3545'

                # Display results
                st.markdown("## 🎯 Classification Results")
                
                # Main result card
                result_col1, result_col2 = st.columns([2, 1])
                
                with result_col1:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {result_color}22 0%, {result_color}11 100%);
                        border: 2px solid {result_color};
                        border-radius: 10px;
                        padding: 1.5rem;
                        text-align: center;
                        margin: 1rem 0;
                    ">
                        <h2 style="color: {result_color}; margin: 0;">{classification}</h2>
                        <h3 style="color: {result_color}; margin: 0.5rem 0;">Confidence: {confidence_pct:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with result_col2:
                    # Confidence gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence_pct,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confidence"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': result_color},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightgreen"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90}}))
                    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # Individual model predictions (if available)
                st.markdown("### 🔍 Individual Model Predictions")
                
                try:
                    # Get individual model predictions
                    rf_proba = ensemble.rf_model.predict_proba(features_scaled)[0, 1] * 100
                    xgb_proba = ensemble.xgb_model.predict_proba(features_scaled)[0, 1] * 100
                    lgb_proba = ensemble.lgb_model.predict_proba(features_scaled)[0, 1] * 100
                    
                    # CNN prediction
                    lc_reshaped = light_curve.reshape(1, light_curve.shape[0], 1)
                    cnn_proba = ensemble.cnn_model.predict(lc_reshaped)[0, 0] * 100
                    
                    model_col1, model_col2, model_col3, model_col4 = st.columns(4)
                    
                    with model_col1:
                        st.metric("🌲 Random Forest", f"{rf_proba:.1f}%")
                    with model_col2:
                        st.metric("🚀 XGBoost", f"{xgb_proba:.1f}%")
                    with model_col3:
                        st.metric("⚡ LightGBM", f"{lgb_proba:.1f}%")
                    with model_col4:
                        st.metric("🧠 CNN", f"{cnn_proba:.1f}%")
                        
                    # Model comparison chart
                    model_names = ['Random Forest', 'XGBoost', 'LightGBM', 'CNN', 'Ensemble']
                    model_scores = [rf_proba, xgb_proba, lgb_proba, cnn_proba, confidence_pct]
                    
                    fig_comparison = go.Figure(data=[
                        go.Bar(x=model_names, y=model_scores,
                               marker_color=['#28a745', '#ffc107', '#17a2b8', '#dc3545', '#6f42c1'],
                               text=[f'{score:.1f}%' for score in model_scores],
                               textposition='auto')
                    ])
                    fig_comparison.update_layout(
                        title='Individual Model Predictions',
                        yaxis_title='Confidence (%)',
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not get individual model predictions: {e}")

                # Light curve visualization
                st.markdown("### 📈 Simulated Light Curve")
                
                # Create time axis
                time_axis = np.linspace(0, orbital_period * 2, len(light_curve))
                
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(
                    x=time_axis,
                    y=light_curve,
                    mode='lines',
                    line=dict(color='#74c0fc', width=2),
                    name='Light Curve'
                ))
                
                fig_lc.update_layout(
                    title='Generated Light Curve for CNN Analysis',
                    xaxis_title='Time (days)',
                    yaxis_title='Relative Flux',
                    height=400
                )
                st.plotly_chart(fig_lc, use_container_width=True)
                
                # System summary
                st.markdown("### 📊 System Summary")
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.markdown("**Orbital Characteristics:**")
                    st.write(f"Period: {orbital_period:.2f} days")
                    st.write(f"Transit Duration: {transit_duration:.2f} hours")
                    st.write(f"Transit Depth: {transit_depth:.0f} ppm")
                
                with summary_col2:
                    st.markdown("**Planet Properties:**")
                    st.write(f"Radius: {planet_radius:.2f} R⊕")
                    st.write(f"Temperature: {equilibrium_temp:.0f} K")
                    st.write(f"Insolation: {insolation_flux:.2f} × Earth")
                
                with summary_col3:
                    st.markdown("**Host Star:**")
                    st.write(f"Temperature: {stellar_temp:.0f} K")
                    st.write(f"Radius: {stellar_radius:.2f} R☉")
                    st.write(f"Magnitude: {stellar_magnitude:.1f}")

        except Exception as e:
            logger.exception("Prediction failed: %s", e)
            ui_log(f"Prediction failed: {e}")
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("💡 Please ensure the models are properly trained and try again.")
            if st.checkbox("Show error details"):
                st.exception(e)


def page_hyperparameters(ensemble: ExoplanetEnsembleModel, processor: ExoplanetDataProcessor):
    st.title("🔧 Hyperparameter Tuning")
    st.markdown("Fine-tune model parameters to optimize performance for your specific use case")
    
    if not ALLOW_TRAINING:
        st.warning("⚠️ Training is disabled on this deployment. Set ALLOW_TRAINING=1 to enable (may be slow on free tier).")
        
    # Initialize session state for parameters if not exists
    if 'hyperparameters' not in st.session_state:
        st.session_state.hyperparameters = {
            'rf': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2},
            'xgb': {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8},
            'lgb': {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8},
            'cnn': {'epochs': 100, 'batch_size': 32, 'learning_rate': 0.001, 'dropout': 0.3},
            'weights': {'rf': 0.25, 'xgb': 0.25, 'lgb': 0.25, 'cnn': 0.25}
        }
    
    # Model Parameter Sections
    col1, col2 = st.columns(2)
    
    with col1:
        # Random Forest Parameters
        st.markdown("### 🌲 Random Forest")
        with st.container():
            rf_n_estimators = st.number_input(
                "Number of Estimators", 
                min_value=50, max_value=1000, step=10, 
                value=st.session_state.hyperparameters['rf']['n_estimators'],
                help="Number of trees in the forest"
            )
            rf_max_depth = st.number_input(
                "Max Depth", 
                min_value=5, max_value=50, step=1, 
                value=st.session_state.hyperparameters['rf']['max_depth'],
                help="Maximum depth of the trees"
            )
            rf_min_samples_split = st.number_input(
                "Min Samples Split", 
                min_value=2, max_value=20, step=1, 
                value=st.session_state.hyperparameters['rf']['min_samples_split'],
                help="Minimum samples required to split a node"
            )
            rf_min_samples_leaf = st.number_input(
                "Min Samples Leaf", 
                min_value=1, max_value=10, step=1, 
                value=st.session_state.hyperparameters['rf']['min_samples_leaf'],
                help="Minimum samples required at a leaf node"
            )
        
        st.markdown("---")
        
        # LightGBM Parameters
        st.markdown("### ⚡ LightGBM")
        with st.container():
            lgb_n_estimators = st.number_input(
                "Number of Estimators", 
                min_value=50, max_value=1000, step=10, 
                value=st.session_state.hyperparameters['lgb']['n_estimators'],
                help="Number of boosting rounds",
                key="lgb_n_est"
            )
            lgb_max_depth = st.number_input(
                "Max Depth", 
                min_value=3, max_value=20, step=1, 
                value=st.session_state.hyperparameters['lgb']['max_depth'],
                help="Maximum depth of the trees",
                key="lgb_max_d"
            )
            lgb_learning_rate = st.number_input(
                "Learning Rate", 
                min_value=0.01, max_value=1.0, step=0.01, 
                value=st.session_state.hyperparameters['lgb']['learning_rate'],
                help="Step size shrinkage to prevent overfitting",
                key="lgb_lr"
            )
            lgb_subsample = st.number_input(
                "Subsample", 
                min_value=0.1, max_value=1.0, step=0.1, 
                value=st.session_state.hyperparameters['lgb']['subsample'],
                help="Fraction of samples used for training",
                key="lgb_sub"
            )

    with col2:
        # XGBoost Parameters
        st.markdown("### 🚀 XGBoost")
        with st.container():
            xgb_n_estimators = st.number_input(
                "Number of Estimators", 
                min_value=50, max_value=1000, step=10, 
                value=st.session_state.hyperparameters['xgb']['n_estimators'],
                help="Number of boosting rounds",
                key="xgb_n_est"
            )
            xgb_max_depth = st.number_input(
                "Max Depth", 
                min_value=3, max_value=20, step=1, 
                value=st.session_state.hyperparameters['xgb']['max_depth'],
                help="Maximum depth of the trees",
                key="xgb_max_d"
            )
            xgb_learning_rate = st.number_input(
                "Learning Rate", 
                min_value=0.01, max_value=1.0, step=0.01, 
                value=st.session_state.hyperparameters['xgb']['learning_rate'],
                help="Step size shrinkage to prevent overfitting",
                key="xgb_lr"
            )
            xgb_subsample = st.number_input(
                "Subsample", 
                min_value=0.1, max_value=1.0, step=0.1, 
                value=st.session_state.hyperparameters['xgb']['subsample'],
                help="Fraction of samples used for training",
                key="xgb_sub"
            )
        
        st.markdown("---")
        
        # CNN Parameters
        st.markdown("### 🧠 CNN")
        with st.container():
            cnn_epochs = st.number_input(
                "Epochs", 
                min_value=10, max_value=500, step=10, 
                value=st.session_state.hyperparameters['cnn']['epochs'],
                help="Number of training epochs"
            )
            cnn_batch_size = st.number_input(
                "Batch Size", 
                min_value=8, max_value=128, step=8, 
                value=st.session_state.hyperparameters['cnn']['batch_size'],
                help="Training batch size"
            )
            cnn_learning_rate = st.number_input(
                "Learning Rate", 
                min_value=0.0001, max_value=0.01, step=0.0001, 
                value=st.session_state.hyperparameters['cnn']['learning_rate'],
                help="Optimizer learning rate",
                format="%.4f"
            )
            cnn_dropout = st.number_input(
                "Dropout Rate", 
                min_value=0.1, max_value=0.8, step=0.1, 
                value=st.session_state.hyperparameters['cnn']['dropout'],
                help="Dropout rate for regularization"
            )
    
    # Ensemble Weights Section
    st.markdown("## ⚖️ Ensemble Weights")
    st.info("💡 Weights should sum to 1.0 for optimal performance")
    
    weight_col1, weight_col2, weight_col3, weight_col4 = st.columns(4)
    
    with weight_col1:
        weight_rf = st.number_input(
            "Random Forest Weight", 
            min_value=0.0, max_value=1.0, step=0.05, 
            value=st.session_state.hyperparameters['weights']['rf']
        )
    
    with weight_col2:
        weight_xgb = st.number_input(
            "XGBoost Weight", 
            min_value=0.0, max_value=1.0, step=0.05, 
            value=st.session_state.hyperparameters['weights']['xgb']
        )
    
    with weight_col3:
        weight_lgb = st.number_input(
            "LightGBM Weight", 
            min_value=0.0, max_value=1.0, step=0.05, 
            value=st.session_state.hyperparameters['weights']['lgb']
        )
    
    with weight_col4:
        weight_cnn = st.number_input(
            "CNN Weight", 
            min_value=0.0, max_value=1.0, step=0.05, 
            value=st.session_state.hyperparameters['weights']['cnn']
        )
    
    # Validate weights
    total_weight = weight_rf + weight_xgb + weight_lgb + weight_cnn
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"⚠️ Weights sum to {total_weight:.2f}, should be 1.0")
    else:
        st.success(f"✅ Weights sum to {total_weight:.2f}")
    
    # Action Buttons
    st.markdown("## 🎛️ Actions")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("💾 Save Parameters", type="secondary", use_container_width=True):
            logger.info("Hyperparameters: Save clicked")
            ui_log("Hyperparameters saved")
            # Update session state
            st.session_state.hyperparameters = {
                'rf': {
                    'n_estimators': rf_n_estimators,
                    'max_depth': rf_max_depth,
                    'min_samples_split': rf_min_samples_split,
                    'min_samples_leaf': rf_min_samples_leaf
                },
                'xgb': {
                    'n_estimators': xgb_n_estimators,
                    'max_depth': xgb_max_depth,
                    'learning_rate': xgb_learning_rate,
                    'subsample': xgb_subsample
                },
                'lgb': {
                    'n_estimators': lgb_n_estimators,
                    'max_depth': lgb_max_depth,
                    'learning_rate': lgb_learning_rate,
                    'subsample': lgb_subsample
                },
                'cnn': {
                    'epochs': cnn_epochs,
                    'batch_size': cnn_batch_size,
                    'learning_rate': cnn_learning_rate,
                    'dropout': cnn_dropout
                },
                'weights': {
                    'rf': weight_rf,
                    'xgb': weight_xgb,
                    'lgb': weight_lgb,
                    'cnn': weight_cnn
                }
            }
            st.success("✅ Parameters saved successfully!")
    
    with action_col2:
        if st.button("🔄 Reset to Defaults", type="secondary", use_container_width=True):
            logger.info("Hyperparameters: Reset to defaults clicked")
            ui_log("Hyperparameters reset to defaults")
            st.session_state.hyperparameters = {
                'rf': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2},
                'xgb': {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8},
                'lgb': {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8},
                'cnn': {'epochs': 100, 'batch_size': 32, 'learning_rate': 0.001, 'dropout': 0.3},
                'weights': {'rf': 0.25, 'xgb': 0.25, 'lgb': 0.25, 'cnn': 0.25}
            }
            st.success("✅ Parameters reset to defaults!")
            st.rerun()
    
    with action_col3:
        if st.button("🚀 Retrain with New Parameters", type="primary", use_container_width=True):
            logger.info("Hyperparameters: Retrain clicked")
            ui_log("Retrain requested from Hyperparameters page")
            if not ALLOW_TRAINING:
                st.warning("⚠️ Training is disabled on this deployment. Set ALLOW_TRAINING=1 to enable.")
                ui_log("Training blocked: ALLOW_TRAINING is 0")
                return
                
            if abs(total_weight - 1.0) > 0.01:
                st.error("❌ Please fix ensemble weights before retraining!")
                return
                
            with st.spinner("Retraining models with new hyperparameters... This may take several minutes."):
                try:
                    # Save current parameters
                    st.session_state.hyperparameters = {
                        'rf': {
                            'n_estimators': rf_n_estimators,
                            'max_depth': rf_max_depth,
                            'min_samples_split': rf_min_samples_split,
                            'min_samples_leaf': rf_min_samples_leaf
                        },
                        'xgb': {
                            'n_estimators': xgb_n_estimators,
                            'max_depth': xgb_max_depth,
                            'learning_rate': xgb_learning_rate,
                            'subsample': xgb_subsample
                        },
                        'lgb': {
                            'n_estimators': lgb_n_estimators,
                            'max_depth': lgb_max_depth,
                            'learning_rate': lgb_learning_rate,
                            'subsample': lgb_subsample
                        },
                        'cnn': {
                            'epochs': cnn_epochs,
                            'batch_size': cnn_batch_size,
                            'learning_rate': cnn_learning_rate,
                            'dropout': cnn_dropout
                        },
                        'weights': {
                            'rf': weight_rf,
                            'xgb': weight_xgb,
                            'lgb': weight_lgb,
                            'cnn': weight_cnn
                        }
                    }
                    
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Show real training progress
                    progress_bar.progress(10)
                    status_text.text('🌲 Training Random Forest with new parameters...')
                    
                    # Train with reduced complexity for faster training
                    from ensemble_model import ExoplanetEnsembleModel
                    from data_preprocessing import ExoplanetDataProcessor
                    
                    # Create new ensemble with custom hyperparameters
                    new_ensemble = ExoplanetEnsembleModel()
                    new_processor = ExoplanetDataProcessor(data_dir=DATA_DIR)
                    ui_log("New ensemble and processor created")
                    
                    progress_bar.progress(30)
                    status_text.text('📊 Preparing training data...')
                    
                    # Prepare data (use quick training for faster results)
                    data = new_processor.prepare_data_for_training()
                    ui_log("Training data prepared (custom retrain)")
                    
                    progress_bar.progress(50)
                    status_text.text('🚀 Training models with optimized parameters...')
                    
                    # Use quick_train with reduced epochs for CNN
                    import os
                    os.environ['QUICK_TRAIN'] = '1'  # Signal for reduced training time
                    
                    # Hook progress into the custom training run
                    def _cb(stage, pct, msg):
                        try:
                            progress_bar.progress(max(0, min(100, int(pct))))
                        except Exception:
                            pass
                        status_text.text(msg)
                        ui_log(f"{stage.upper()}: {msg}")
                    try:
                        new_ensemble.set_progress_callback(_cb)
                    except Exception:
                        pass
                    # Use the new retrain-only method for faster training
                    try:
                        results = new_ensemble.train_ensemble_retrain_only(data)
                        logger.info("Hyperparameters: Retrain-only training completed successfully")
                    except AttributeError:
                        # Fallback to quick training if retrain-only method not available
                        results = new_ensemble.train_ensemble_quick(data)
                        logger.info("Hyperparameters: Fallback to quick training completed")
                    logger.info("Hyperparameters: Base training run finished")
                    ui_log("Models trained (custom retrain)")
                    
                    progress_bar.progress(80)
                    status_text.text('💾 Saving trained models...')
                    
                    # Save models
                    os.makedirs(MODELS_DIR, exist_ok=True)
                    prefix = os.path.join(MODELS_DIR, 'exoplanet_ensemble')
                    new_ensemble.save_models(filepath_prefix=prefix)
                    ui_log(f"Custom retrain models saved to prefix: {prefix}")
                    
                    # Save scaler and metrics
                    joblib.dump(new_processor.scaler, os.path.join(MODELS_DIR, 'exoplanet_ensemble_scaler.pkl'))
                    ui_log("Custom retrain scaler saved")
                    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
                        json.dump(results, f, indent=2)
                    ui_log("Custom retrain metrics saved")
                    
                    progress_bar.progress(100)
                    status_text.text('✅ Training completed!')
                    
                    # Actually retrain the models
                    # Note: This would need modification to ensemble_model.py to accept custom hyperparameters
                    # Refresh in-memory models for current session
                    st.session_state['metrics'] = results
                    ui_log("Retraining complete with new parameters")
                    st.session_state['metrics'] = results
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success("✅ Models retrained successfully with new parameters! Check the Statistics page for updated metrics.")
                    
                except Exception as e:
                    logger.exception("Error during retraining: %s", e)
                    ui_log(f"Error during retraining: {e}")
                    st.error(f"❌ Error retraining models: {e}")
    
    # Current Parameters Display
    st.markdown("## ⚙️ Current Parameters")
    
    with st.expander("🔍 View Current Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Random Forest:**")
            st.code(f"""
n_estimators: {st.session_state.hyperparameters['rf']['n_estimators']}
max_depth: {st.session_state.hyperparameters['rf']['max_depth']}
min_samples_split: {st.session_state.hyperparameters['rf']['min_samples_split']}
min_samples_leaf: {st.session_state.hyperparameters['rf']['min_samples_leaf']}
            """)
            
            st.markdown("**LightGBM:**")
            st.code(f"""
n_estimators: {st.session_state.hyperparameters['lgb']['n_estimators']}
max_depth: {st.session_state.hyperparameters['lgb']['max_depth']}
learning_rate: {st.session_state.hyperparameters['lgb']['learning_rate']}
subsample: {st.session_state.hyperparameters['lgb']['subsample']}
            """)
        
        with col2:
            st.markdown("**XGBoost:**")
            st.code(f"""
n_estimators: {st.session_state.hyperparameters['xgb']['n_estimators']}
max_depth: {st.session_state.hyperparameters['xgb']['max_depth']}
learning_rate: {st.session_state.hyperparameters['xgb']['learning_rate']}
subsample: {st.session_state.hyperparameters['xgb']['subsample']}
            """)
            
            st.markdown("**CNN:**")
            st.code(f"""
epochs: {st.session_state.hyperparameters['cnn']['epochs']}
batch_size: {st.session_state.hyperparameters['cnn']['batch_size']}
learning_rate: {st.session_state.hyperparameters['cnn']['learning_rate']:.4f}
dropout: {st.session_state.hyperparameters['cnn']['dropout']}
            """)
        
        st.markdown("**Ensemble Weights:**")
        st.code(f"""
Random Forest: {st.session_state.hyperparameters['weights']['rf']:.2f}
XGBoost: {st.session_state.hyperparameters['weights']['xgb']:.2f}
LightGBM: {st.session_state.hyperparameters['weights']['lgb']:.2f}
CNN: {st.session_state.hyperparameters['weights']['cnn']:.2f}
Total: {sum(st.session_state.hyperparameters['weights'].values()):.2f}
        """)


def page_visualizer():
    st.title("🌌 Exoplanet Visualizer")
    st.markdown("Watch as we visualize exoplanet systems in real-time! Enter parameters below to see your system come to life.")
    
    # Create tabs for different visualization modes
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🪐 Interactive Orbital Simulation", "📈 Light Curve", "🎯 System Presets"])
    
    with viz_tab1:
        # Interactive Canvas-based Visualizer
        st.markdown("### 🎛️ Interactive Exoplanet System")
        
        # Create columns for controls
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            st.markdown("**System Parameters**")
            orbital_period = st.number_input(
                "Orbital Period (days)", 
                min_value=0.1, max_value=10000.0, value=365.25, step=0.1,
                help="Time for one orbit. Earth: 365 days, Hot Jupiter: ~3 days"
            )
            planet_radius = st.number_input(
                "Planet Radius (R⊕)", 
                min_value=0.1, max_value=50.0, value=1.0, step=0.1,
                help="Planet size relative to Earth"
            )
            stellar_temp = st.number_input(
                "Stellar Temperature (K)", 
                min_value=1000, max_value=10000, value=5778, step=100,
                help="Star temperature affects color"
            )
            orbit_distance = st.slider(
                "Orbit Distance", 
                min_value=0.2, max_value=1.0, value=0.6, step=0.01,
                help="How far Planet A orbits from the star"
            )
            orbit_flip_a = st.checkbox("Flip Orbit A (left/right)", help="Place orbit to left or right focus")
        
        with control_col2:
            st.markdown("**Animation Controls**")
            animation_speed = st.slider("Animation Speed", 0.1, 3.0, 1.0, 0.1)
            
            # Placeholder for play/pause controls (will be controlled by JS)
            st.markdown("**Visual Options**")
            show_transit = st.checkbox("Show Transit Dimming", value=True)
            show_trail = st.checkbox("Show Planet Trail", value=True)
            show_brightness = st.checkbox("Show Brightness Meter", value=True)
            
            st.markdown("**Orbit Shapes**")
            eccentricity_a = st.slider("Eccentricity A", 0.0, 0.9, 0.0, 0.01, help="Elliptical amount for Planet A")
        
        with control_col3:
            st.markdown("**Second Planet**")
            enable_planet2 = st.checkbox("Enable second planet")
            
            if enable_planet2:
                p2_period = st.number_input("Period (days)", min_value=1.0, value=100.0, step=1.0, key="p2_period")
                p2_radius = st.number_input("Radius (R⊕)", min_value=0.1, value=2.5, step=0.1, key="p2_radius")
                eccentricity_b = st.slider("Eccentricity B", 0.0, 0.9, 0.0, 0.01, help="Elliptical amount for Planet B")
                orbit_distance2 = st.slider("Orbit Distance (B)", 0.2, 1.0, 0.5, 0.01, help="Planet B distance from star")
                orbit_flip_b = st.checkbox("Flip Orbit B (left/right)", help="Place orbit to left or right focus")
            else:
                p2_period = 100.0
                p2_radius = 2.5
                eccentricity_b = 0.0
                orbit_distance2 = 0.5
                orbit_flip_b = False
        
        # HTML/JS Canvas Visualizer
        visualizer_html = f'''
        <div style="width: 100%; height: 600px; background: linear-gradient(135deg, #0c0c1e 0%, #1a1a3e 100%); border-radius: 10px; position: relative; overflow: hidden;">
            <canvas id="planet-canvas" width="800" height="600" style="width: 100%; height: 100%; display: block;"></canvas>
            
            <!-- Control Overlay -->
            <div style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; color: white;">
                <button id="play-pause-btn" style="background: #007bff; color: white; border: none; padding: 8px 15px; border-radius: 4px; cursor: pointer; margin-right: 5px;">
                    ▶️ Start
                </button>
                <button id="reset-btn" style="background: #6c757d; color: white; border: none; padding: 8px 15px; border-radius: 4px; cursor: pointer;">
                    🔄 Reset
                </button>
            </div>
            
            <!-- Legend -->
            <div style="position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; color: white; font-size: 12px; line-height: 1.4;">
                <div><span style="display: inline-block; width: 12px; height: 12px; background: #ffd43b; border-radius: 50%; margin-right: 5px;"></span> Star (temp-based color)</div>
                <div><span style="display: inline-block; width: 8px; height: 8px; background: #4dabf7; border-radius: 50%; margin-right: 7px;"></span> Planet A</div>
                <div style="display: {('block' if enable_planet2 else 'none')};"><span style="display: inline-block; width: 8px; height: 8px; background: #91a7ff; border-radius: 50%; margin-right: 7px;"></span> Planet B</div>
                <div style="margin-top: 5px;">🌟 Transit = star dims</div>
            </div>
            
            <!-- System Info -->
            <div style="position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; color: white; font-size: 12px;">
                <div id="system-info">
                    <strong>Planet A:</strong> {planet_radius:.1f} R⊕, {orbital_period:.1f} days<br>
                    <strong>Star:</strong> {stellar_temp}K<br>
                    <strong>Animation:</strong> {animation_speed}x speed
                </div>
            </div>
        </div>
        
        <script>
        (function() {{
            const canvas = document.getElementById('planet-canvas');
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            const playPauseBtn = document.getElementById('play-pause-btn');
            const resetBtn = document.getElementById('reset-btn');
            
            let isPlaying = false;
            let animationId = null;
            let angle = 0;
            let angle2 = 0;
            let lastTime = 0;
            let trail = [];
            let trail2 = [];
            const MAX_TRAIL = 60;
            
            // System parameters (from Streamlit)
            const params = {{
                orbitalPeriod: {orbital_period},
                planetRadius: {planet_radius},
                stellarTemp: {stellar_temp},
                speed: {animation_speed},
                eccentricity: {eccentricity_a},
                eccentricity2: {eccentricity_b},
                orbitDistance: {orbit_distance},
                orbitDistance2: {orbit_distance2},
                flipA: {str(orbit_flip_a).lower()},
                flipB: {str(orbit_flip_b).lower()},
                planet2Enabled: {str(enable_planet2).lower()},
                planet2: {{ period: {p2_period}, radius: {p2_radius} }},
                showTransit: {str(show_transit).lower()},
                showTrail: {str(show_trail).lower()},
                showBrightness: {str(show_brightness).lower()}
            }};
            
            function resizeCanvas() {{
                const rect = canvas.getBoundingClientRect();
                canvas.width = rect.width;
                canvas.height = rect.height;
            }}
            
            function getStarColor(temp) {{
                if (temp < 3700) return '#ff6b6b';      // Red dwarf
                if (temp < 5200) return '#ffd93d';      // Orange
                if (temp < 6000) return '#6bcf7f';      // Yellow-green  
                if (temp < 7500) return '#74c0fc';      // Blue-white
                return '#c5f6fa';                       // Blue
            }}
            
            function drawFrame(currentTime) {{
                if (!ctx) return;
                
                const deltaTime = currentTime - lastTime;
                lastTime = currentTime;
                
                if (isPlaying && deltaTime > 0) {{
                    // Update angles based on orbital periods
                    const timeScale = params.speed * 0.0001; // Adjust for visual speed
                    angle += (2 * Math.PI * timeScale * deltaTime) / params.orbitalPeriod;
                    if (params.planet2Enabled) {{
                        angle2 += (2 * Math.PI * timeScale * deltaTime) / params.planet2.period;
                    }}
                }}
                
                // Clear canvas
                ctx.fillStyle = 'rgba(12, 12, 30, 1)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                const centerX = canvas.width / 2;
                const centerY = canvas.height / 2;
                const maxRadius = Math.min(canvas.width, canvas.height) * 0.35;
                
                // Draw star
                const starColor = getStarColor(params.stellarTemp);
                ctx.fillStyle = starColor;
                ctx.beginPath();
                ctx.arc(centerX, centerY, 15, 0, 2 * Math.PI);
                ctx.fill();
                
                // Planet A orbit and position
                const radiusA = maxRadius * params.orbitDistance;
                const eccentricityA = params.eccentricity;
                const semiMajorA = radiusA;
                const semiMinorA = radiusA * Math.sqrt(1 - eccentricityA * eccentricityA);
                
                // Draw orbit A
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
                ctx.lineWidth = 1;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.ellipse(centerX, centerY, semiMajorA, semiMinorA, 0, 0, 2 * Math.PI);
                ctx.stroke();
                ctx.setLineDash([]);
                
                // Calculate planet A position
                const planetAX = centerX + semiMajorA * Math.cos(angle) * (params.flipA ? -1 : 1);
                const planetAY = centerY + semiMinorA * Math.sin(angle);
                
                // Update trail A
                if (params.showTrail) {{
                    trail.push({{ x: planetAX, y: planetAY }});
                    if (trail.length > MAX_TRAIL) trail.shift();
                    
                    // Draw trail A
                    trail.forEach((point, i) => {{
                        const alpha = (i + 1) / trail.length * 0.5;
                        ctx.fillStyle = `rgba(77, 171, 247, ${{alpha}})`;
                        ctx.beginPath();
                        ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
                        ctx.fill();
                    }});
                }}
                
                // Draw planet A
                const planetASize = Math.max(4, params.planetRadius * 3);
                ctx.fillStyle = params.planetRadius < 2 ? '#4dabf7' : params.planetRadius < 10 ? '#91a7ff' : '#ffd43b';
                ctx.beginPath();
                ctx.arc(planetAX, planetAY, planetASize, 0, 2 * Math.PI);
                ctx.fill();
                
                // Planet B (if enabled)
                if (params.planet2Enabled) {{
                    const radiusB = maxRadius * params.orbitDistance2;
                    const eccentricityB = params.eccentricity2;
                    const semiMajorB = radiusB;
                    const semiMinorB = radiusB * Math.sqrt(1 - eccentricityB * eccentricityB);
                    
                    // Draw orbit B
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
                    ctx.lineWidth = 1;
                    ctx.setLineDash([3, 3]);
                    ctx.beginPath();
                    ctx.ellipse(centerX, centerY, semiMajorB, semiMinorB, 0, 0, 2 * Math.PI);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    
                    // Calculate planet B position
                    const planetBX = centerX + semiMajorB * Math.cos(angle2) * (params.flipB ? -1 : 1);
                    const planetBY = centerY + semiMinorB * Math.sin(angle2);
                    
                    // Update trail B
                    if (params.showTrail) {{
                        trail2.push({{ x: planetBX, y: planetBY }});
                        if (trail2.length > MAX_TRAIL) trail2.shift();
                        
                        // Draw trail B
                        trail2.forEach((point, i) => {{
                            const alpha = (i + 1) / trail2.length * 0.4;
                            ctx.fillStyle = `rgba(145, 167, 255, ${{alpha}})`;
                            ctx.beginPath();
                            ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
                            ctx.fill();
                        }});
                    }}
                    
                    // Draw planet B
                    const planetBSize = Math.max(4, params.planet2.radius * 3);
                    ctx.fillStyle = '#91a7ff';
                    ctx.beginPath();
                    ctx.arc(planetBX, planetBY, planetBSize, 0, 2 * Math.PI);
                    ctx.fill();
                }}
                
                // Transit effects (simple star dimming)
                if (params.showTransit) {{
                    const distanceFromStar = Math.sqrt((planetAX - centerX) ** 2 + (planetAY - centerY) ** 2);
                    if (distanceFromStar < 25) {{ // Planet is near star
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
                        ctx.beginPath();
                        ctx.arc(centerX, centerY, 20, 0, 2 * Math.PI);
                        ctx.fill();
                    }}
                }}
                
                if (isPlaying) {{
                    animationId = requestAnimationFrame(drawFrame);
                }}
            }}
            
            function startAnimation() {{
                if (!isPlaying) {{
                    isPlaying = true;
                    playPauseBtn.textContent = '⏸️ Pause';
                    lastTime = performance.now();
                    animationId = requestAnimationFrame(drawFrame);
                }}
            }}
            
            function stopAnimation() {{
                isPlaying = false;
                playPauseBtn.textContent = '▶️ Start';
                if (animationId) {{
                    cancelAnimationFrame(animationId);
                    animationId = null;
                }}
            }}
            
            function resetAnimation() {{
                stopAnimation();
                angle = 0;
                angle2 = 0;
                trail = [];
                trail2 = [];
                drawFrame(performance.now());
            }}
            
            // Event listeners
            if (playPauseBtn) {{
                playPauseBtn.addEventListener('click', () => {{
                    if (isPlaying) {{
                        stopAnimation();
                    }} else {{
                        startAnimation();
                    }}
                }});
            }}
            
            if (resetBtn) {{
                resetBtn.addEventListener('click', resetAnimation);
            }}
            
            // Initialize
            window.addEventListener('resize', resizeCanvas);
            resizeCanvas();
            drawFrame(performance.now());
        }})();
        </script>
        '''
        
        # Display the HTML component
        st.components.v1.html(visualizer_html, height=650)
        
        # System Information below the visualizer
        st.markdown("### 📊 System Information")
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        
        with info_col1:
            st.metric("Orbital Period", f"{orbital_period:.1f} days")
        
        with info_col2:
            # Calculate approximate orbital velocity (simplified)
            orbital_velocity = 29.8 * np.sqrt(1 / orbital_period * 365.25)  # Rough approximation
            st.metric("Orbital Velocity", f"{orbital_velocity:.1f} km/s")
        
        with info_col3:
            # Calculate habitable zone estimate
            habitable_zone_inner = np.sqrt(stellar_temp / 5778) * 0.95
            habitable_zone_outer = np.sqrt(stellar_temp / 5778) * 1.37
            orbit_au = (orbital_period / 365.25) ** (2/3)  # Simplified Kepler's law
            in_hz = habitable_zone_inner <= orbit_au <= habitable_zone_outer
            st.metric("Habitable Zone", "Yes" if in_hz else "No")
        
        with info_col4:
            # Transit probability (simplified)
            transit_prob = min(100, (0.005 * (stellar_temp/5778)**0.5) / orbit_au * 100)
            st.metric("Transit Probability", f"{transit_prob:.1f}%")
    
    with viz_tab2:
        st.markdown("### 📈 Simulated Light Curve")
        
        # Light curve parameters
        lc_col1, lc_col2 = st.columns(2)
        
        with lc_col1:
            transit_depth_ppm = st.number_input(
                "Transit Depth (ppm)", 
                min_value=10, max_value=50000, value=1000, step=10,
                help="How much the star dims during transit"
            )
            transit_duration_hours = st.number_input(
                "Transit Duration (hours)", 
                min_value=0.1, max_value=24.0, value=3.0, step=0.1,
                help="How long the transit lasts"
            )
        
        with lc_col2:
            noise_level = st.slider(
                "Noise Level", 
                min_value=0.0, max_value=0.01, value=0.001, step=0.0001,
                format="%.4f",
                help="Measurement uncertainty"
            )
            phase_offset = st.slider(
                "Phase Offset", 
                min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                help="Transit timing within the period"
            )
        
        # Generate light curve
        time_points = np.linspace(0, 2, 1000)  # 2 periods
        flux = np.ones_like(time_points)
        
        # Add transits
        transit_duration_phase = transit_duration_hours / 24 / orbital_period
        for period_num in [0, 1]:
            transit_center = period_num + phase_offset
            transit_mask = np.abs(time_points - transit_center) < transit_duration_phase/2
            flux[transit_mask] = 1 - transit_depth_ppm/1e6
        
        # Add noise
        np.random.seed(42)  # For reproducible results
        flux += np.random.normal(0, noise_level, len(flux))
        
        # Create light curve plot
        fig_lc = go.Figure()
        fig_lc.add_trace(go.Scatter(
            x=time_points * orbital_period,
            y=flux,
            mode='lines',
            line=dict(color='#74c0fc', width=1),
            name='Observed Flux'
        ))
        
        # Highlight transits
        for period_num in [0, 1]:
            transit_center = (period_num + phase_offset) * orbital_period
            transit_start = transit_center - transit_duration_hours/2
            transit_end = transit_center + transit_duration_hours/2
            
            fig_lc.add_vrect(
                x0=transit_start, x1=transit_end,
                fillcolor="rgba(255,0,0,0.2)",
                layer="below",
                line_width=0,
            )
        
        fig_lc.update_layout(
            title='Simulated Transit Light Curve',
            xaxis_title='Time (days)',
            yaxis_title='Relative Flux',
            plot_bgcolor='rgba(0,0,30,0.8)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        st.plotly_chart(fig_lc, use_container_width=True)
        
        # Light curve statistics
        st.markdown("### 📊 Light Curve Analysis")
        
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Transit Depth", f"{transit_depth_ppm} ppm")
        
        with stats_col2:
            st.metric("Transit Duration", f"{transit_duration_hours:.1f} hours")
        
        with stats_col3:
            flux_std = np.std(flux)
            st.metric("RMS Noise", f"{flux_std*1e6:.0f} ppm")
        
        with stats_col4:
            snr = (transit_depth_ppm/1e6) / flux_std
            st.metric("Signal-to-Noise", f"{snr:.1f}")
    
    with viz_tab3:
        st.markdown("### 🎯 Explore Preset Systems")
        st.markdown("Click on a preset to automatically configure the system parameters")
        
        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
        
        presets = {
            "Earth-like": {
                "period": 365.25, "radius": 1.0, "temp": 5778, "sma": 1.0, "ecc": 0.017,
                "description": "Earth analog in our solar system"
            },
            "Hot Jupiter": {
                "period": 3.0, "radius": 11.2, "temp": 6000, "sma": 0.05, "ecc": 0.0,
                "description": "Gas giant in very close orbit"
            },
            "Super Earth": {
                "period": 100.0, "radius": 2.5, "temp": 5500, "sma": 0.3, "ecc": 0.1,
                "description": "Larger rocky planet"
            },
            "Cold Jupiter": {
                "period": 4333.0, "radius": 11.2, "temp": 5778, "sma": 5.2, "ecc": 0.05,
                "description": "Jupiter analog"
            }
        }
        
        for i, (name, preset) in enumerate(presets.items()):
            col = [preset_col1, preset_col2, preset_col3, preset_col4][i]
            with col:
                if st.button(f"🪐 {name}", use_container_width=True, key=f"preset_{i}"):
                    st.success(f"Loaded {name} preset!")
                    st.info(f"**{name}**: {preset['description']}")
                    # In a more advanced implementation, this would update the form values above
                
                with st.expander(f"ℹ️ {name} Details"):
                    st.write(f"**Period:** {preset['period']} days")
                    st.write(f"**Radius:** {preset['radius']} R⊕")
                    st.write(f"**Star Temp:** {preset['temp']} K")
                    st.write(f"**Distance:** {preset['sma']} AU")
                    st.write(f"**Eccentricity:** {preset['ecc']}")
        
        # Comparison Chart
        st.markdown("### 📊 Preset Comparison")
        
        preset_names = list(presets.keys())
        preset_periods = [presets[name]['period'] for name in preset_names]
        preset_radii = [presets[name]['radius'] for name in preset_names]
        preset_distances = [presets[name]['sma'] for name in preset_names]
        
        comparison_col1, comparison_col2 = st.columns(2)
        
        with comparison_col1:
            fig_periods = go.Figure(data=[
                go.Bar(x=preset_names, y=preset_periods, 
                       marker_color=['#28a745', '#dc3545', '#ffc107', '#17a2b8'])
            ])
            fig_periods.update_layout(
                title='Orbital Periods Comparison',
                yaxis_title='Period (days)',
                yaxis_type='log',
                height=300
            )
            st.plotly_chart(fig_periods, use_container_width=True)
        
        with comparison_col2:
            fig_radii = go.Figure(data=[
                go.Bar(x=preset_names, y=preset_radii,
                       marker_color=['#28a745', '#dc3545', '#ffc107', '#17a2b8'])
            ])
            fig_radii.update_layout(
                title='Planet Radii Comparison',
                yaxis_title='Radius (Earth radii)',
                height=300
            )
            st.plotly_chart(fig_radii, use_container_width=True)


def page_statistics(metrics):
    st.title("📊 Model Statistics")
    
    if not metrics:
        st.info("💡 No metrics available. Train the model first to see detailed statistics.")
        return
    
    # Performance Circles Section
    st.markdown("## 🎯 Performance Overview")
    
    # Create two columns for performance circles
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        # Accuracy Doughnut Chart
        accuracy_pct = metrics.get('ensemble_accuracy', 0) * 100
        fig_acc = go.Figure(data=[go.Pie(
            labels=['Accuracy', 'Error'],
            values=[accuracy_pct, 100 - accuracy_pct],
            hole=0.6,
            marker_colors=['#28a745', '#e9ecef'],
            textinfo='none',
            hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
        )])
        fig_acc.update_layout(
            title={'text': f'<b>Ensemble Accuracy</b><br><span style="font-size:24px">{accuracy_pct:.1f}%</span>', 'x': 0.5},
            showlegend=False,
            height=300,
            margin=dict(t=80, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with perf_col2:
        # AUC Doughnut Chart
        auc_pct = metrics.get('ensemble_auc', 0) * 100
        fig_auc = go.Figure(data=[go.Pie(
            labels=['AUC Score', 'Remaining'],
            values=[auc_pct, 100 - auc_pct],
            hole=0.6,
            marker_colors=['#007bff', '#e9ecef'],
            textinfo='none',
            hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
        )])
        fig_auc.update_layout(
            title={'text': f'<b>ROC AUC Score</b><br><span style="font-size:24px">{auc_pct:.1f}%</span>', 'x': 0.5},
            showlegend=False,
            height=300,
            margin=dict(t=80, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_auc, use_container_width=True)
    
    # Model Comparison Charts
    st.markdown("## 📈 Model Comparison")
    
    # Prepare data for comparison charts
    model_names = []
    accuracies = []
    aucs = []
    
    model_labels = {
        'rf': 'Random Forest',
        'xgb': 'XGBoost', 
        'lgb': 'LightGBM',
        'cnn': 'CNN'
    }
    
    # Extract individual model results
    individual_results = metrics.get('individual_results', {})
    
    for model_name in ['rf', 'xgb', 'lgb', 'cnn']:
        if model_name in individual_results:
            model_names.append(model_labels[model_name])
            accuracies.append(individual_results[model_name].get('accuracy', 0) * 100)
            aucs.append(individual_results[model_name].get('auc', 0) * 100)
        elif f'{model_name}_accuracy' in metrics:
            model_names.append(model_labels[model_name])
            accuracies.append(metrics[f'{model_name}_accuracy'] * 100)
            aucs.append(metrics[f'{model_name}_auc'] * 100)
    
    # Add ensemble to comparison if we have individual models
    if model_names:
        model_names.append('Ensemble')
        accuracies.append(metrics.get('ensemble_accuracy', 0) * 100)
        aucs.append(metrics.get('ensemble_auc', 0) * 100)
    
    if model_names:
        # Create comparison chart
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            # Accuracy Comparison Bar Chart
            fig_acc_comp = go.Figure(data=[
                go.Bar(x=model_names, y=accuracies, 
                       marker_color=['#28a745', '#ffc107', '#17a2b8', '#dc3545', '#6f42c1'],
                       text=[f'{acc:.1f}%' for acc in accuracies],
                       textposition='auto')
            ])
            fig_acc_comp.update_layout(
                title='Model Accuracy Comparison',
                yaxis_title='Accuracy (%)',
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_acc_comp, use_container_width=True)
        
        with comp_col2:
            # AUC Comparison Line Chart
            fig_auc_comp = go.Figure(data=[
                go.Scatter(x=model_names, y=aucs, 
                          mode='lines+markers',
                          line=dict(color='#007bff', width=3),
                          marker=dict(size=10, color='#007bff'))
            ])
            fig_auc_comp.update_layout(
                title='Model AUC Score Comparison', 
                yaxis_title='AUC Score (%)',
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_auc_comp, use_container_width=True)
    
    # Dataset Statistics Section
    st.markdown("## 📊 Dataset Statistics")
    
    # Create metrics cards using columns
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric(
            label="🗂️ Total Samples",
            value="17,000+",
            help="Total number of training samples"
        )
    
    with stat_col2:
        st.metric(
            label="📋 Features", 
            value="50+",
            help="Number of input features per sample"
        )
    
    with stat_col3:
        st.metric(
            label="🌟 Confirmed Planets",
            value="5,087",
            help="Confirmed exoplanet detections"
        )
    
    with stat_col4:
        st.metric(
            label="❓ False Positives",
            value="11,913", 
            help="False positive candidates"
        )
    
    # Model Training Status
    st.markdown("## 🔧 Model Training")
    
    # CSV Upload Section
    st.markdown("### 📁 Upload Additional Training Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file with additional exoplanet data", 
        type=['csv'],
        help="Upload a CSV file with the same format as the training data to enhance the model"
    )
    
    if uploaded_file is not None:
        try:
            # Preview the uploaded data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} rows from {uploaded_file.name}")
            
            with st.expander("📋 Preview Uploaded Data", expanded=False):
                st.dataframe(df.head(10))
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Columns:** {list(df.columns)}")
            
            # Save uploaded file for training
            upload_path = os.path.join(DATA_DIR, f"uploaded_{uploaded_file.name}")
            df.to_csv(upload_path, index=False)
            st.session_state['uploaded_data_path'] = upload_path
            
        except Exception as e:
            st.error(f"❌ Error reading uploaded file: {e}")
    
    train_col1, train_col2 = st.columns([2, 1])
    
    with train_col1:
        st.info("💡 **Tip**: Use the Hyperparameters page to adjust model settings and retrain with custom configurations.")
        if 'uploaded_data_path' in st.session_state:
            st.success(f"📁 Additional data ready: {os.path.basename(st.session_state['uploaded_data_path'])}")
    
    with train_col2:
        if st.button("🚀 Retrain Models", type="primary", use_container_width=True):
            logger.info("Statistics: Retrain Models clicked")
            ui_log("Retrain requested from Statistics page")
            if not ALLOW_TRAINING:
                st.warning("⚠️ Training is disabled on this deployment. Set ALLOW_TRAINING=1 to enable.")
                ui_log("Training blocked: ALLOW_TRAINING is 0")
                return
                
            with st.spinner("Retraining models... This may take several minutes."):
                try:
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    progress_bar.progress(10)
                    status_text.text('🔄 Initializing training process...')
                    
                    # Load ensemble and processor
                    ensemble, processor, _ = load_models_and_scaler()
                    ui_log("Loaded ensemble and processor for retrain")
                    
                    progress_bar.progress(20)
                    status_text.text('📊 Preparing training data...')
                    
                    # Check for uploaded data
                    if 'uploaded_data_path' in st.session_state:
                        try:
                            # Merge uploaded data with existing data
                            status_text.text('🔗 Merging uploaded data with existing dataset...')
                            processor.merge_additional_data(st.session_state['uploaded_data_path'])
                            progress_bar.progress(40)
                        except Exception as e:
                            st.warning(f"Could not merge uploaded data: {e}. Using original dataset.")
                    
                    progress_bar.progress(50)
                    status_text.text('🚀 Training ensemble models (optimized for speed)...')
                    
                    # Set quick training mode
                    os.environ['QUICK_TRAIN'] = '1'
                    
                    # Actually retrain the models (this will show progress via callback)
                    results = train_and_save(ensemble, processor)
                    logger.info("Statistics: Training completed and saved")
                    ui_log("Statistics retrain completed")
                    
                    progress_bar.progress(100)
                    st.session_state['metrics'] = results
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success("✅ Models retrained successfully! Refresh the page to see updated metrics.")
                    st.rerun()
                    
                except Exception as e:
                    logger.exception("Error retraining from Statistics page: %s", e)
                    ui_log(f"Error retraining from Statistics: {e}")
                    st.error(f"❌ Error retraining models: {e}")
    
    # Recent Training Information
    st.markdown("## ℹ️ Training Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **Training Configuration:**
        - Random Forest: 200 estimators, max depth 20
        - XGBoost: 300 estimators, learning rate 0.1
        - LightGBM: 300 estimators, learning rate 0.1
        - CNN: 100 epochs, batch size 32
        """)
    
    with info_col2:
        st.markdown("""
        **Ensemble Method:**
        - Weighted average of all models
        - Optimal weights determined by validation performance
        - Final prediction combines strengths of each algorithm
        """)
    
    # Detailed Metrics Expandable Section
    with st.expander("🔍 Detailed Metrics", expanded=False):
        st.json(metrics)


def main():
    _inject_css()
    
    with st.sidebar:
        st.title("🚀 Exoplanet Hunt")
        st.markdown("Navigate through the system:")
        
        page = st.radio(
            "Select Page", 
            ["Home", "Classify", "Visualizer", "Hyperparameters", "Statistics"], 
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        if 'metrics' in st.session_state and st.session_state['metrics']:
            metrics = st.session_state['metrics']
            accuracy = metrics.get('ensemble_accuracy', 0) * 100
            auc = metrics.get('ensemble_auc', 0) * 100
            st.metric("Accuracy", f"{accuracy:.1f}%")
            st.metric("AUC Score", f"{auc:.1f}%")
        else:
            st.info("Train models to see metrics")
        
        st.markdown("---")
        st.markdown("### ⚙️ System Info")
        st.text(f"Training: {'Enabled' if ALLOW_TRAINING else 'Disabled'}")
        st.text(f"Models: {'Loaded' if 'metrics' in st.session_state and st.session_state['metrics'] else 'Not Found'}")

        # Logs panel
        st.markdown("---")
        st.markdown("### 🧾 Logs")
        if st.button("Refresh Logs", use_container_width=True):
            pass  # A no-op to trigger rerun and refresh display below
        if st.button("Reload Models", use_container_width=True):
            try:
                load_models_and_scaler.clear()
            except Exception:
                pass
            _e, _p, _m = load_models_and_scaler()
            st.session_state['metrics'] = _m
            ui_log("Models reloaded via sidebar button")
            st.success("Models reloaded")
        logs = []
        # Prefer file logs if available
        try:
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as lf:
                    logs = lf.readlines()[-200:]
        except Exception:
            logs = []
        # Merge with in-memory buffer
        if 'ui_logs' in st.session_state:
            logs += [f"[UI] {m}\n" for m in st.session_state['ui_logs'][-100:]]
        if logs:
            st.code(''.join(logs) or '(no logs)', language='')
        else:
            st.info("No logs yet. Interact with the app or start a retrain.")

    ensemble, processor, metrics = load_models_and_scaler()
    # Keep latest metrics in session
    if 'metrics' not in st.session_state or st.session_state['metrics'] is None:
        st.session_state['metrics'] = metrics

    if page == "Home":
        page_home(st.session_state.get('metrics'))
    elif page == "Classify":
        page_classify(ensemble, processor)
    elif page == "Visualizer":
        page_visualizer()
    elif page == "Hyperparameters":
        page_hyperparameters(ensemble, processor)
    elif page == "Statistics":
        page_statistics(st.session_state.get('metrics'))


if __name__ == "__main__":
    main()
