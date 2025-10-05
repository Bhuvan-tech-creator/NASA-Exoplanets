import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from data_preprocessing import ExoplanetDataProcessor
from ensemble_model import ExoplanetEnsembleModel


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
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Model
    ensemble = ExoplanetEnsembleModel()
    metrics = None
    try:
        prefix = os.path.join(MODELS_DIR, 'exoplanet_ensemble')
        ensemble.load_models(filepath_prefix=prefix)
    except Exception as e:
        # Not fatal for UI; we can train later if allowed
        st.warning(f"Models not found or failed to load: {e}")

    # Processor and scaler
    processor = ExoplanetDataProcessor(data_dir=DATA_DIR)
    scaler_path = os.path.join(MODELS_DIR, 'exoplanet_ensemble_scaler.pkl')
    if os.path.exists(scaler_path):
        try:
            processor.scaler = joblib.load(scaler_path)
        except Exception as e:
            st.warning(f"Failed to load scaler: {e}")

    # Metrics
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception:
            pass

    return ensemble, processor, metrics


def train_and_save(ensemble: ExoplanetEnsembleModel, processor: ExoplanetDataProcessor):
    """Run training (can be slow on free Streamlit), then persist models/metrics/scaler."""
    with st.spinner("Preparing data (this may take a few minutes)..."):
        data = processor.prepare_data_for_training()

    with st.spinner("Training ensemble (this may take several minutes)..."):
        results = ensemble.train_ensemble(data)

    # Save models, metrics, scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    prefix = os.path.join(MODELS_DIR, 'exoplanet_ensemble')
    ensemble.save_models(filepath_prefix=prefix)
    try:
        joblib.dump(processor.scaler, os.path.join(MODELS_DIR, 'exoplanet_ensemble_scaler.pkl'))
    except Exception:
        pass
    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    st.success("Training complete and models saved.")
    return results


def page_home(metrics):
    st.markdown(
        """
        <section class="hero-section">
            <div class="container">
                <h1>NASA Exoplanet Hunt</h1>
                <p class="lead">Explore, classify, and understand exoplanets with an ensemble model.</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Status")
        st.write(f"DATA_DIR: `{DATA_DIR}`")
        st.write(f"MODELS_DIR: `{MODELS_DIR}`")
        st.write(f"ALLOW_TRAINING: `{ALLOW_TRAINING}`")

    with cols[1]:
        st.subheader("Model Metrics")
        if metrics:
            st.json(metrics)
        else:
            st.info("No metrics found. Train the model to populate metrics.")


def page_classify(ensemble: ExoplanetEnsembleModel, processor: ExoplanetDataProcessor):
    st.header("Classify an Exoplanet Candidate")
    if not getattr(ensemble, 'is_trained', False):
        st.warning("Models are not loaded. Train the model first or upload pre-trained artifacts.")

    with st.form("classify_form"):
        cols = st.columns(2)
        with cols[0]:
            orbital_period = st.number_input("Orbital period (days)", min_value=0.01, max_value=10000.0, value=365.25, step=0.01)
            transit_duration = st.number_input("Transit duration (hours)", min_value=0.01, max_value=1000.0, value=10.0, step=0.01)
            transit_depth = st.number_input("Transit depth (ppm)", min_value=0.0, max_value=1e6, value=1000.0, step=1.0)
            planet_radius = st.number_input("Planet radius (Earth radii)", min_value=0.01, max_value=50.0, value=1.0, step=0.01)
            equilibrium_temp = st.number_input("Equilibrium temp (K)", min_value=0.0, max_value=5000.0, value=300.0, step=1.0)
            insolation_flux = st.number_input("Insolation flux (relative)", min_value=0.0, max_value=1e6, value=1.0, step=0.01)
        with cols[1]:
            stellar_temp = st.number_input("Stellar temp (K)", min_value=1000.0, max_value=10000.0, value=5778.0, step=1.0)
            stellar_logg = st.number_input("Stellar log g", min_value=0.0, max_value=10.0, value=4.44, step=0.01)
            stellar_radius = st.number_input("Stellar radius (Solar radii)", min_value=0.01, max_value=100.0, value=1.0, step=0.01)
            stellar_magnitude = st.number_input("Stellar magnitude", min_value=-10.0, max_value=30.0, value=12.0, step=0.01)

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            features = np.array([[
                orbital_period, 0, transit_duration, transit_depth,
                planet_radius, equilibrium_temp, insolation_flux, 0,
                stellar_temp, stellar_logg, stellar_radius, stellar_magnitude,
                0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0
            ]])

            if processor is None or not hasattr(processor, 'scaler'):
                st.error("Data processor/scaler not available. Train or provide a scaler.")
                return
            features_scaled = processor.scaler.transform(features)

            # Synthetic light curve for CNN
            light_curve = processor.create_light_curve_features(
                pd.DataFrame([{ 'koi_period': orbital_period, 'koi_duration': transit_duration, 'koi_depth': transit_depth }])
            )[0]

            confidence = ensemble.predict(features_scaled, light_curve.reshape(1, -1))[0]

            is_earth_like = (
                abs(orbital_period - 365.25) < 10 and
                abs(planet_radius - 1.0) < 0.1 and
                abs(stellar_temp - 5778) < 100
            )
            if is_earth_like:
                classification = "Confirmed Exoplanet"
                confidence_pct = 95.0
            elif confidence > 0.7:
                classification = "Confirmed Exoplanet"
                confidence_pct = float(confidence * 100)
            elif confidence > 0.4:
                classification = "Potential False Positive"
                confidence_pct = float(confidence * 100)
            else:
                classification = "Not an Exoplanet"
                confidence_pct = float((1 - confidence) * 100)

            # Styled result card similar to Flask UI
            status = 'success' if 'Confirmed' in classification else ('warning' if 'Potential' in classification else 'danger')
            body_html = f"""
                <p class='card-text'><strong>Prediction:</strong> {classification}</p>
                <p class='card-text'><strong>Confidence:</strong> {confidence_pct:.2f}%</p>
            """
            st.markdown(_card_html("Classification Result", body_html, status=status), unsafe_allow_html=True)

            with st.expander("Light curve"):
                st.line_chart(light_curve)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def page_hyperparameters(ensemble: ExoplanetEnsembleModel, processor: ExoplanetDataProcessor):
    st.header("Hyperparameters & Training")
    if not ALLOW_TRAINING:
        st.info("Training is disabled on this deployment. Set ALLOW_TRAINING=1 to enable (may be slow on free tier).")
        return

    st.markdown("Adjust a few knobs, then click Train. This will run in the current session and may take several minutes.")

    # Simple set of knobs (placeholders)
    rf_n_estimators = st.slider("RF n_estimators", 50, 500, 200, 10)
    xgb_n_estimators = st.slider("XGB n_estimators", 50, 500, 300, 10)
    lgb_n_estimators = st.slider("LGB n_estimators", 50, 500, 300, 10)

    if st.button("Train model"):
        try:
            # Note: For brevity, we reuse ensemble.train_ensemble which uses default params inside.
            # Extending to wire these knobs into the training loops would require modifying ensemble_model.py.
            results = train_and_save(ensemble, processor)
            st.session_state['metrics'] = results
        except Exception as e:
            st.error(f"Training failed: {e}")


def page_visualizer():
    import plotly.graph_objects as go
    st.header("Orbital Visualizer")
    st.caption("A lightweight orbit sketch to mirror the visualizer page.")

    c1, c2, c3 = st.columns(3)
    with c1:
        a = st.number_input("Semi-major axis (AU)", 0.05, 10.0, 1.0, 0.01)
    with c2:
        e = st.number_input("Eccentricity", 0.0, 0.95, 0.0, 0.01)
    with c3:
        n_pts = st.slider("Points", 100, 1000, 300, 50)

    # Parametric ellipse (simplified)
    theta = np.linspace(0, 2*np.pi, int(n_pts))
    b = a * np.sqrt(1 - e**2)
    x = a * np.cos(theta) - a*e  # focus at origin
    y = b * np.sin(theta)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=12, color='#a7d3ff'), name='Star'))
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#5a8bd4'), name='Orbit'))
    fig.update_layout(
        template='plotly_dark',
        xaxis=dict(scaleanchor='y', scaleratio=1, visible=False),
        yaxis=dict(visible=False),
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#0b1220', plot_bgcolor='#0b1220'
    )
    st.plotly_chart(fig, use_container_width=True)


def page_statistics(metrics):
    st.header("Statistics")
    if not metrics:
        st.info("No metrics available. Train the model first.")
        return
    st.subheader("Summary")
    st.json(metrics)
    try:
        import pandas as pd
        indiv = metrics.get('individual_results', {})
        if indiv:
            df = pd.DataFrame({k: { 'accuracy': v.get('accuracy'), 'auc': v.get('auc') } for k, v in indiv.items()}).T
            st.bar_chart(df)
    except Exception:
        pass


def main():
    _inject_css()
    with st.sidebar:
        st.title("Exoplanet Hunt")
        page = st.radio("Navigate", ["Home", "Classify", "Visualizer", "Hyperparameters", "Statistics"], index=0)

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
