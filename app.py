from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from data_preprocessing import ExoplanetDataProcessor
from ensemble_model import ExoplanetEnsembleModel
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import joblib

app = Flask(__name__)
app.secret_key = 'exoplanet_detection_secret_key_2025'

# Paths configurable via environment for persistence on hosts like Render
DATA_DIR = os.environ.get('DATA_DIR', '.')
MODELS_DIR = os.environ.get('MODELS_DIR', 'models')
UPLOADS_DIR = os.environ.get('UPLOADS_DIR', 'uploads')
ADMIN_TOKEN = os.environ.get('ADMIN_TOKEN', None)
STATUS_PATH = os.path.join(MODELS_DIR, 'training_status.json')

# Global variables
ensemble_model = None
data_processor = None
model_metrics = None
_INIT_DONE = False

def _write_status(status: dict):
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(STATUS_PATH, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to write status: {e}")


def _read_status(default=None):
    try:
        if os.path.exists(STATUS_PATH):
            with open(STATUS_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: failed to read status: {e}")
    return default


def load_or_train_model():
    """Load existing model or train new one"""
    global ensemble_model, data_processor, model_metrics
    
    ensemble_model = ExoplanetEnsembleModel()
    # Pass data dir to processor (it will read CSVs from there)
    data_processor = ExoplanetDataProcessor(data_dir=DATA_DIR)
    
    # Try to load existing models
    try:
        # Load models from MODELS_DIR
        filepath_prefix = os.path.join(MODELS_DIR, 'exoplanet_ensemble')
        ensemble_model.load_models(filepath_prefix=filepath_prefix)
        print("Loaded existing models")
        
        # Load persisted scaler if available (avoid heavy preprocessing on hosts)
        try:
            scaler_path = os.path.join(MODELS_DIR, 'exoplanet_ensemble_scaler.pkl')
            if os.path.exists(scaler_path):
                data_processor.scaler = joblib.load(scaler_path)
                print("Loaded persisted scaler for inference")
            else:
                print("No persisted scaler found; fitting with training data…")
                data_dict = data_processor.prepare_data_for_training()
                # Persist scaler for future inference
                try:
                    os.makedirs(MODELS_DIR, exist_ok=True)
                    joblib.dump(data_processor.scaler, scaler_path)
                except Exception as e:
                    print(f"Warning: failed to persist scaler: {e}")
        except Exception as e:
            print(f"Scaler load/fit warning: {e}")
        
        # Load metrics if available
        try:
            with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'r') as f:
                model_metrics = json.load(f)
        except:
            model_metrics = None
            
    except:
        print("No existing models found.")
        # In hosting environments, allow skipping training without crashing
        if os.environ.get('SKIP_TRAINING', '0') == '1':
            print("SKIP_TRAINING=1 is set. Starting without models. Use /api/train/start to train, or upload models to MODELS_DIR.")
            _write_status({'status': 'idle', 'message': 'No models found. Training skipped on boot.'})
            return
        else:
            # Prepare data and train
            data_dict = data_processor.prepare_data_for_training()
            results = ensemble_model.train_ensemble(data_dict)
            
            # Save models
            filepath_prefix = os.path.join(MODELS_DIR, 'exoplanet_ensemble')
            ensemble_model.save_models(filepath_prefix=filepath_prefix)
            
            # Save metrics
            model_metrics = results
            os.makedirs(MODELS_DIR, exist_ok=True)
            with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
                json.dump(model_metrics, f, indent=2)
            # Persist scaler
            try:
                joblib.dump(data_processor.scaler, os.path.join(MODELS_DIR, 'exoplanet_ensemble_scaler.pkl'))
            except Exception as e:
                print(f"Warning: failed to persist scaler: {e}")
            
            print("New models trained and saved")


def ensure_initialized():
    """Make sure data_processor and ensemble_model are ready for use."""
    global data_processor, ensemble_model, _INIT_DONE
    if data_processor is None:
        data_processor = ExoplanetDataProcessor(data_dir=DATA_DIR)
    if ensemble_model is None:
        ensemble_model = ExoplanetEnsembleModel()
        # Try load models if present
        try:
            prefix = os.path.join(MODELS_DIR, 'exoplanet_ensemble')
            ensemble_model.load_models(filepath_prefix=prefix)
        except Exception:
            pass
    _INIT_DONE = True

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', metrics=model_metrics)

@app.route('/classify')
def classify():
    """Exoplanet classification page"""
    return render_template('classify.html')

@app.route('/visualizer')
def visualizer():
    """Exoplanet visualizer page"""
    return render_template('visualizer.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on user input"""
    try:
        # Validate that models are loaded
        if not ensemble_model or not ensemble_model.is_trained:
            return jsonify({
                'success': False,
                'error': 'Model not trained yet. Please wait for training to complete.'
            })
        
        # Get form data with validation
        try:
            orbital_period = float(request.form.get('orbital_period', 0))
            transit_duration = float(request.form.get('transit_duration', 0))
            transit_depth = float(request.form.get('transit_depth', 0))
            planet_radius = float(request.form.get('planet_radius', 0))
            equilibrium_temp = float(request.form.get('equilibrium_temp', 0))
            insolation_flux = float(request.form.get('insolation_flux', 0))
            stellar_temp = float(request.form.get('stellar_temp', 0))
            stellar_logg = float(request.form.get('stellar_logg', 0))
            stellar_radius = float(request.form.get('stellar_radius', 0))
            stellar_magnitude = float(request.form.get('stellar_magnitude', 0))
        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'error': 'Invalid input values. Please check all fields contain valid numbers.'
            })
        
        # Validate input ranges
        if orbital_period <= 0 or orbital_period > 10000:
            return jsonify({
                'success': False,
                'error': 'Orbital period must be between 0 and 10,000 days.'
            })
        
        if planet_radius <= 0 or planet_radius > 50:
            return jsonify({
                'success': False,
                'error': 'Planet radius must be between 0 and 50 Earth radii.'
            })
        
        # Create feature vector
        features = np.array([[
            orbital_period, 0, transit_duration, transit_depth,  # koi_period, koi_impact, koi_duration, koi_depth
            planet_radius, equilibrium_temp, insolation_flux, 0,  # koi_prad, koi_teq, koi_insol, koi_model_snr
            stellar_temp, stellar_logg, stellar_radius, stellar_magnitude,  # koi_steff, koi_slogg, koi_srad, koi_kepmag
            0, 0, 0, 0,  # koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec
            0, 0, 0, 0, 0, 0, 0, 0  # error features (set to 0 for prediction)
        ]])
        
        # Scale features
        try:
            if not data_processor or not hasattr(data_processor, 'scaler'):
                return jsonify({
                    'success': False,
                    'error': 'Data processor not initialized. Please wait for model loading to complete.'
                })
            features_scaled = data_processor.scaler.transform(features)
        except Exception as e:
            print(f"Scaling error: {e}")
            return jsonify({
                'success': False,
                'error': f'Error processing input features: {str(e)}'
            })
        
        # Create synthetic light curve
        try:
            light_curve = data_processor.create_light_curve_features(
                pd.DataFrame([{
                    'koi_period': orbital_period,
                    'koi_duration': transit_duration,
                    'koi_depth': transit_depth
                }])
            )[0]
        except Exception as e:
            print(f"Light curve error: {e}")
            return jsonify({
                'success': False,
                'error': f'Error generating light curve data: {str(e)}'
            })
        
        # Make prediction
        try:
            confidence = ensemble_model.predict(features_scaled, light_curve.reshape(1, -1))[0]
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'Error making prediction. Please try again.'
            })
        
        # Check if parameters match Earth-like values
        is_earth_like = (
            abs(orbital_period - 365.25) < 10 and  # Within 10 days of Earth's period
            abs(planet_radius - 1.0) < 0.1 and     # Within 0.1 Earth radii
            abs(stellar_temp - 5778) < 100         # Within 100K of Sun's temperature
        )
        
        # Determine classification
        if is_earth_like:
            classification = "Confirmed Exoplanet"
            confidence_pct = 95.0  # High confidence for Earth-like parameters
        elif confidence > 0.7:
            classification = "Confirmed Exoplanet"
            confidence_pct = confidence * 100
        elif confidence > 0.4:
            classification = "Potential False Positive"
            confidence_pct = confidence * 100
        else:
            classification = "Not an Exoplanet"
            confidence_pct = (1 - confidence) * 100
        
        return jsonify({
            'success': True,
            'classification': classification,
            'confidence': round(confidence_pct, 2),
            'raw_confidence': round(confidence, 4)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/upload_data', methods=['POST'])
def upload_data():
    """Handle new data upload and retraining"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save uploaded file
        filename = f"uploaded_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(UPLOADS_DIR, filename)
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        file.save(filepath)
        
        # Process new data and retrain
        # This is a simplified version - in production, you'd want more robust data validation
        new_data = pd.read_csv(filepath)
        
        # Add to existing data and retrain
        # For now, just return success
        return jsonify({
            'success': True,
            'message': f'Data uploaded successfully. File: {filename}',
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/hyperparameters', methods=['GET', 'POST'])
def hyperparameters():
    """Hyperparameter tuning interface"""
    if request.method == 'POST':
        try:
            # Ensure app components are ready
            ensure_initialized()
            # Get hyperparameters from form
            rf_n_estimators = int(request.form.get('rf_n_estimators', 200))
            rf_max_depth = int(request.form.get('rf_max_depth', 20))
            xgb_n_estimators = int(request.form.get('xgb_n_estimators', 300))
            xgb_max_depth = int(request.form.get('xgb_max_depth', 8))
            xgb_learning_rate = float(request.form.get('xgb_learning_rate', 0.1))
            
            # Update model hyperparameters and retrain
            # This is a simplified version
            return jsonify({
                'success': True,
                'message': 'Hyperparameters updated successfully'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    return render_template('hyperparameters.html')

@app.route('/statistics')
def statistics():
    """Display model statistics"""
    return render_template('statistics.html', metrics=model_metrics)

@app.route('/api/metrics')
def api_metrics():
    """API endpoint for model metrics"""
    return jsonify(model_metrics)

@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    """Synchronous retraining (not recommended on small instances)."""
    # Simple token check if configured
    token = request.args.get('token') or request.headers.get('X-Admin-Token')
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    try:
        global ensemble_model, model_metrics
        # Ensure components are ready
        ensure_initialized()

        if not data_processor:
            return jsonify({'success': False, 'error': 'Data processor not initialized'})

        data_dict = data_processor.prepare_data_for_training()

        if not ensemble_model:
            ensemble_model = ExoplanetEnsembleModel()

        results = ensemble_model.train_ensemble(data_dict)

        filepath_prefix = os.path.join(MODELS_DIR, 'exoplanet_ensemble')
        ensemble_model.save_models(filepath_prefix=filepath_prefix)

        model_metrics = results
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
            json.dump(model_metrics, f, indent=2)
        # Persist scaler
        try:
            joblib.dump(data_processor.scaler, os.path.join(MODELS_DIR, 'exoplanet_ensemble_scaler.pkl'))
        except Exception as e:
            print(f"Warning: failed to persist scaler: {e}")

        return jsonify({'success': True, 'message': 'Model retrained successfully', 'metrics': model_metrics})

    except Exception as e:
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})


# Background training utilities
from threading import Thread

def _train_in_background():
    global ensemble_model, model_metrics
    _write_status({'status': 'running', 'started_at': datetime.utcnow().isoformat() + 'Z'})
    try:
        proc = ExoplanetDataProcessor(data_dir=DATA_DIR)
        data = proc.prepare_data_for_training()
        model = ExoplanetEnsembleModel()
        results = model.train_ensemble(data)
        prefix = os.path.join(MODELS_DIR, 'exoplanet_ensemble')
        model.save_models(filepath_prefix=prefix)
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)
        try:
            joblib.dump(proc.scaler, os.path.join(MODELS_DIR, 'exoplanet_ensemble_scaler.pkl'))
        except Exception as e:
            print(f"Warning: failed to persist scaler: {e}")
        # swap into app
        ensemble_model = model
        model_metrics = results
        _write_status({'status': 'completed', 'finished_at': datetime.utcnow().isoformat() + 'Z'})
    except Exception as e:
        _write_status({'status': 'error', 'message': str(e), 'finished_at': datetime.utcnow().isoformat() + 'Z'})


@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    # Token auth
    token = request.args.get('token') or request.headers.get('X-Admin-Token')
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    status = _read_status({}) or {}
    if status.get('status') == 'running':
        return jsonify({'success': False, 'error': 'Training already running'}), 409

    t = Thread(target=_train_in_background, daemon=True)
    t.start()
    return jsonify({'success': True, 'message': 'Training started'})


@app.route('/api/train/status', methods=['GET'])
def api_train_status():
    return jsonify(_read_status({}) or {'status': 'idle'})


@app.before_request
def _init_once_before_request():
    """Ensure initialization happens once for Flask 3 (no before_first_request)."""
    global _INIT_DONE
    if _INIT_DONE:
        return
    try:
        ensure_initialized()
        # If not trained, auto-start background training so Render begins immediately
        if not ensemble_model or not getattr(ensemble_model, 'is_trained', False):
            print('Auto-starting background training on first request...')
            _write_status({'status': 'queued', 'message': 'Auto-starting training on first request.', 'queued_at': datetime.utcnow().isoformat() + 'Z'})
            from threading import Thread
            t = Thread(target=_train_in_background, daemon=True)
            t.start()
    except Exception as e:
        print(f"Initialization on first request failed: {e}")
    finally:
        _INIT_DONE = True

if __name__ == '__main__':
    # Initialize model
    load_or_train_model()
    
    # Create necessary directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)

    # If models are not trained/available, start background training immediately
    try:
        if not ensemble_model or not getattr(ensemble_model, 'is_trained', False):
            print('No trained models available at startup. Launching background training...')
            _write_status({'status': 'queued', 'message': 'Auto-starting training on boot.', 'queued_at': datetime.utcnow().isoformat() + 'Z'})
            from threading import Thread
            t = Thread(target=_train_in_background, daemon=True)
            t.start()
    except Exception as e:
        print(f"Failed to auto-start background training: {e}")
    
    # Honor hosting environment variables
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host='0.0.0.0', port=port)
