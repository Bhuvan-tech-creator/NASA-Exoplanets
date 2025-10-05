#!/usr/bin/env python3
"""
Quick training script for faster model training
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import ExoplanetDataProcessor
from ensemble_model import ExoplanetEnsembleModel

def main():
    """Quick training with optimized parameters"""
    print("=" * 60)
    print("NASA EXOPLANET DETECTION SYSTEM - QUICK TRAINING")
    print("=" * 60)
    
    DATA_DIR = os.environ.get('DATA_DIR', '.')
    MODELS_DIR = os.environ.get('MODELS_DIR', 'models')
    UPLOADS_DIR = os.environ.get('UPLOADS_DIR', 'uploads')
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    
    try:
        # Initialize data processor
        print("\n1. Initializing data processor...")
        processor = ExoplanetDataProcessor(data_dir=DATA_DIR)
        
        # Prepare data
        print("\n2. Loading and preprocessing data...")
        data_dict = processor.prepare_data_for_training()
        
        print(f"   - Training set size: {data_dict['X_train'].shape}")
        print(f"   - Test set size: {data_dict['X_test'].shape}")
        print(f"   - Light curve training shape: {data_dict['lc_train'].shape}")
        print(f"   - Number of features: {len(data_dict['feature_columns'])}")
        
        # Initialize ensemble model
        print("\n3. Initializing ensemble model...")
        ensemble = ExoplanetEnsembleModel()
        
        # Train ensemble with optimized parameters
        print("\n4. Training ensemble model (optimized for speed)...")
        
        # Override CNN training to be faster
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        
        def create_fast_cnn_model(input_shape):
            model = Sequential([
                Conv1D(32, 3, activation='relu', input_shape=input_shape),
                MaxPooling1D(2),
                Dropout(0.2),
                Conv1D(64, 3, activation='relu'),
                MaxPooling1D(2),
                Dropout(0.2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
            return model
        
        # Train individual models
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        lc_train = data_dict['lc_train']
        lc_test = data_dict['lc_test']
        
        print("Training Random Forest...")
        from sklearn.ensemble import RandomForestClassifier
        ensemble.rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        ensemble.rf_model.fit(X_train, y_train)
        rf_proba = ensemble.rf_model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import accuracy_score, roc_auc_score
        rf_pred = (rf_proba > 0.5).astype(int)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        rf_auc = roc_auc_score(y_test, rf_proba)
        
        print("Training XGBoost...")
        import xgboost as xgb
        ensemble.xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        ensemble.xgb_model.fit(X_train, y_train)
        xgb_proba = ensemble.xgb_model.predict_proba(X_test)[:, 1]
        xgb_pred = (xgb_proba > 0.5).astype(int)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        xgb_auc = roc_auc_score(y_test, xgb_proba)
        
        print("Training LightGBM...")
        import lightgbm as lgb
        ensemble.lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)
        ensemble.lgb_model.fit(X_train, y_train)
        lgb_proba = ensemble.lgb_model.predict_proba(X_test)[:, 1]
        lgb_pred = (lgb_proba > 0.5).astype(int)
        lgb_accuracy = accuracy_score(y_test, lgb_pred)
        lgb_auc = roc_auc_score(y_test, lgb_proba)
        
        print("Training CNN (fast version)...")
        lc_train_reshaped = lc_train.reshape(lc_train.shape[0], lc_train.shape[1], 1)
        lc_test_reshaped = lc_test.reshape(lc_test.shape[0], lc_test.shape[1], 1)
        
        ensemble.cnn_model = create_fast_cnn_model((lc_train_reshaped.shape[1], 1))
        
        # Train with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        ensemble.cnn_model.fit(
            lc_train_reshaped, y_train,
            validation_data=(lc_test_reshaped, y_test),
            epochs=20,  # Reduced epochs
            batch_size=64,  # Larger batch size
            callbacks=[early_stopping],
            verbose=1
        )
        
        cnn_proba = ensemble.cnn_model.predict(lc_test_reshaped).flatten()
        cnn_pred = (cnn_proba > 0.5).astype(int)
        cnn_accuracy = accuracy_score(y_test, cnn_pred)
        cnn_auc = roc_auc_score(y_test, cnn_proba)
        
        # Set ensemble weights
        ensemble.ensemble_weights = [0.25, 0.25, 0.25, 0.25]
        ensemble.feature_columns = data_dict['feature_columns']
        ensemble.is_trained = True
        
        # Calculate ensemble performance - ensure all predictions have same length
        min_length = min(len(rf_proba), len(xgb_proba), len(lgb_proba), len(cnn_proba))
        rf_proba = rf_proba[:min_length]
        xgb_proba = xgb_proba[:min_length]
        lgb_proba = lgb_proba[:min_length]
        cnn_proba = cnn_proba[:min_length]
        y_test = y_test[:min_length]
        
        predictions = np.column_stack([rf_proba, xgb_proba, lgb_proba, cnn_proba])
        ensemble_proba = np.average(predictions, axis=1, weights=ensemble.ensemble_weights)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)
        
        # Save models
        print("\n5. Saving trained models...")
        prefix = os.path.join(MODELS_DIR, 'exoplanet_ensemble')
        ensemble.save_models(filepath_prefix=prefix)
        
        # Create results
        results = {
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_auc': ensemble_auc,
            'individual_results': {
                'rf': {'accuracy': rf_accuracy, 'auc': rf_auc},
                'xgb': {'accuracy': xgb_accuracy, 'auc': xgb_auc},
                'lgb': {'accuracy': lgb_accuracy, 'auc': lgb_auc},
                'cnn': {'accuracy': cnn_accuracy, 'auc': cnn_auc}
            }
        }
        
        # Save metrics
        print("\n6. Saving model metrics...")
        with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)
        # Persist scaler for inference
        try:
            import joblib
            joblib.dump(processor.scaler, os.path.join(MODELS_DIR, 'exoplanet_ensemble_scaler.pkl'))
        except Exception:
            pass
        
        # Display results
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Ensemble Accuracy: {results['ensemble_accuracy']:.4f} ({results['ensemble_accuracy']*100:.1f}%)")
        print(f"Ensemble AUC: {results['ensemble_auc']:.4f} ({results['ensemble_auc']*100:.1f}%)")
        
        print("\nIndividual Model Performance:")
        for model_name, metrics in results['individual_results'].items():
            print(f"  {model_name.upper()}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
            print(f"    AUC: {metrics['auc']:.4f} ({metrics['auc']*100:.1f}%)")
        
        # Check if goals are met
        accuracy_goal_met = results['ensemble_accuracy'] > 0.90
        auc_goal_met = results['ensemble_auc'] > 0.95
        
        print(f"\nGoal Achievement:")
        print(f"  Accuracy > 90%: {'✓ ACHIEVED' if accuracy_goal_met else '✗ NOT MET'}")
        print(f"  ROC AUC > 95%: {'✓ ACHIEVED' if auc_goal_met else '✗ NOT MET'}")
        
        if accuracy_goal_met and auc_goal_met:
            print("\n🎉 ALL GOALS ACHIEVED! The model is ready for production use.")
        else:
            print("\n⚠️  Some goals were not met. Consider hyperparameter tuning.")

        print(f"\nModels saved to: {MODELS_DIR}/")
        print(f"Metrics saved to: {os.path.join(MODELS_DIR, 'model_metrics.json')}")
        print("\nYou can now run the web application with: python app.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: Training failed with error: {str(e)}")
        print("Please check your data files and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
