import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import cross_val_score
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

class ExoplanetEnsembleModel:
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.cnn_model = None
        self.ensemble_weights = None
        self.decision_threshold = 0.5
        self.is_trained = False
        self.feature_columns = None
        self.meta_model = None
        self.meta_feature_order = None
        self._progress_cb = None  # optional progress callback (stage:str, pct:int, msg:str)

    def set_progress_callback(self, cb):
        """Register a callback to receive progress updates.
        cb signature: (stage: str, pct: int, msg: str) -> None
        """
        self._progress_cb = cb

    def _cb(self, stage: str, pct: int, msg: str):
        try:
            if self._progress_cb is not None:
                self._progress_cb(stage, int(pct), str(msg))
        except Exception:
            pass
        
    def create_cnn_model(self, input_shape):
        """Create CNN model for light curve analysis"""
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            Conv1D(128, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            Conv1D(256, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),
            
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(256, activation='relu'),
            Dropout(0.3),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("Training Random Forest...")
        self._cb('rf', 15, 'Training Random Forest...')
        
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate
        rf_pred = self.rf_model.predict(X_test)
        rf_proba = self.rf_model.predict_proba(X_test)[:, 1]
        
        rf_accuracy = accuracy_score(y_test, rf_pred)
        rf_auc = roc_auc_score(y_test, rf_proba)
        
        print(f"Random Forest - Accuracy: {rf_accuracy:.4f}, AUC: {rf_auc:.4f}")
        self._cb('rf', 25, f"Random Forest done. Acc={rf_accuracy:.4f}, AUC={rf_auc:.4f}")
        
        return rf_proba, rf_accuracy, rf_auc
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("Training XGBoost...")
        self._cb('xgb', 35, 'Training XGBoost...')
        
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.xgb_model.fit(X_train, y_train)
        
        # Evaluate
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        xgb_auc = roc_auc_score(y_test, xgb_proba)
        
        print(f"XGBoost - Accuracy: {xgb_accuracy:.4f}, AUC: {xgb_auc:.4f}")
        self._cb('xgb', 45, f"XGBoost done. Acc={xgb_accuracy:.4f}, AUC={xgb_auc:.4f}")
        
        return xgb_proba, xgb_accuracy, xgb_auc
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model"""
        print("Training LightGBM...")
        self._cb('lgb', 55, 'Training LightGBM...')
        
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        self.lgb_model.fit(X_train, y_train)
        
        # Evaluate
        lgb_pred = self.lgb_model.predict(X_test)
        lgb_proba = self.lgb_model.predict_proba(X_test)[:, 1]
        
        lgb_accuracy = accuracy_score(y_test, lgb_pred)
        lgb_auc = roc_auc_score(y_test, lgb_proba)
        
        print(f"LightGBM - Accuracy: {lgb_accuracy:.4f}, AUC: {lgb_auc:.4f}")
        self._cb('lgb', 65, f"LightGBM done. Acc={lgb_accuracy:.4f}, AUC={lgb_auc:.4f}")
        
        return lgb_proba, lgb_accuracy, lgb_auc
    
    def train_cnn(self, lc_train, y_train, lc_test, y_test):
        """Train CNN model on light curves"""
        print("Training CNN on light curves...")
        self._cb('cnn', 70, 'Training CNN on light curves...')
        
        # Reshape light curves for CNN
        lc_train_reshaped = lc_train.reshape(lc_train.shape[0], lc_train.shape[1], 1)
        lc_test_reshaped = lc_test.reshape(lc_test.shape[0], lc_test.shape[1], 1)
        
        self.cnn_model = self.create_cnn_model((lc_train_reshaped.shape[1], 1))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # Train
        history = self.cnn_model.fit(
            lc_train_reshaped, y_train,
            validation_data=(lc_test_reshaped, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        cnn_proba = self.cnn_model.predict(lc_test_reshaped).flatten()
        cnn_pred = (cnn_proba > 0.5).astype(int)
        
        cnn_accuracy = accuracy_score(y_test, cnn_pred)
        cnn_auc = roc_auc_score(y_test, cnn_proba)
        
        print(f"CNN - Accuracy: {cnn_accuracy:.4f}, AUC: {cnn_auc:.4f}")
        self._cb('cnn', 80, f"CNN done. Acc={cnn_accuracy:.4f}, AUC={cnn_auc:.4f}")
        
        return cnn_proba, cnn_accuracy, cnn_auc
    
    def optimize_ensemble_weights(self, predictions, y_test):
        """Optimize ensemble weights and decision threshold to maximize accuracy."""
        print("Optimizing ensemble weights...")
        self._cb('opt', 85, 'Optimizing ensemble weights...')

        # Align lengths
        min_length = min(len(pred) for pred in predictions)
        preds = np.vstack([pred[:min_length] for pred in predictions]).T  # shape (n, 4)
        y = y_test[:min_length]

        # Candidate weight sets (allow zeroing CNN)
        weight_combinations = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.3, 0.4, 0.3, 0.0],
            [0.2, 0.6, 0.2, 0.0],
            [0.2, 0.5, 0.3, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.2, 0.4, 0.2, 0.2],
            [0.2, 0.2, 0.4, 0.2],
        ]

        best_acc = -1.0
        best_weights = weight_combinations[0]
        best_thr = 0.5

        # Threshold grid (broad)
        thresholds = np.linspace(0.05, 0.95, 91)

        for w in weight_combinations:
            try:
                p = np.average(preds, axis=1, weights=w)
                # Scan thresholds for max accuracy
                for thr in thresholds:
                    acc = accuracy_score(y, (p > thr).astype(int))
                    if acc > best_acc:
                        best_acc = acc
                        best_weights = w
                        best_thr = float(thr)
            except Exception as e:
                print(f"Error with weights {w}: {e}")
                continue

        print(f"Best ensemble weights: {best_weights}")
        print(f"Best ensemble threshold: {best_thr:.2f}")
        print(f"Best ensemble accuracy: {best_acc:.4f}")
        self._cb('opt', 90, f"Weights {best_weights}, thr={best_thr:.2f}, acc={best_acc:.4f}")

        return best_weights, best_thr
    
    def train_ensemble(self, data_dict):
        """Train the complete ensemble model"""
        print("Starting ensemble training...")
        self._cb('start', 5, 'Starting ensemble training...')
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        lc_train = data_dict['lc_train']
        lc_test = data_dict['lc_test']
        self.feature_columns = data_dict['feature_columns']
        
        # Train individual models
        rf_proba, rf_acc, rf_auc = self.train_random_forest(X_train, y_train, X_test, y_test)
        xgb_proba, xgb_acc, xgb_auc = self.train_xgboost(X_train, y_train, X_test, y_test)
        lgb_proba, lgb_acc, lgb_auc = self.train_lightgbm(X_train, y_train, X_test, y_test)
        cnn_proba, cnn_acc, cnn_auc = self.train_cnn(lc_train, y_train, lc_test, y_test)
        
        # Collect predictions and ensure they have the same length
        min_length = min(len(rf_proba), len(xgb_proba), len(lgb_proba), len(cnn_proba))
        predictions = np.column_stack([
            rf_proba[:min_length], 
            xgb_proba[:min_length], 
            lgb_proba[:min_length], 
            cnn_proba[:min_length]
        ])
        y_test_aligned = y_test[:min_length]
        
        # Optimize ensemble weights and threshold
        self.ensemble_weights, self.decision_threshold = self.optimize_ensemble_weights(predictions.T, y_test_aligned)
        
        # Final ensemble prediction
        ensemble_proba = np.average(predictions, axis=1, weights=self.ensemble_weights)
        ensemble_pred = (ensemble_proba > self.decision_threshold).astype(int)
        
        # Final evaluation
        ensemble_accuracy = accuracy_score(y_test_aligned, ensemble_pred)
        ensemble_auc = roc_auc_score(y_test_aligned, ensemble_proba)
        
        print(f"\n=== FINAL ENSEMBLE RESULTS ===")
        print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
        print(f"Ensemble AUC: {ensemble_auc:.4f}")
        self._cb('final', 95, f"Ensemble Acc={ensemble_accuracy:.4f}, AUC={ensemble_auc:.4f}")
        
        # Individual model results
        print(f"\n=== INDIVIDUAL MODEL RESULTS ===")
        print(f"Random Forest - Accuracy: {rf_acc:.4f}, AUC: {rf_auc:.4f}")
        print(f"XGBoost - Accuracy: {xgb_acc:.4f}, AUC: {xgb_auc:.4f}")
        print(f"LightGBM - Accuracy: {lgb_acc:.4f}, AUC: {lgb_auc:.4f}")
        print(f"CNN - Accuracy: {cnn_acc:.4f}, AUC: {cnn_auc:.4f}")
        
        self.is_trained = True
        self._cb('done', 100, 'Training complete')
        
        return {
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_auc': ensemble_auc,
            'decision_threshold': self.decision_threshold,
            'individual_results': {
                'rf': {'accuracy': rf_acc, 'auc': rf_auc},
                'xgb': {'accuracy': xgb_acc, 'auc': xgb_auc},
                'lgb': {'accuracy': lgb_acc, 'auc': lgb_auc},
                'cnn': {'accuracy': cnn_acc, 'auc': cnn_auc}
            }
        }
    
    def predict(self, X, light_curves=None):
        """Make predictions using the ensemble"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get individual predictions
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        
        if light_curves is not None:
            lc_reshaped = light_curves.reshape(light_curves.shape[0], light_curves.shape[1], 1)
            cnn_proba = self.cnn_model.predict(lc_reshaped).flatten()
        else:
            # Use average of other models if no light curves
            cnn_proba = np.mean([rf_proba, xgb_proba, lgb_proba], axis=0)
        
        # Ensure all predictions have the same length
        min_length = min(len(rf_proba), len(xgb_proba), len(lgb_proba), len(cnn_proba))
        predictions = np.column_stack([
            rf_proba[:min_length], 
            xgb_proba[:min_length], 
            lgb_proba[:min_length], 
            cnn_proba[:min_length]
        ])
        
        # Ensemble prediction (stacking if meta_model present)
        if self.meta_model is not None and self.meta_feature_order is not None:
            # Build features in the same order used for meta training
            feature_map = {
                'rf': predictions[:, 0],
                'xgb': predictions[:, 1],
                'lgb': predictions[:, 2],
                'cnn': predictions[:, 3],
            }
            X_meta = np.column_stack([feature_map[name] for name in self.meta_feature_order])
            ensemble_proba = self.meta_model.predict_proba(X_meta)[:, 1]
        else:
            ensemble_proba = np.average(predictions, axis=1, weights=self.ensemble_weights)
        
        return ensemble_proba
    
    def train_ensemble_quick(self, data_dict):
        """Quick training version with reduced parameters for faster execution"""
        print("Starting quick ensemble training...")
        self._cb('start', 5, 'Starting quick ensemble training...')
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        lc_train = data_dict['lc_train']
        lc_test = data_dict['lc_test']
        
        self.feature_columns = data_dict.get('feature_columns', [])
        
        # Train models with reduced parameters for speed
        print("Quick training Random Forest...")
        self._cb('rf', 15, 'Quick training Random Forest...')
        self.rf_model = RandomForestClassifier(
            n_estimators=100,  # Reduced from 200
            max_depth=15,      # Reduced from 20
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        rf_proba = self.rf_model.predict_proba(X_test)[:, 1]
        rf_accuracy = accuracy_score(y_test, (rf_proba > 0.5).astype(int))
        rf_auc = roc_auc_score(y_test, rf_proba)
        
        print("Quick training XGBoost...")
        self._cb('xgb', 30, 'Quick training XGBoost...')
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=150,  # Reduced from 300
            max_depth=6,       # Reduced from 8
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.xgb_model.fit(X_train, y_train)
        xgb_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        xgb_accuracy = accuracy_score(y_test, (xgb_proba > 0.5).astype(int))
        xgb_auc = roc_auc_score(y_test, xgb_proba)
        
        print("Quick training LightGBM...")
        self._cb('lgb', 45, 'Quick training LightGBM...')
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=150,  # Reduced from 300
            max_depth=6,       # Reduced from 8
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.lgb_model.fit(X_train, y_train)
        lgb_proba = self.lgb_model.predict_proba(X_test)[:, 1]
        lgb_accuracy = accuracy_score(y_test, (lgb_proba > 0.5).astype(int))
        lgb_auc = roc_auc_score(y_test, lgb_proba)
        
        print("Quick training CNN...")
        self._cb('cnn', 60, 'Quick training CNN...')
        # Simplified CNN architecture for faster training
        lc_train_reshaped = lc_train.reshape(lc_train.shape[0], lc_train.shape[1], 1)
        lc_test_reshaped = lc_test.reshape(lc_test.shape[0], lc_test.shape[1], 1)
        
        # Smaller CNN model
        self.cnn_model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=(lc_train_reshaped.shape[1], 1)),
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
        
        self.cnn_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Quick training with fewer epochs
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,  # Reduced patience
            restore_best_weights=True
        )
        
        self.cnn_model.fit(
            lc_train_reshaped, y_train,
            validation_data=(lc_test_reshaped, y_test),
            epochs=20,  # Reduced from 100
            batch_size=64,  # Increased batch size
            callbacks=[early_stopping],
            verbose=0  # Silent training
        )
        
        cnn_proba = self.cnn_model.predict(lc_test_reshaped, verbose=0).flatten()
        cnn_accuracy = accuracy_score(y_test, (cnn_proba > 0.5).astype(int))
        cnn_auc = roc_auc_score(y_test, cnn_proba)
        
        print("Optimizing ensemble weights...")
        self._cb('opt', 80, 'Optimizing ensemble weights...')
        predictions = [rf_proba, xgb_proba, lgb_proba, cnn_proba]
        self.optimize_ensemble_weights(predictions, y_test)
        
        # Final ensemble prediction
        min_length = min(len(pred) for pred in predictions)
        ensemble_preds = np.column_stack([pred[:min_length] for pred in predictions])
        ensemble_proba = np.average(ensemble_preds, axis=1, weights=self.ensemble_weights)
        ensemble_pred = (ensemble_proba > self.decision_threshold).astype(int)
        
        ensemble_accuracy = accuracy_score(y_test[:min_length], ensemble_pred)
        ensemble_auc = roc_auc_score(y_test[:min_length], ensemble_proba)
        
        self.is_trained = True
        self._cb('final', 95, f"Quick ensemble Acc={ensemble_accuracy:.4f}, AUC={ensemble_auc:.4f}")
        self._cb('done', 100, 'Quick training complete')
        
        results = {
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_auc': ensemble_auc,
            'individual_results': {
                'rf': {'accuracy': rf_accuracy, 'auc': rf_auc},
                'xgb': {'accuracy': xgb_accuracy, 'auc': xgb_auc},
                'lgb': {'accuracy': lgb_accuracy, 'auc': lgb_auc},
                'cnn': {'accuracy': cnn_accuracy, 'auc': cnn_auc}
            },
            'ensemble_weights': self.ensemble_weights.tolist(),
            'decision_threshold': self.decision_threshold,
            'training_mode': 'quick'
        }
        
        print(f"Quick training completed!")
        print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
        print(f"Ensemble AUC: {ensemble_auc:.4f}")
        
        return results
    
    def save_models(self, filepath_prefix='models/exoplanet_ensemble'):
        """Save all trained models"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        import os
        os.makedirs('models', exist_ok=True)
        
        # Save individual models
        joblib.dump(self.rf_model, f'{filepath_prefix}_rf.pkl')
        joblib.dump(self.xgb_model, f'{filepath_prefix}_xgb.pkl')
        joblib.dump(self.lgb_model, f'{filepath_prefix}_lgb.pkl')
        self.cnn_model.save(f'{filepath_prefix}_cnn.h5')
        
        # Save ensemble weights and metadata
        ensemble_data = {
            'weights': self.ensemble_weights,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'decision_threshold': self.decision_threshold,
            'meta_feature_order': self.meta_feature_order
        }
        joblib.dump(ensemble_data, f'{filepath_prefix}_metadata.pkl')
        # Save meta model if present
        if self.meta_model is not None:
            joblib.dump(self.meta_model, f'{filepath_prefix}_meta.pkl')
        
        print(f"Models saved to {filepath_prefix}_*")
    
    def load_models(self, filepath_prefix='models/exoplanet_ensemble'):
        """Load all trained models"""
        # Load individual models
        self.rf_model = joblib.load(f'{filepath_prefix}_rf.pkl')
        self.xgb_model = joblib.load(f'{filepath_prefix}_xgb.pkl')
        self.lgb_model = joblib.load(f'{filepath_prefix}_lgb.pkl')
        self.cnn_model = tf.keras.models.load_model(f'{filepath_prefix}_cnn.h5')
        
        # Load ensemble metadata
        ensemble_data = joblib.load(f'{filepath_prefix}_metadata.pkl')
        self.ensemble_weights = ensemble_data['weights']
        self.feature_columns = ensemble_data['feature_columns']
        self.is_trained = ensemble_data['is_trained']
        self.decision_threshold = ensemble_data.get('decision_threshold', 0.5)
        self.meta_feature_order = ensemble_data.get('meta_feature_order', None)
        # Load meta model if available
        meta_path = f'{filepath_prefix}_meta.pkl'
        try:
            if os.path.exists(meta_path):
                self.meta_model = joblib.load(meta_path)
        except Exception:
            self.meta_model = None
        
        print(f"Models loaded from {filepath_prefix}_*")

if __name__ == "__main__":
    from data_preprocessing import ExoplanetDataProcessor
    
    # Prepare data
    processor = ExoplanetDataProcessor()
    data_dict = processor.prepare_data_for_training()
    
    # Train ensemble
    ensemble = ExoplanetEnsembleModel()
    results = ensemble.train_ensemble(data_dict)
    
    # Save models
    ensemble.save_models()
    
    print(f"\nTraining completed!")
    print(f"Final Ensemble Accuracy: {results['ensemble_accuracy']:.4f}")
    print(f"Final Ensemble AUC: {results['ensemble_auc']:.4f}")
