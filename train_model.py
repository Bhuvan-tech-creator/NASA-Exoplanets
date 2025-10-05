#!/usr/bin/env python3
"""
NASA Exoplanet Detection System - Model Training Script
This script trains the ensemble model and saves it for use in the web application.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import ExoplanetDataProcessor
from ensemble_model import ExoplanetEnsembleModel

def main():
    """Main training function"""
    print("=" * 60)
    print("NASA EXOPLANET DETECTION SYSTEM - MODEL TRAINING")
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
        
        # Train ensemble
        print("\n4. Training ensemble model...")
        print("   This may take several minutes depending on your hardware...")
        
        results = ensemble.train_ensemble(data_dict)
        
        # Save models
        print("\n5. Saving trained models...")
        prefix = os.path.join(MODELS_DIR, 'exoplanet_ensemble')
        ensemble.save_models(filepath_prefix=prefix)
        
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
