import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')

class ExoplanetDataProcessor:
    def __init__(self, data_dir: str = '.'):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.data_dir = data_dir
        
    def load_and_combine_data(self):
        """Load and combine Kepler and TESS data"""
        print("Loading Kepler data...")
        kepler_path = os.path.join(self.data_dir, 'cumulative_data.csv')
        kepler_data = pd.read_csv(kepler_path, comment='#')
        
        print("Loading TESS data...")
        tess_path = os.path.join(self.data_dir, 'cumulative_data2.csv')
        tess_data = pd.read_csv(tess_path, comment='#')
        
        # Process Kepler data
        kepler_processed = self._process_kepler_data(kepler_data)
        
        # Process TESS data
        tess_processed = self._process_tess_data(tess_data)
        
        # Combine datasets
        combined_data = pd.concat([kepler_processed, tess_processed], ignore_index=True)
        
        print(f"Combined dataset shape: {combined_data.shape}")
        return combined_data
    
    def _process_kepler_data(self, data):
        """Process Kepler data"""
        print("Processing Kepler data...")
        
        # Create target variable based on disposition
        data['is_exoplanet'] = data['koi_disposition'].map({
            'CONFIRMED': 1,
            'CANDIDATE': 1,
            'FALSE POSITIVE': 0
        })
        
        # Select relevant features (only numeric columns)
        kepler_features = [
            'koi_period', 'koi_impact', 'koi_duration', 'koi_depth',
            'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
            'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag',
            'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
        ]
        
        # Add error features
        error_features = [
            'koi_period_err1', 'koi_period_err2', 'koi_depth_err1', 'koi_depth_err2',
            'koi_prad_err1', 'koi_prad_err2', 'koi_teq_err1', 'koi_teq_err2'
        ]
        
        all_features = kepler_features + error_features + ['is_exoplanet']
        
        # Filter data and ensure numeric columns
        kepler_processed = data[all_features].copy()
        
        # Convert all columns to numeric, replacing non-numeric with NaN
        for col in kepler_processed.columns:
            if col != 'is_exoplanet':
                kepler_processed[col] = pd.to_numeric(kepler_processed[col], errors='coerce')
        
        kepler_processed['mission'] = 'Kepler'
        
        return kepler_processed
    
    def _process_tess_data(self, data):
        """Process TESS data"""
        print("Processing TESS data...")
        
        # Create target variable based on disposition
        data['is_exoplanet'] = data['tfopwg_disp'].map({
            'CP': 1,  # Confirmed Planet
            'PC': 1,  # Planet Candidate
            'FP': 0,  # False Positive
            'KP': 0   # Known Planet (treating as confirmed)
        })
        
        # Select relevant features (mapped to Kepler equivalents)
        tess_features = [
            'pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade',
            'pl_eqt', 'pl_insol', 'st_teff', 'st_logg', 'st_rad', 'st_tmag'
        ]
        
        # Add error features
        error_features = [
            'pl_orbpererr1', 'pl_orbpererr2', 'pl_trandeperr1', 'pl_trandeperr2',
            'pl_radeerr1', 'pl_radeerr2', 'pl_eqterr1', 'pl_eqterr2'
        ]
        
        all_features = tess_features + error_features + ['is_exoplanet']
        
        # Filter data
        tess_processed = data[all_features].copy()
        
        # Convert all columns to numeric, replacing non-numeric with NaN
        for col in tess_processed.columns:
            if col != 'is_exoplanet':
                tess_processed[col] = pd.to_numeric(tess_processed[col], errors='coerce')
        
        tess_processed['mission'] = 'TESS'
        
        # Rename columns to match Kepler format
        column_mapping = {
            'pl_orbper': 'koi_period',
            'pl_trandurh': 'koi_duration',
            'pl_trandep': 'koi_depth',
            'pl_rade': 'koi_prad',
            'pl_eqt': 'koi_teq',
            'pl_insol': 'koi_insol',
            'st_teff': 'koi_steff',
            'st_logg': 'koi_slogg',
            'st_rad': 'koi_srad',
            'st_tmag': 'koi_kepmag',
            'pl_orbpererr1': 'koi_period_err1',
            'pl_orbpererr2': 'koi_period_err2',
            'pl_trandeperr1': 'koi_depth_err1',
            'pl_trandeperr2': 'koi_depth_err2',
            'pl_radeerr1': 'koi_prad_err1',
            'pl_radeerr2': 'koi_prad_err2',
            'pl_eqterr1': 'koi_teq_err1',
            'pl_eqterr2': 'koi_teq_err2'
        }
        
        tess_processed = tess_processed.rename(columns=column_mapping)
        
        # Add missing columns with default values
        missing_cols = ['koi_impact', 'koi_model_snr', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
        for col in missing_cols:
            tess_processed[col] = 0
        
        return tess_processed
    
    def preprocess_features(self, data):
        """Preprocess features for training"""
        print("Preprocessing features...")
        
        # Handle missing values - only for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        
        # Remove rows with missing target
        data = data.dropna(subset=['is_exoplanet'])
        
        # Separate features and target
        feature_cols = [col for col in data.columns if col not in ['is_exoplanet', 'mission']]
        self.feature_columns = feature_cols
        
        print(f"Feature columns: {feature_cols}")
        print(f"Data columns: {list(data.columns)}")
        
        X = data[feature_cols].copy()
        y = data['is_exoplanet'].copy()
        missions = data['mission'].copy()
        
        # Ensure all columns are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y.values, missions.values
    
    def create_light_curve_features(self, data):
        """Create synthetic light curve features for CNN"""
        print("Creating light curve features...")
        
        # Generate synthetic light curve data based on transit parameters
        n_samples = len(data)
        light_curves = []
        
        for idx, row in data.iterrows():
            # Create a synthetic light curve based on transit parameters
            period = row.get('koi_period', 10)  # days
            duration = row.get('koi_duration', 2)  # hours
            depth = row.get('koi_depth', 1000)  # ppm
            
            # Generate time series
            time = np.linspace(0, period * 2, 200)  # 2 periods worth of data
            
            # Create transit signal
            light_curve = np.ones_like(time)
            
            # Add transit dips
            transit_center = period / 2
            transit_half_width = (duration / 24) / 2  # Convert hours to days
            
            transit_mask = (time >= transit_center - transit_half_width) & (time <= transit_center + transit_half_width)
            light_curve[transit_mask] = 1 - (depth / 1e6)  # Convert ppm to fraction
            
            # Add noise
            noise = np.random.normal(0, 0.001, len(time))
            light_curve += noise
            
            # Normalize
            light_curve = (light_curve - np.mean(light_curve)) / np.std(light_curve)
            
            light_curves.append(light_curve)
        
        return np.array(light_curves)
    
    def prepare_data_for_training(self):
        """Main method to prepare all data for training"""
        # Load and combine data
        combined_data = self.load_and_combine_data()
        
        # Preprocess features
        X, y, missions = self.preprocess_features(combined_data)
        
        # Create light curve features
        light_curves = self.create_light_curve_features(combined_data)
        
        # Split data
        X_train, X_test, y_train, y_test, missions_train, missions_test = train_test_split(
            X, y, missions, test_size=0.2, random_state=42, stratify=y
        )
        
        lc_train, lc_test = train_test_split(
            light_curves, test_size=0.2, random_state=42
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'lc_train': lc_train,
            'lc_test': lc_test,
            'missions_train': missions_train,
            'missions_test': missions_test,
            'feature_columns': self.feature_columns
        }
    
    def merge_additional_data(self, csv_path):
        """Merge additional CSV data with existing datasets for enhanced training"""
        print(f"Merging additional data from {csv_path}...")
        
        try:
            # Load the additional data
            additional_data = pd.read_csv(csv_path)
            print(f"Loaded {len(additional_data)} rows from additional dataset")
            
            # Try to map columns to standard format
            column_mapping = {
                # Common variations for period
                'period': 'koi_period',
                'orbital_period': 'koi_period',
                'pl_orbper': 'koi_period',
                
                # Common variations for radius
                'radius': 'koi_prad',
                'planet_radius': 'koi_prad',
                'pl_rade': 'koi_prad',
                
                # Common variations for temperature
                'temperature': 'koi_teq',
                'temp': 'koi_teq',
                'pl_eqt': 'koi_teq',
                
                # Common variations for stellar properties
                'stellar_temp': 'koi_steff',
                'star_temp': 'koi_steff',
                'st_teff': 'koi_steff',
                
                # Common variations for target
                'target': 'is_exoplanet',
                'label': 'is_exoplanet',
                'disposition': 'koi_disposition',
                'confirmed': 'is_exoplanet'
            }
            
            # Apply column mapping
            for old_col, new_col in column_mapping.items():
                if old_col in additional_data.columns:
                    additional_data = additional_data.rename(columns={old_col: new_col})
            
            # Process disposition column if present
            if 'koi_disposition' in additional_data.columns and 'is_exoplanet' not in additional_data.columns:
                additional_data['is_exoplanet'] = additional_data['koi_disposition'].map({
                    'CONFIRMED': 1,
                    'CANDIDATE': 1, 
                    'PC': 1,
                    'CP': 1,
                    'FALSE POSITIVE': 0,
                    'FP': 0,
                    'NOT DISPOSITION': 0,
                    'FALSE_POSITIVE': 0
                })
            
            # Ensure target column exists
            if 'is_exoplanet' not in additional_data.columns:
                print("No target column found. Assuming all samples are positive examples.")
                additional_data['is_exoplanet'] = 1
            
            # Add mission identifier
            additional_data['mission'] = 'Additional'
            
            # Fill missing columns with default values
            required_columns = [
                'koi_period', 'koi_impact', 'koi_duration', 'koi_depth',
                'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
                'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag'
            ]
            
            for col in required_columns:
                if col not in additional_data.columns:
                    # Set reasonable defaults based on column
                    if 'period' in col:
                        additional_data[col] = 100.0
                    elif 'temp' in col or 'steff' in col:
                        additional_data[col] = 5778.0
                    elif 'radius' in col or 'prad' in col:
                        additional_data[col] = 1.0
                    elif 'mag' in col:
                        additional_data[col] = 12.0
                    else:
                        additional_data[col] = 0.0
            
            # Store the additional data for use in training
            self.additional_data = additional_data
            print(f"Successfully prepared {len(additional_data)} additional samples for training")
            
        except Exception as e:
            print(f"Error merging additional data: {e}")
            raise e

if __name__ == "__main__":
    processor = ExoplanetDataProcessor()
    data_dict = processor.prepare_data_for_training()
    
    print(f"Training set size: {data_dict['X_train'].shape}")
    print(f"Test set size: {data_dict['X_test'].shape}")
    print(f"Light curve training shape: {data_dict['lc_train'].shape}")
    print(f"Feature columns: {len(data_dict['feature_columns'])}")
