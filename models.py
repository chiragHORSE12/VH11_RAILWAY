import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import xgboost as xgb
import pickle

class DelayPredictor:
    """
    Machine learning model for predicting train delays based on operational features.
    Supports both Random Forest and XGBoost algorithms.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the delay predictor.
        
        Args:
            model_type (str): Type of model to use ('random_forest' or 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.feature_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        # Initialize the model based on type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError("Model type must be 'random_forest' or 'xgboost'")
    
    def _prepare_features(self, data, fit_encoders=False):
        """
        Prepare features for training or prediction.
        
        Args:
            data (pd.DataFrame): Input data
            fit_encoders (bool): Whether to fit new encoders (True for training)
            
        Returns:
            np.ndarray: Prepared feature matrix
        """
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Define categorical and numerical features
        categorical_features = ['train_type', 'day_of_week', 'weather_severity']
        numerical_features = [
            'upstream_delay', 'passenger_load_percentage', 'scheduled_headway', 'hour'
        ]
        boolean_features = ['is_holiday', 'platform_available', 'crew_available', 'is_peak_hour']
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in df.columns:
                if fit_encoders:
                    encoder = LabelEncoder()
                    df[feature] = encoder.fit_transform(df[feature].astype(str))
                    self.feature_encoders[feature] = encoder
                else:
                    if feature in self.feature_encoders:
                        # Handle unseen categories
                        encoder = self.feature_encoders[feature]
                        df[feature] = df[feature].astype(str)
                        # Map unseen categories to a default value (0)
                        df[feature] = df[feature].apply(
                            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0
                        )
                    else:
                        df[feature] = 0  # Default value if encoder doesn't exist
        
        # Convert boolean features to integers
        for feature in boolean_features:
            if feature in df.columns:
                df[feature] = df[feature].astype(int)
        
        # Select final features
        feature_columns = categorical_features + numerical_features + boolean_features
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns].values
        
        # Store feature names for later use
        if fit_encoders:
            self.feature_names = feature_columns
        
        # Scale features for XGBoost (Random Forest doesn't require scaling)
        if self.model_type == 'xgboost':
            if fit_encoders:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
        
        return X
    
    def train(self, data, test_size=0.2, random_state=42):
        """
        Train the delay prediction model.
        
        Args:
            data (pd.DataFrame): Training data with features and 'actual_delay' target
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Training results including metrics and predictions
        """
        # Prepare features and target
        X = self._prepare_features(data, fit_encoders=True)
        y = data['actual_delay'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results = {
            'model_type': self.model_type,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'mae': test_mae,  # For backward compatibility
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'rmse': test_rmse,  # For backward compatibility
            'train_r2': train_r2,
            'test_r2': test_r2,
            'r2_score': test_r2,  # For backward compatibility
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_pred': y_pred_test,  # For backward compatibility
            'feature_importance': self.get_feature_importance()
        }
        
        return results
    
    def predict(self, data):
        """
        Predict delays for new data.
        
        Args:
            data (pd.DataFrame): New data to predict
            
        Returns:
            np.ndarray: Predicted delays
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self._prepare_features(data, fit_encoders=False)
        predictions = self.model.predict(X)
        
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model.
        
        Returns:
            dict: Feature names and their importance scores
        """
        if not self.is_trained:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
            if self.feature_names:
                return dict(zip(self.feature_names, importance_scores))
        
        return {}
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_encoders': self.feature_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a trained model from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_encoders = model_data['feature_encoders']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']


class ActionClassifier:
    """
    Machine learning model for classifying recommended actions based on operational features.
    Supports both Random Forest and XGBoost algorithms.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the action classifier.
        
        Args:
            model_type (str): Type of model to use ('random_forest' or 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.feature_encoders = {}
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        # Initialize the model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError("Model type must be 'random_forest' or 'xgboost'")
    
    def _prepare_features(self, data, fit_encoders=False):
        """
        Prepare features for training or prediction.
        
        Args:
            data (pd.DataFrame): Input data
            fit_encoders (bool): Whether to fit new encoders (True for training)
            
        Returns:
            np.ndarray: Prepared feature matrix
        """
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Include actual_delay as a feature for action classification
        categorical_features = ['train_type', 'day_of_week', 'weather_severity']
        numerical_features = [
            'upstream_delay', 'passenger_load_percentage', 'scheduled_headway', 
            'hour', 'actual_delay'
        ]
        boolean_features = ['is_holiday', 'platform_available', 'crew_available', 'is_peak_hour']
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in df.columns:
                if fit_encoders:
                    encoder = LabelEncoder()
                    df[feature] = encoder.fit_transform(df[feature].astype(str))
                    self.feature_encoders[feature] = encoder
                else:
                    if feature in self.feature_encoders:
                        encoder = self.feature_encoders[feature]
                        df[feature] = df[feature].astype(str)
                        df[feature] = df[feature].apply(
                            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0
                        )
                    else:
                        df[feature] = 0
        
        # Convert boolean features to integers
        for feature in boolean_features:
            if feature in df.columns:
                df[feature] = df[feature].astype(int)
        
        # Select final features
        feature_columns = categorical_features + numerical_features + boolean_features
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns].values
        
        # Store feature names for later use
        if fit_encoders:
            self.feature_names = feature_columns
        
        # Scale features for XGBoost
        if self.model_type == 'xgboost':
            if fit_encoders:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
        
        return X
    
    def train(self, data, test_size=0.2, random_state=42):
        """
        Train the action classification model.
        
        Args:
            data (pd.DataFrame): Training data with features and 'recommended_action' target
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Training results including metrics and predictions
        """
        # Prepare features and target
        X = self._prepare_features(data, fit_encoders=True)
        y = self.label_encoder.fit_transform(data['recommended_action'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        # Classification report and confusion matrix
        labels = self.label_encoder.classes_
        classification_rep = classification_report(
            y_test, y_pred_test, target_names=labels
        )
        conf_matrix = confusion_matrix(y_test, y_pred_test)
        
        results = {
            'model_type': self.model_type,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'accuracy': test_accuracy,  # For backward compatibility
            'f1_score': test_f1,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix,
            'labels': labels,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'feature_importance': self.get_feature_importance()
        }
        
        return results
    
    def predict(self, data):
        """
        Predict actions for new data.
        
        Args:
            data (pd.DataFrame): New data to predict
            
        Returns:
            np.ndarray: Predicted actions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self._prepare_features(data, fit_encoders=False)
        predictions = self.model.predict(X)
        
        # Convert back to original labels
        predicted_actions = self.label_encoder.inverse_transform(predictions)
        
        return predicted_actions
    
    def predict_proba(self, data):
        """
        Predict action probabilities for new data.
        
        Args:
            data (pd.DataFrame): New data to predict
            
        Returns:
            np.ndarray: Predicted probabilities for each action
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self._prepare_features(data, fit_encoders=False)
        probabilities = self.model.predict_proba(X)
        
        return probabilities
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model.
        
        Returns:
            dict: Feature names and their importance scores
        """
        if not self.is_trained:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
            if self.feature_names:
                return dict(zip(self.feature_names, importance_scores))
        
        return {}
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_encoders': self.feature_encoders,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a trained model from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_encoders = model_data['feature_encoders']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
