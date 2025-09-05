import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import os

class DataUtils:
    """
    Utility functions for data handling, processing, and file operations.
    """
    
    @staticmethod
    def save_dataframe(df, filepath, format='csv'):
        """
        Save DataFrame to file in specified format.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filepath (str): Path to save the file
            format (str): File format ('csv', 'json', 'pickle')
        """
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records', indent=2)
        elif format == 'pickle':
            df.to_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def load_dataframe(filepath, format='csv'):
        """
        Load DataFrame from file.
        
        Args:
            filepath (str): Path to the file
            format (str): File format ('csv', 'json', 'pickle')
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        if format == 'csv':
            return pd.read_csv(filepath)
        elif format == 'json':
            return pd.read_json(filepath)
        elif format == 'pickle':
            return pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def validate_train_data(data):
        """
        Validate train operations data for required columns and data types.
        
        Args:
            data (pd.DataFrame): Train operations data
            
        Returns:
            tuple: (is_valid, error_messages)
        """
        required_columns = [
            'train_id', 'train_type', 'day_of_week', 'is_holiday',
            'upstream_delay', 'passenger_load_percentage', 'weather_severity',
            'platform_available', 'crew_available', 'scheduled_headway'
        ]
        
        error_messages = []
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            error_messages.append(f"Missing required columns: {missing_columns}")
        
        # Check data types and ranges
        if 'upstream_delay' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['upstream_delay']):
                error_messages.append("upstream_delay must be numeric")
            elif (data['upstream_delay'] < 0).any():
                error_messages.append("upstream_delay cannot be negative")
        
        if 'passenger_load_percentage' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['passenger_load_percentage']):
                error_messages.append("passenger_load_percentage must be numeric")
            elif ((data['passenger_load_percentage'] < 0) | (data['passenger_load_percentage'] > 100)).any():
                error_messages.append("passenger_load_percentage must be between 0 and 100")
        
        if 'train_type' in data.columns:
            valid_train_types = ['Express', 'Local']
            invalid_types = data[~data['train_type'].isin(valid_train_types)]['train_type'].unique()
            if len(invalid_types) > 0:
                error_messages.append(f"Invalid train types: {invalid_types}")
        
        is_valid = len(error_messages) == 0
        return is_valid, error_messages
    
    @staticmethod
    def clean_train_data(data):
        """
        Clean and preprocess train operations data.
        
        Args:
            data (pd.DataFrame): Raw train operations data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        df = data.copy()
        
        # Handle missing values
        if 'upstream_delay' in df.columns:
            df['upstream_delay'] = df['upstream_delay'].fillna(0)
        
        if 'passenger_load_percentage' in df.columns:
            df['passenger_load_percentage'] = df['passenger_load_percentage'].fillna(50)
        
        # Ensure boolean columns are proper boolean type
        boolean_columns = ['is_holiday', 'platform_available', 'crew_available', 'is_peak_hour']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Ensure numeric columns are proper numeric type
        numeric_columns = ['upstream_delay', 'passenger_load_percentage', 'scheduled_headway', 'actual_delay']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Cap passenger load at 100%
        if 'passenger_load_percentage' in df.columns:
            df['passenger_load_percentage'] = df['passenger_load_percentage'].clip(0, 100)
        
        # Ensure delays are non-negative
        if 'actual_delay' in df.columns:
            df['actual_delay'] = df['actual_delay'].clip(lower=0)
        if 'upstream_delay' in df.columns:
            df['upstream_delay'] = df['upstream_delay'].clip(lower=0)
        
        return df
    
    @staticmethod
    def generate_data_summary(data):
        """
        Generate a comprehensive summary of train operations data.
        
        Args:
            data (pd.DataFrame): Train operations data
            
        Returns:
            dict: Data summary statistics
        """
        summary = {
            'total_records': len(data),
            'date_range': {
                'start': data.index.min() if isinstance(data.index, pd.DatetimeIndex) else 'N/A',
                'end': data.index.max() if isinstance(data.index, pd.DatetimeIndex) else 'N/A'
            },
            'train_statistics': {},
            'delay_statistics': {},
            'operational_statistics': {}
        }
        
        # Train statistics
        if 'train_id' in data.columns:
            summary['train_statistics']['unique_trains'] = data['train_id'].nunique()
        
        if 'train_type' in data.columns:
            summary['train_statistics']['train_types'] = data['train_type'].value_counts().to_dict()
        
        # Delay statistics
        if 'actual_delay' in data.columns:
            delays = data['actual_delay']
            summary['delay_statistics'] = {
                'average_delay': delays.mean(),
                'median_delay': delays.median(),
                'max_delay': delays.max(),
                'delay_rate': (delays > 0).mean(),
                'on_time_rate': (delays <= 5).mean()
            }
        
        # Operational statistics
        if 'passenger_load_percentage' in data.columns:
            summary['operational_statistics']['average_load'] = data['passenger_load_percentage'].mean()
        
        if 'platform_available' in data.columns:
            summary['operational_statistics']['platform_availability'] = data['platform_available'].mean()
        
        if 'crew_available' in data.columns:
            summary['operational_statistics']['crew_availability'] = data['crew_available'].mean()
        
        if 'recommended_action' in data.columns:
            summary['operational_statistics']['actions'] = data['recommended_action'].value_counts().to_dict()
        
        return summary


class ModelUtils:
    """
    Utility functions for model operations and performance analysis.
    """
    
    @staticmethod
    def save_model_results(results, filepath):
        """
        Save model training results to file.
        
        Args:
            results (dict): Model training results
            filepath (str): Path to save the results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif hasattr(value, 'tolist'):  # pandas Series
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    @staticmethod
    def load_model_results(filepath):
        """
        Load model training results from file.
        
        Args:
            filepath (str): Path to the results file
            
        Returns:
            dict: Model training results
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        # Convert lists back to numpy arrays where appropriate
        array_keys = ['y_train', 'y_test', 'y_pred_train', 'y_pred_test', 'y_pred']
        for key in array_keys:
            if key in results and isinstance(results[key], list):
                results[key] = np.array(results[key])
        
        return results
    
    @staticmethod
    def compare_model_performance(results_dict):
        """
        Compare performance across multiple models.
        
        Args:
            results_dict (dict): Dictionary of model results
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for model_name, results in results_dict.items():
            row = {'Model': model_name}
            
            # Extract relevant metrics
            if 'mae' in results:
                row['MAE'] = results['mae']
            if 'r2_score' in results:
                row['R² Score'] = results['r2_score']
            if 'accuracy' in results:
                row['Accuracy'] = results['accuracy']
            if 'f1_score' in results:
                row['F1 Score'] = results['f1_score']
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)


class VisualizationUtils:
    """
    Utility functions for creating visualizations and plots.
    """
    
    @staticmethod
    def prepare_delay_distribution_data(data):
        """
        Prepare data for delay distribution visualization.
        
        Args:
            data (pd.DataFrame): Train operations data
            
        Returns:
            dict: Data prepared for plotting
        """
        if 'actual_delay' not in data.columns:
            return {'delays': [], 'bins': []}
        
        delays = data['actual_delay'].values
        
        return {
            'delays': delays,
            'mean_delay': np.mean(delays),
            'median_delay': np.median(delays),
            'delay_rate': (delays > 0).mean(),
            'on_time_rate': (delays <= 5).mean()
        }
    
    @staticmethod
    def prepare_performance_comparison_data(original_data, optimized_data):
        """
        Prepare data for performance comparison visualization.
        
        Args:
            original_data (pd.DataFrame): Original performance data
            optimized_data (pd.DataFrame): Optimized performance data
            
        Returns:
            dict: Comparison data
        """
        metrics = ['avg_delay', 'on_time_rate', 'cancellations']
        
        comparison = {
            'metrics': metrics,
            'original': [],
            'optimized': [],
            'improvement': []
        }
        
        for metric in metrics:
            if metric == 'avg_delay':
                orig_val = original_data['actual_delay'].mean()
                opt_val = optimized_data['actual_delay'].mean()
            elif metric == 'on_time_rate':
                orig_val = (original_data['actual_delay'] <= 5).mean()
                opt_val = (optimized_data['actual_delay'] <= 5).mean()
            elif metric == 'cancellations':
                orig_val = (original_data['recommended_action'] == 'Cancel').sum()
                opt_val = (optimized_data['recommended_action'] == 'Cancel').sum()
            
            comparison['original'].append(orig_val)
            comparison['optimized'].append(opt_val)
            
            if metric == 'on_time_rate':
                improvement = (opt_val - orig_val) * 100  # Percentage point improvement
            else:
                improvement = ((orig_val - opt_val) / orig_val * 100) if orig_val > 0 else 0
            
            comparison['improvement'].append(improvement)
        
        return comparison


class ConfigUtils:
    """
    Utility functions for configuration management.
    """
    
    @staticmethod
    def load_config(filepath):
        """
        Load configuration from JSON file.
        
        Args:
            filepath (str): Path to configuration file
            
        Returns:
            dict: Configuration parameters
        """
        if not os.path.exists(filepath):
            return ConfigUtils.get_default_config()
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_config(config, filepath):
        """
        Save configuration to JSON file.
        
        Args:
            config (dict): Configuration parameters
            filepath (str): Path to save configuration
        """
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def get_default_config():
        """
        Get default configuration parameters.
        
        Returns:
            dict: Default configuration
        """
        return {
            'data_generation': {
                'n_trains': 200,
                'n_stations': 50,
                'delay_probability': 0.25,
                'weather_severity_probability': 0.15,
                'holiday_probability': 0.1
            },
            'model_training': {
                'test_size': 0.2,
                'random_state': 42,
                'delay_model_type': 'random_forest',
                'action_model_type': 'random_forest'
            },
            'optimization': {
                'strategy': 'greedy',
                'weights': {
                    'passenger_delay': 0.5,
                    'cancellations': 0.3,
                    'congestion': 0.2
                }
            },
            'evaluation': {
                'cost_params': {
                    'delay_cost_per_minute': 2.5,
                    'cancellation_cost': 150,
                    'implementation_cost': 500000,
                    'annual_operating_cost': 100000,
                    'average_passengers_per_train': 200
                }
            }
        }


class LoggingUtils:
    """
    Utility functions for logging and monitoring.
    """
    
    @staticmethod
    def log_operation(operation, status, details=None):
        """
        Log system operations.
        
        Args:
            operation (str): Operation name
            status (str): Operation status
            details (dict): Additional details
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'operation': operation,
            'status': status,
            'details': details or {}
        }
        
        # In a production system, this would write to a proper logging system
        print(f"[{timestamp}] {operation}: {status}")
        if details:
            print(f"Details: {details}")
    
    @staticmethod
    def create_performance_report(optimization_results, evaluation_results):
        """
        Create a performance report.
        
        Args:
            optimization_results (dict): Optimization results
            evaluation_results (dict): Evaluation results
            
        Returns:
            dict: Performance report
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_trains_analyzed': len(optimization_results['original_data']),
                'optimization_strategy': optimization_results['strategy'],
                'delay_reduction_percent': evaluation_results['optimization_impact']['delay_reduction_percent'],
                'annual_savings': evaluation_results['financial_impact']['annual_savings'],
                'system_efficiency_score': evaluation_results['efficiency_score']
            },
            'detailed_metrics': {
                'model_performance': evaluation_results['model_performance'],
                'optimization_impact': evaluation_results['optimization_impact'],
                'financial_impact': evaluation_results['financial_impact']
            }
        }
        
        return report
