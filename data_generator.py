import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class TrainDataGenerator:
    """
    Generates synthetic train operations data for testing and development.
    This class creates realistic train operation scenarios with various features
    that affect delays and scheduling decisions.
    """
    
    def __init__(self, n_trains=200, n_stations=50, delay_probability=0.25, 
                 weather_severity_probability=0.15, holiday_probability=0.1):
        """
        Initialize the data generator with configuration parameters.
        
        Args:
            n_trains (int): Number of different trains in the system
            n_stations (int): Number of stations in the network
            delay_probability (float): Probability of a train experiencing delays
            weather_severity_probability (float): Probability of severe weather
            holiday_probability (float): Probability of a day being a holiday
        """
        self.n_trains = n_trains
        self.n_stations = n_stations
        self.delay_probability = delay_probability
        self.weather_severity_probability = weather_severity_probability
        self.holiday_probability = holiday_probability
        
        # Define train types and their characteristics
        self.train_types = {
            'Express': {'base_speed': 120, 'capacity': 400, 'delay_factor': 0.8},
            'Local': {'base_speed': 80, 'capacity': 200, 'delay_factor': 1.2}
        }
        
        # Define possible actions for rescheduling
        self.actions = ['NoChange', 'Delay', 'ShortTurn', 'Cancel']
        
        # Weather severity levels
        self.weather_levels = ['Clear', 'Light', 'Moderate', 'Severe']
        
        # Days of the week
        self.days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
    def generate_dataset(self, n_samples):
        """
        Generate a complete synthetic dataset with all required features.
        
        Args:
            n_samples (int): Number of train operation records to generate
            
        Returns:
            pd.DataFrame: Generated dataset with all features and target variables
        """
        np.random.seed(42)  # For reproducible results
        random.seed(42)
        
        data = []
        
        for i in range(n_samples):
            record = self._generate_single_record()
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Generate target variables based on features
        df = self._generate_targets(df)
        
        return df
    
    def _generate_single_record(self):
        """Generate a single train operation record with all features."""
        
        # Basic train information
        train_id = f"T{random.randint(1, self.n_trains):03d}"
        train_type = random.choice(list(self.train_types.keys()))
        
        # Time and date features
        day_of_week = random.choice(self.days_of_week)
        is_holiday = random.random() < self.holiday_probability
        
        # Operational features
        upstream_delay = max(0, np.random.normal(2, 5))  # Mean 2 min, some negative values clipped
        passenger_load_percentage = random.uniform(30, 95)
        
        # Weather conditions
        weather_severity = np.random.choice(
            self.weather_levels,
            p=[0.4, 0.3, 0.2, 0.1]  # More likely to have good weather
        )
        
        # Infrastructure availability
        platform_available = random.random() > 0.05  # 95% platform availability
        crew_available = random.random() > 0.03  # 97% crew availability
        
        # Scheduled headway (gap between trains)
        if train_type == 'Express':
            scheduled_headway = random.uniform(15, 30)  # Express trains less frequent
        else:
            scheduled_headway = random.uniform(5, 15)   # Local trains more frequent
        
        # Station information
        origin_station = f"S{random.randint(1, self.n_stations):02d}"
        destination_station = f"S{random.randint(1, self.n_stations):02d}"
        
        # Ensure origin and destination are different
        while destination_station == origin_station:
            destination_station = f"S{random.randint(1, self.n_stations):02d}"
        
        # Peak hour indicator
        hour = random.randint(0, 23)
        is_peak_hour = hour in [7, 8, 9, 17, 18, 19]  # Morning and evening peaks
        
        return {
            'train_id': train_id,
            'train_type': train_type,
            'day_of_week': day_of_week,
            'is_holiday': is_holiday,
            'upstream_delay': upstream_delay,
            'passenger_load_percentage': passenger_load_percentage,
            'weather_severity': weather_severity,
            'platform_available': platform_available,
            'crew_available': crew_available,
            'scheduled_headway': scheduled_headway,
            'origin_station': origin_station,
            'destination_station': destination_station,
            'hour': hour,
            'is_peak_hour': is_peak_hour
        }
    
    def _generate_targets(self, df):
        """
        Generate target variables (actual delay and recommended action) based on features.
        
        Args:
            df (pd.DataFrame): DataFrame with input features
            
        Returns:
            pd.DataFrame: DataFrame with added target variables
        """
        actual_delays = []
        recommended_actions = []
        
        for _, row in df.iterrows():
            # Calculate base delay probability
            delay_prob = self.delay_probability
            
            # Adjust probability based on features
            if row['train_type'] == 'Express':
                delay_prob *= 0.8  # Express trains more reliable
            
            if row['is_holiday']:
                delay_prob *= 0.7  # Less traffic on holidays
            
            if row['day_of_week'] in ['Saturday', 'Sunday']:
                delay_prob *= 0.6  # Weekend has less congestion
            
            if row['weather_severity'] == 'Severe':
                delay_prob *= 2.0
            elif row['weather_severity'] == 'Moderate':
                delay_prob *= 1.5
            elif row['weather_severity'] == 'Light':
                delay_prob *= 1.2
            
            if not row['platform_available']:
                delay_prob *= 3.0
            
            if not row['crew_available']:
                delay_prob *= 4.0
            
            if row['upstream_delay'] > 5:
                delay_prob *= 1.5
            
            if row['passenger_load_percentage'] > 80:
                delay_prob *= 1.3
            
            if row['is_peak_hour']:
                delay_prob *= 1.4
            
            # Cap probability at 1.0
            delay_prob = min(delay_prob, 1.0)
            
            # Generate actual delay
            if random.random() < delay_prob:
                # Calculate delay magnitude
                base_delay = np.random.exponential(8)  # Exponential distribution for delays
                
                # Adjust delay based on severity factors
                if row['weather_severity'] == 'Severe':
                    base_delay *= 2.0
                elif row['weather_severity'] == 'Moderate':
                    base_delay *= 1.5
                
                if not row['platform_available']:
                    base_delay += random.uniform(10, 30)
                
                if not row['crew_available']:
                    base_delay += random.uniform(15, 45)
                
                if row['upstream_delay'] > 0:
                    base_delay += row['upstream_delay'] * 0.7
                
                actual_delay = max(0, base_delay)
            else:
                actual_delay = 0
            
            actual_delays.append(actual_delay)
            
            # Determine recommended action based on delay and other factors
            action = self._determine_action(actual_delay, row)
            recommended_actions.append(action)
        
        df['actual_delay'] = actual_delays
        df['recommended_action'] = recommended_actions
        
        return df
    
    def _determine_action(self, delay, row):
        """
        Determine the recommended action based on delay and operational factors.
        
        Args:
            delay (float): Actual delay in minutes
            row (pd.Series): Train operation record
            
        Returns:
            str: Recommended action (NoChange, Delay, ShortTurn, Cancel)
        """
        # No change for minimal delays
        if delay < 5:
            return 'NoChange'
        
        # Consider cancellation for extreme situations
        if (delay > 60 or 
            not row['crew_available'] or 
            (not row['platform_available'] and delay > 30)):
            return 'Cancel'
        
        # Short turn for moderate delays during peak hours or with high passenger load
        if (delay > 20 and delay < 45 and 
            (row['is_peak_hour'] or row['passenger_load_percentage'] > 85)):
            return 'ShortTurn'
        
        # Delay for manageable delays
        if delay >= 5:
            return 'Delay'
        
        return 'NoChange'
    
    def get_feature_descriptions(self):
        """
        Get descriptions of all features in the dataset.
        
        Returns:
            dict: Dictionary with feature names as keys and descriptions as values
        """
        return {
            'train_id': 'Unique identifier for each train',
            'train_type': 'Type of train (Express/Local)',
            'day_of_week': 'Day of the week for the operation',
            'is_holiday': 'Boolean flag indicating if the day is a holiday',
            'upstream_delay': 'Delay accumulated from previous stations (minutes)',
            'passenger_load_percentage': 'Percentage of train capacity occupied by passengers',
            'weather_severity': 'Weather conditions (Clear/Light/Moderate/Severe)',
            'platform_available': 'Boolean flag indicating platform availability',
            'crew_available': 'Boolean flag indicating crew availability',
            'scheduled_headway': 'Planned time gap between consecutive trains (minutes)',
            'origin_station': 'Starting station identifier',
            'destination_station': 'Ending station identifier',
            'hour': 'Hour of the day (0-23)',
            'is_peak_hour': 'Boolean flag indicating peak hours (7-9 AM, 5-7 PM)',
            'actual_delay': 'Target variable: Actual delay experienced (minutes)',
            'recommended_action': 'Target variable: Recommended scheduling action'
        }
    
    def generate_real_time_data(self, n_records=50):
        """
        Generate real-time operational data for testing the system.
        
        Args:
            n_records (int): Number of current train operations to generate
            
        Returns:
            pd.DataFrame: Real-time operational data
        """
        data = []
        current_hour = datetime.now().hour
        
        for i in range(n_records):
            record = self._generate_single_record()
            # Override hour to current time +/- few hours
            record['hour'] = (current_hour + random.randint(-2, 2)) % 24
            record['is_peak_hour'] = record['hour'] in [7, 8, 9, 17, 18, 19]
            data.append(record)
        
        df = pd.DataFrame(data)
        return df
