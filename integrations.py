import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

class RailwayIntegrationManager:
    """
    Integration manager for connecting with external railway control systems.
    Provides secure API interfaces for TMS, signalling systems, and rolling stock status.
    """
    
    def __init__(self):
        """Initialize the integration manager with default configurations."""
        self.connections = {}
        self.api_endpoints = {
            'tms': 'https://api.tms.railway.local',
            'signalling': 'https://api.signals.railway.local', 
            'rolling_stock': 'https://api.fleet.railway.local',
            'weather': 'https://api.weather.railway.local',
            'passenger_info': 'https://api.passenger.railway.local'
        }
        
        # Security configurations
        self.auth_tokens = {}
        self.rate_limits = {
            'tms': 100,  # requests per minute
            'signalling': 200,
            'rolling_stock': 50,
            'weather': 30,
            'passenger_info': 150
        }
        
        # Data cache for reducing API calls
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes default TTL
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def register_api_connection(self, system_name: str, endpoint: str, auth_token: str):
        """
        Register a new API connection.
        
        Args:
            system_name (str): Name of the external system
            endpoint (str): API endpoint URL
            auth_token (str): Authentication token
        """
        self.api_endpoints[system_name] = endpoint
        self.auth_tokens[system_name] = auth_token
        self.connections[system_name] = {
            'status': 'registered',
            'last_connected': None,
            'requests_made': 0,
            'errors': 0
        }
        
        self.logger.info(f"Registered API connection for {system_name}")
    
    def get_real_time_train_positions(self) -> pd.DataFrame:
        """
        Fetch real-time train positions from TMS.
        
        Returns:
            pd.DataFrame: Train positions with coordinates and status
        """
        try:
            # Simulate API call to TMS
            # In production, this would make actual API calls
            cached_data = self._get_cached_data('train_positions')
            if cached_data is not None:
                return cached_data
            
            # Simulate real-time train position data
            num_trains = np.random.randint(20, 50)
            positions_data = []
            
            for i in range(num_trains):
                train_data = {
                    'train_id': f'TRN_{i+1:03d}',
                    'latitude': np.random.uniform(51.4, 51.6),  # London area coordinates
                    'longitude': np.random.uniform(-0.2, 0.1),
                    'speed_kmh': np.random.uniform(0, 120),
                    'direction': np.random.choice(['North', 'South', 'East', 'West']),
                    'next_station': f'Station_{np.random.randint(1, 20)}',
                    'estimated_arrival': datetime.now() + timedelta(minutes=np.random.randint(2, 30)),
                    'passenger_count': np.random.randint(50, 400),
                    'status': np.random.choice(['On-Time', 'Delayed', 'Approaching'], p=[0.6, 0.3, 0.1]),
                    'delay_minutes': np.random.exponential(3) if np.random.random() < 0.3 else 0
                }
                positions_data.append(train_data)
            
            positions_df = pd.DataFrame(positions_data)
            
            # Cache the data
            self._cache_data('train_positions', positions_df)
            
            # Update connection stats
            self._update_connection_stats('tms', success=True)
            
            return positions_df
            
        except Exception as e:
            self.logger.error(f"Error fetching train positions: {str(e)}")
            self._update_connection_stats('tms', success=False)
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def get_signal_status(self, section_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch current signal status from signalling system.
        
        Args:
            section_id (str, optional): Specific section to query
            
        Returns:
            dict: Signal status information
        """
        try:
            cached_data = self._get_cached_data(f'signals_{section_id}')
            if cached_data is not None:
                return cached_data
            
            # Simulate signal system data
            signals = {}
            signal_count = 50 if section_id is None else 10
            
            for i in range(signal_count):
                signal_id = f'SIG_{section_id}_{i+1:02d}' if section_id else f'SIG_{i+1:03d}'
                signals[signal_id] = {
                    'status': np.random.choice(['Green', 'Red', 'Yellow'], p=[0.6, 0.2, 0.2]),
                    'next_change_time': datetime.now() + timedelta(seconds=np.random.randint(30, 300)),
                    'section': section_id or f'Section_{i//5 + 1}',
                    'train_approaching': np.random.choice([True, False], p=[0.3, 0.7]),
                    'maintenance_mode': np.random.choice([True, False], p=[0.05, 0.95]),
                    'last_updated': datetime.now()
                }
            
            signal_data = {
                'signals': signals,
                'section_id': section_id,
                'total_signals': len(signals),
                'operational_signals': sum(1 for s in signals.values() if not s['maintenance_mode']),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the data
            self._cache_data(f'signals_{section_id}', signal_data)
            self._update_connection_stats('signalling', success=True)
            
            return signal_data
            
        except Exception as e:
            self.logger.error(f"Error fetching signal status: {str(e)}")
            self._update_connection_stats('signalling', success=False)
            return {}
    
    def get_rolling_stock_status(self) -> pd.DataFrame:
        """
        Fetch rolling stock status and availability.
        
        Returns:
            pd.DataFrame: Rolling stock information
        """
        try:
            cached_data = self._get_cached_data('rolling_stock')
            if cached_data is not None:
                return cached_data
            
            # Simulate rolling stock data
            stock_data = []
            for i in range(100):  # 100 train units
                unit_data = {
                    'unit_id': f'UNIT_{i+1:03d}',
                    'type': np.random.choice(['Electric', 'Diesel', 'Hybrid'], p=[0.6, 0.3, 0.1]),
                    'capacity': np.random.choice([200, 300, 400, 500]),
                    'status': np.random.choice(['In-Service', 'Maintenance', 'Available', 'Out-of-Service'], 
                                            p=[0.7, 0.15, 0.1, 0.05]),
                    'location': f'Depot_{np.random.randint(1, 10)}',
                    'last_maintenance': datetime.now() - timedelta(days=np.random.randint(1, 30)),
                    'next_maintenance': datetime.now() + timedelta(days=np.random.randint(1, 90)),
                    'mileage': np.random.randint(50000, 500000),
                    'fuel_level': np.random.uniform(20, 100) if np.random.random() < 0.4 else None,  # Diesel units only
                    'battery_level': np.random.uniform(80, 100) if np.random.random() < 0.7 else None  # Electric/Hybrid
                }
                stock_data.append(unit_data)
            
            stock_df = pd.DataFrame(stock_data)
            
            # Cache the data
            self._cache_data('rolling_stock', stock_df)
            self._update_connection_stats('rolling_stock', success=True)
            
            return stock_df
            
        except Exception as e:
            self.logger.error(f"Error fetching rolling stock status: {str(e)}")
            self._update_connection_stats('rolling_stock', success=False)
            return pd.DataFrame()
    
    def get_weather_data(self, location: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch weather data affecting railway operations.
        
        Args:
            location (str, optional): Specific location to query
            
        Returns:
            dict: Weather information
        """
        try:
            cached_data = self._get_cached_data(f'weather_{location}')
            if cached_data is not None:
                return cached_data
            
            # Simulate weather data
            weather_conditions = ['Clear', 'Light Rain', 'Heavy Rain', 'Snow', 'Fog', 'High Wind']
            current_condition = np.random.choice(weather_conditions, p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.1])
            
            weather_data = {
                'location': location or 'Network-wide',
                'current_condition': current_condition,
                'temperature_celsius': np.random.uniform(-5, 35),
                'humidity_percent': np.random.uniform(30, 95),
                'wind_speed_kmh': np.random.uniform(0, 80),
                'visibility_km': np.random.uniform(0.1, 50) if current_condition == 'Fog' else np.random.uniform(10, 50),
                'precipitation_mm': np.random.uniform(0, 20) if 'Rain' in current_condition else 0,
                'impact_level': self._calculate_weather_impact(current_condition),
                'forecast_6h': [
                    {
                        'time': datetime.now() + timedelta(hours=i),
                        'condition': np.random.choice(weather_conditions),
                        'temperature': np.random.uniform(-5, 35)
                    } for i in range(1, 7)
                ],
                'alerts': self._generate_weather_alerts(current_condition),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the data
            self._cache_data(f'weather_{location}', weather_data)
            self._update_connection_stats('weather', success=True)
            
            return weather_data
            
        except Exception as e:
            self.logger.error(f"Error fetching weather data: {str(e)}")
            self._update_connection_stats('weather', success=False)
            return {}
    
    def get_passenger_information(self) -> Dict[str, Any]:
        """
        Fetch passenger information and crowding data.
        
        Returns:
            dict: Passenger information
        """
        try:
            cached_data = self._get_cached_data('passenger_info')
            if cached_data is not None:
                return cached_data
            
            # Simulate passenger data
            stations = [f'Station_{i}' for i in range(1, 21)]
            passenger_data = {
                'total_passengers': np.random.randint(5000, 15000),
                'peak_hour_factor': np.random.uniform(1.2, 2.0),
                'station_crowding': {
                    station: {
                        'waiting_passengers': np.random.randint(0, 200),
                        'crowding_level': np.random.choice(['Low', 'Medium', 'High'], p=[0.5, 0.3, 0.2]),
                        'platform_capacity': np.random.randint(300, 800),
                        'accessibility_issues': np.random.choice([True, False], p=[0.1, 0.9])
                    } for station in stations
                },
                'service_disruptions': [
                    {
                        'type': 'Elevator Out of Service',
                        'location': np.random.choice(stations),
                        'impact': 'Medium',
                        'estimated_fix': datetime.now() + timedelta(hours=np.random.randint(1, 8))
                    } for _ in range(np.random.randint(0, 3))
                ],
                'passenger_feedback': {
                    'satisfaction_score': np.random.uniform(3.5, 4.8),
                    'complaints_today': np.random.randint(0, 25),
                    'common_issues': ['Delays', 'Overcrowding', 'Information Display', 'Cleanliness']
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the data
            self._cache_data('passenger_info', passenger_data)
            self._update_connection_stats('passenger_info', success=True)
            
            return passenger_data
            
        except Exception as e:
            self.logger.error(f"Error fetching passenger information: {str(e)}")
            self._update_connection_stats('passenger_info', success=False)
            return {}
    
    def send_control_command(self, system: str, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send control commands to external systems.
        
        Args:
            system (str): Target system ('signals', 'tms', etc.)
            command (dict): Command parameters
            
        Returns:
            dict: Command execution result
        """
        try:
            # Validate command
            if not self._validate_command(system, command):
                return {
                    'success': False,
                    'error': 'Invalid command parameters',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Log command for audit trail
            self.logger.info(f"Sending command to {system}: {command}")
            
            # Simulate command execution
            # In production, this would make actual API calls
            execution_time = np.random.uniform(0.1, 2.0)  # Simulate processing time
            success_probability = 0.95  # 95% success rate
            
            if np.random.random() < success_probability:
                result = {
                    'success': True,
                    'command_id': f"CMD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}",
                    'execution_time': execution_time,
                    'system': system,
                    'command': command,
                    'timestamp': datetime.now().isoformat()
                }
                
                self._update_connection_stats(system, success=True)
            else:
                result = {
                    'success': False,
                    'error': 'System temporarily unavailable',
                    'retry_after': 30,  # seconds
                    'timestamp': datetime.now().isoformat()
                }
                
                self._update_connection_stats(system, success=False)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error sending command to {system}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get health status of all integrated systems.
        
        Returns:
            dict: System health information
        """
        health_status = {
            'overall_status': 'Operational',
            'systems': {},
            'last_updated': datetime.now().isoformat()
        }
        
        operational_systems = 0
        total_systems = len(self.connections)
        
        for system_name, connection_info in self.connections.items():
            # Calculate system health score
            requests_made = connection_info.get('requests_made', 0)
            errors = connection_info.get('errors', 0)
            
            if requests_made > 0:
                error_rate = errors / requests_made
                if error_rate < 0.05:
                    status = 'Operational'
                    operational_systems += 1
                elif error_rate < 0.15:
                    status = 'Degraded'
                else:
                    status = 'Critical'
            else:
                status = 'Unknown'
            
            health_status['systems'][system_name] = {
                'status': status,
                'error_rate': f"{(errors/max(requests_made, 1))*100:.1f}%",
                'requests_made': requests_made,
                'errors': errors,
                'last_connected': connection_info.get('last_connected'),
                'uptime': np.random.uniform(95, 100)  # Simulate uptime percentage
            }
        
        # Update overall status
        if operational_systems == total_systems:
            health_status['overall_status'] = 'All Systems Operational'
        elif operational_systems / total_systems > 0.8:
            health_status['overall_status'] = 'Mostly Operational'
        else:
            health_status['overall_status'] = 'System Issues Detected'
        
        return health_status
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Check if data exists in cache and is still valid."""
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if datetime.now() - cache_entry['timestamp'] < timedelta(seconds=self.cache_ttl):
                return cache_entry['data']
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        return None
    
    def _cache_data(self, cache_key: str, data: Any):
        """Store data in cache with timestamp."""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def _update_connection_stats(self, system: str, success: bool):
        """Update connection statistics for a system."""
        if system in self.connections:
            self.connections[system]['requests_made'] += 1
            if not success:
                self.connections[system]['errors'] += 1
            self.connections[system]['last_connected'] = datetime.now()
    
    def _validate_command(self, system: str, command: Dict[str, Any]) -> bool:
        """Validate command parameters before sending."""
        required_fields = ['action', 'target']
        
        # Check required fields
        for field in required_fields:
            if field not in command:
                return False
        
        # System-specific validation
        if system == 'signals':
            valid_actions = ['set_signal', 'clear_signal', 'maintenance_mode']
            return command['action'] in valid_actions
        elif system == 'tms':
            valid_actions = ['route_train', 'hold_train', 'cancel_train']
            return command['action'] in valid_actions
        
        return True
    
    def _calculate_weather_impact(self, condition: str) -> str:
        """Calculate weather impact level on railway operations."""
        impact_map = {
            'Clear': 'None',
            'Light Rain': 'Low',
            'Heavy Rain': 'Medium',
            'Snow': 'High',
            'Fog': 'Medium',
            'High Wind': 'Medium'
        }
        return impact_map.get(condition, 'Unknown')
    
    def _generate_weather_alerts(self, condition: str) -> List[Dict[str, str]]:
        """Generate weather-related alerts."""
        alerts = []
        
        if condition == 'Heavy Rain':
            alerts.append({
                'type': 'Weather Warning',
                'message': 'Heavy rain may cause delays due to reduced visibility and flooding risk',
                'severity': 'Medium'
            })
        elif condition == 'Snow':
            alerts.append({
                'type': 'Weather Alert',
                'message': 'Snow conditions may significantly impact train operations',
                'severity': 'High'
            })
        elif condition == 'Fog':
            alerts.append({
                'type': 'Visibility Warning',
                'message': 'Poor visibility may require reduced speeds',
                'severity': 'Medium'
            })
        elif condition == 'High Wind':
            alerts.append({
                'type': 'Wind Alert',
                'message': 'High winds may affect service stability',
                'severity': 'Medium'
            })
        
        return alerts

# Singleton instance for global access
integration_manager = RailwayIntegrationManager()