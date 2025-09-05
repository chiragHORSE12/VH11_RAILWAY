import pandas as pd
import numpy as np
from copy import deepcopy
import random

class TrainScheduleOptimizer:
    """
    Optimization engine for train schedule rescheduling to minimize delays and improve efficiency.
    Implements multiple optimization strategies including greedy and weighted approaches.
    """
    
    def __init__(self, delay_predictor, action_classifier, strategy='greedy'):
        """
        Initialize the schedule optimizer.
        
        Args:
            delay_predictor: Trained delay prediction model
            action_classifier: Trained action classification model
            strategy (str): Optimization strategy ('greedy', 'weighted_greedy')
        """
        self.delay_predictor = delay_predictor
        self.action_classifier = action_classifier
        self.strategy = strategy
        
        # Default weights for optimization objectives
        self.weights = {
            'passenger_delay': 0.5,
            'cancellations': 0.3,
            'congestion': 0.2
        }
        
        # Action costs and benefits
        self.action_costs = {
            'NoChange': 0,
            'Delay': 1,
            'ShortTurn': 3,
            'Cancel': 10
        }
        
    def set_weights(self, passenger_delay=0.5, cancellations=0.3, congestion=0.2):
        """
        Set weights for optimization objectives.
        
        Args:
            passenger_delay (float): Weight for minimizing passenger delays
            cancellations (float): Weight for minimizing cancellations
            congestion (float): Weight for minimizing network congestion
        """
        total = passenger_delay + cancellations + congestion
        self.weights = {
            'passenger_delay': passenger_delay / total,
            'cancellations': cancellations / total,
            'congestion': congestion / total
        }
    
    def optimize_schedule(self, train_data):
        """
        Optimize the train schedule using the specified strategy.
        
        Args:
            train_data (pd.DataFrame): Current train operations data
            
        Returns:
            dict: Optimization results with original and optimized metrics
        """
        # Create copies for optimization
        original_data = train_data.copy()
        optimized_data = train_data.copy()
        
        # Calculate original metrics
        original_metrics = self._calculate_metrics(original_data)
        
        # Apply optimization strategy
        if self.strategy == 'greedy':
            optimized_data = self._greedy_optimization(optimized_data)
        elif self.strategy == 'weighted_greedy':
            optimized_data = self._weighted_greedy_optimization(optimized_data)
        else:
            raise ValueError(f"Unknown optimization strategy: {self.strategy}")
        
        # Calculate optimized metrics
        optimized_metrics = self._calculate_metrics(optimized_data)
        
        # Prepare results
        results = {
            'strategy': self.strategy,
            'original_data': original_data,
            'optimized_data': optimized_data,
            'original_metrics': original_metrics,
            'optimized_metrics': optimized_metrics,
            'original_delays': original_data['actual_delay'].tolist(),
            'optimized_delays': optimized_data['actual_delay'].tolist(),
            'original_actions': original_data['recommended_action'].tolist(),
            'optimized_actions': optimized_data['recommended_action'].tolist(),
            'improvements': self._calculate_improvements(original_metrics, optimized_metrics)
        }
        
        return results
    
    def _greedy_optimization(self, data):
        """
        Apply greedy optimization strategy.
        
        Args:
            data (pd.DataFrame): Train operations data to optimize
            
        Returns:
            pd.DataFrame: Optimized train operations data
        """
        optimized_data = data.copy()
        
        # Sort by delay impact (highest delays first)
        optimized_data = optimized_data.sort_values('actual_delay', ascending=False)
        
        for idx, row in optimized_data.iterrows():
            # Skip if already has minimal delay
            if row['actual_delay'] < 5:
                continue
            
            # Get current predictions
            current_data = pd.DataFrame([row])
            predicted_delay = self.delay_predictor.predict(current_data)[0]
            
            # Try different actions and evaluate impact
            best_action = row['recommended_action']
            best_score = float('inf')
            
            possible_actions = ['NoChange', 'Delay', 'ShortTurn', 'Cancel']
            
            for action in possible_actions:
                # Simulate the action
                simulated_row = row.copy()
                simulated_delay = self._simulate_action_impact(row, action)
                simulated_row['actual_delay'] = simulated_delay
                simulated_row['recommended_action'] = action
                
                # Calculate optimization score
                score = self._calculate_optimization_score(simulated_row, optimized_data)
                
                if score < best_score:
                    best_score = score
                    best_action = action
            
            # Apply best action
            optimized_delay = self._simulate_action_impact(row, best_action)
            optimized_data.at[idx, 'actual_delay'] = optimized_delay
            optimized_data.at[idx, 'recommended_action'] = best_action
        
        return optimized_data
    
    def _weighted_greedy_optimization(self, data):
        """
        Apply weighted greedy optimization strategy.
        
        Args:
            data (pd.DataFrame): Train operations data to optimize
            
        Returns:
            pd.DataFrame: Optimized train operations data
        """
        optimized_data = data.copy()
        
        # Calculate priority scores for each train
        priorities = []
        for idx, row in optimized_data.iterrows():
            priority = self._calculate_priority_score(row)
            priorities.append((idx, priority))
        
        # Sort by priority (highest first)
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        for idx, _ in priorities:
            row = optimized_data.loc[idx]
            
            # Skip if already optimized
            if row['actual_delay'] < 3:
                continue
            
            # Find optimal action considering network effects
            best_action = self._find_optimal_action(row, optimized_data)
            
            # Apply the action
            optimized_delay = self._simulate_action_impact(row, best_action)
            optimized_data.at[idx, 'actual_delay'] = optimized_delay
            optimized_data.at[idx, 'recommended_action'] = best_action
        
        return optimized_data
    
    def _simulate_action_impact(self, train_record, action):
        """
        Simulate the impact of an action on train delay.
        
        Args:
            train_record (pd.Series): Train operation record
            action (str): Action to simulate
            
        Returns:
            float: Simulated delay after action
        """
        current_delay = train_record['actual_delay']
        
        if action == 'NoChange':
            return current_delay
        elif action == 'Delay':
            # Delaying might increase delay but reduce congestion
            if current_delay < 15:
                return current_delay + random.uniform(2, 5)
            else:
                return current_delay * random.uniform(1.1, 1.3)
        elif action == 'ShortTurn':
            # Short turn reduces delay but may impact passengers
            return max(0, current_delay * random.uniform(0.3, 0.6))
        elif action == 'Cancel':
            # Cancellation eliminates delay but has high passenger impact
            return 0
        
        return current_delay
    
    def _calculate_priority_score(self, train_record):
        """
        Calculate priority score for optimization order.
        
        Args:
            train_record (pd.Series): Train operation record
            
        Returns:
            float: Priority score (higher = more important to optimize)
        """
        score = 0
        
        # Delay impact
        score += train_record['actual_delay'] * 2
        
        # Passenger load impact
        score += train_record['passenger_load_percentage'] / 100 * 10
        
        # Peak hour impact
        if train_record['is_peak_hour']:
            score += 15
        
        # Express train priority
        if train_record['train_type'] == 'Express':
            score += 10
        
        # Infrastructure issues
        if not train_record['platform_available']:
            score += 20
        if not train_record['crew_available']:
            score += 25
        
        return score
    
    def _find_optimal_action(self, train_record, network_data):
        """
        Find the optimal action considering network effects.
        
        Args:
            train_record (pd.Series): Train operation record
            network_data (pd.DataFrame): Current network state
            
        Returns:
            str: Optimal action
        """
        possible_actions = ['NoChange', 'Delay', 'ShortTurn', 'Cancel']
        best_action = 'NoChange'
        best_score = float('inf')
        
        for action in possible_actions:
            # Calculate network impact score
            score = self._calculate_network_impact(train_record, action, network_data)
            
            if score < best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _calculate_network_impact(self, train_record, action, network_data):
        """
        Calculate the network-wide impact of an action.
        
        Args:
            train_record (pd.Series): Train operation record
            action (str): Action to evaluate
            network_data (pd.DataFrame): Current network state
            
        Returns:
            float: Network impact score (lower = better)
        """
        impact_score = 0
        
        # Direct delay impact
        simulated_delay = self._simulate_action_impact(train_record, action)
        passenger_impact = simulated_delay * train_record['passenger_load_percentage'] / 100
        impact_score += passenger_impact * self.weights['passenger_delay']
        
        # Cancellation penalty
        if action == 'Cancel':
            cancellation_penalty = train_record['passenger_load_percentage'] * 2
            impact_score += cancellation_penalty * self.weights['cancellations']
        
        # Congestion impact
        if action == 'Delay':
            # Check if delay increases congestion on the route
            same_route_trains = network_data[
                (network_data['origin_station'] == train_record['origin_station']) |
                (network_data['destination_station'] == train_record['destination_station'])
            ]
            congestion_factor = len(same_route_trains) / 10  # Normalize
            impact_score += congestion_factor * self.weights['congestion']
        
        # Action cost
        impact_score += self.action_costs.get(action, 0)
        
        return impact_score
    
    def _calculate_optimization_score(self, train_record, network_data):
        """
        Calculate optimization score for a single train action.
        
        Args:
            train_record (pd.Series): Train operation record
            network_data (pd.DataFrame): Network state
            
        Returns:
            float: Optimization score
        """
        score = 0
        
        # Delay penalty
        delay_penalty = train_record['actual_delay'] * train_record['passenger_load_percentage'] / 100
        score += delay_penalty * self.weights['passenger_delay']
        
        # Cancellation penalty
        if train_record['recommended_action'] == 'Cancel':
            score += 100 * self.weights['cancellations']
        
        # Action cost
        score += self.action_costs.get(train_record['recommended_action'], 0)
        
        return score
    
    def _calculate_metrics(self, data):
        """
        Calculate performance metrics for a schedule.
        
        Args:
            data (pd.DataFrame): Train operations data
            
        Returns:
            dict: Performance metrics
        """
        total_trains = len(data)
        total_delay = data['actual_delay'].sum()
        avg_delay = data['actual_delay'].mean()
        on_time_trains = (data['actual_delay'] <= 5).sum()
        on_time_rate = on_time_trains / total_trains if total_trains > 0 else 0
        cancellations = (data['recommended_action'] == 'Cancel').sum()
        
        # Calculate congestion score (simplified)
        congestion_score = self._calculate_congestion_score(data)
        
        return {
            'total_trains': total_trains,
            'total_delay': total_delay,
            'avg_delay': avg_delay,
            'on_time_rate': on_time_rate,
            'cancellations': cancellations,
            'congestion_score': congestion_score
        }
    
    def _calculate_congestion_score(self, data):
        """
        Calculate network congestion score.
        
        Args:
            data (pd.DataFrame): Train operations data
            
        Returns:
            float: Congestion score (higher = more congested)
        """
        # Group by route and calculate congestion
        route_congestion = {}
        
        for _, row in data.iterrows():
            route = f"{row['origin_station']}-{row['destination_station']}"
            if route not in route_congestion:
                route_congestion[route] = []
            route_congestion[route].append(row['actual_delay'])
        
        # Calculate average congestion across routes
        total_congestion = 0
        route_count = 0
        
        for route, delays in route_congestion.items():
            avg_route_delay = np.mean(delays)
            train_density = len(delays)
            route_congestion_score = avg_route_delay * (train_density / 10)  # Normalize
            total_congestion += route_congestion_score
            route_count += 1
        
        return total_congestion / route_count if route_count > 0 else 0
    
    def _calculate_improvements(self, original_metrics, optimized_metrics):
        """
        Calculate improvement percentages between original and optimized metrics.
        
        Args:
            original_metrics (dict): Original performance metrics
            optimized_metrics (dict): Optimized performance metrics
            
        Returns:
            dict: Improvement percentages
        """
        improvements = {}
        
        # Delay improvement
        if original_metrics['avg_delay'] > 0:
            delay_improvement = (
                (original_metrics['avg_delay'] - optimized_metrics['avg_delay']) /
                original_metrics['avg_delay'] * 100
            )
            improvements['avg_delay'] = delay_improvement
        else:
            improvements['avg_delay'] = 0
        
        # On-time rate improvement
        on_time_improvement = (
            (optimized_metrics['on_time_rate'] - original_metrics['on_time_rate']) * 100
        )
        improvements['on_time_rate'] = on_time_improvement
        
        # Cancellation change
        cancellation_change = (
            optimized_metrics['cancellations'] - original_metrics['cancellations']
        )
        improvements['cancellations'] = cancellation_change
        
        # Congestion improvement
        if original_metrics['congestion_score'] > 0:
            congestion_improvement = (
                (original_metrics['congestion_score'] - optimized_metrics['congestion_score']) /
                original_metrics['congestion_score'] * 100
            )
            improvements['congestion'] = congestion_improvement
        else:
            improvements['congestion'] = 0
        
        return improvements
    
    def simulate_real_time_optimization(self, current_data, time_horizon_minutes=60):
        """
        Simulate real-time optimization for a given time horizon.
        
        Args:
            current_data (pd.DataFrame): Current train operations
            time_horizon_minutes (int): Optimization time horizon in minutes
            
        Returns:
            dict: Real-time optimization results
        """
        # Filter trains within the time horizon
        relevant_trains = current_data[
            current_data['actual_delay'] <= time_horizon_minutes
        ].copy()
        
        if len(relevant_trains) == 0:
            return {
                'message': 'No trains requiring optimization in the time horizon',
                'optimized_count': 0
            }
        
        # Apply optimization
        optimization_results = self.optimize_schedule(relevant_trains)
        
        # Calculate real-time metrics
        real_time_metrics = {
            'optimized_trains': len(relevant_trains),
            'total_delay_saved': (
                optimization_results['original_metrics']['total_delay'] -
                optimization_results['optimized_metrics']['total_delay']
            ),
            'actions_recommended': dict(
                pd.Series(optimization_results['optimized_actions']).value_counts()
            ),
            'estimated_passenger_impact': self._estimate_passenger_impact(
                optimization_results
            )
        }
        
        return {
            'optimization_results': optimization_results,
            'real_time_metrics': real_time_metrics,
            'time_horizon': time_horizon_minutes
        }
    
    def _estimate_passenger_impact(self, optimization_results):
        """
        Estimate the impact on passengers from optimization decisions.
        
        Args:
            optimization_results (dict): Results from optimization
            
        Returns:
            dict: Passenger impact metrics
        """
        original_data = optimization_results['original_data']
        optimized_data = optimization_results['optimized_data']
        
        # Calculate passenger-minutes saved
        original_passenger_minutes = (
            original_data['actual_delay'] * original_data['passenger_load_percentage'] / 100
        ).sum()
        
        optimized_passenger_minutes = (
            optimized_data['actual_delay'] * optimized_data['passenger_load_percentage'] / 100
        ).sum()
        
        passenger_minutes_saved = original_passenger_minutes - optimized_passenger_minutes
        
        # Calculate affected passengers
        affected_passengers = optimized_data[
            optimized_data['recommended_action'] != 'NoChange'
        ]['passenger_load_percentage'].sum()
        
        return {
            'passenger_minutes_saved': passenger_minutes_saved,
            'affected_passengers': affected_passengers,
            'average_time_saved_per_passenger': (
                passenger_minutes_saved / max(affected_passengers, 1)
            )
        }
