import pandas as pd
import numpy as np
from copy import deepcopy
import random
from datetime import datetime, timedelta

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
            strategy (str): Optimization strategy ('greedy', 'weighted_greedy', 'constraint_based')
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
        
        # Constraint parameters for section control
        self.constraints = {
            'max_trains_per_section': 5,
            'min_headway_seconds': 120,
            'max_platform_occupancy': 2,
            'signal_capacity': 3,
            'safety_margin_seconds': 30
        }
        
        # Track network state for real-time optimization
        self.network_state = {
            'active_trains': {},
            'section_occupancy': {},
            'signal_states': {},
            'platform_status': {},
            'conflicts': []
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
        elif self.strategy == 'constraint_based':
            optimized_data = self._constraint_based_optimization(optimized_data)
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
    
    def generate_section_controller_recommendations(self, active_trains):
        """
        Generate recommendations for section controllers managing active trains.
        
        Args:
            active_trains (pd.DataFrame): Currently active trains in the section
            
        Returns:
            list: List of controller recommendations
        """
        recommendations = []
        
        for _, train in active_trains.iterrows():
            # Convert train data to dictionary
            train_dict = train.to_dict()
            
            # Get real-time recommendation
            recommendation = self.real_time_decision_support(train_dict, self.network_state)
            
            # Format for controller interface
            controller_rec = {
                'train_id': train['train_id'],
                'action': recommendation['recommended_action'],
                'priority': recommendation['priority'],
                'confidence': recommendation['confidence'],
                'explanation': self._generate_explanation(recommendation),
                'alternatives': recommendation['alternatives'][:2],  # Top 2 alternatives
                'urgency': 'high' if recommendation['conflicts'] else 'normal'
            }
            
            recommendations.append(controller_rec)
        
        # Sort by priority and urgency
        recommendations.sort(key=lambda x: (
            x['urgency'] == 'high',
            x['priority'] == 'high',
            x['confidence']
        ), reverse=True)
        
        return recommendations
    
    def _generate_explanation(self, recommendation):
        """
        Generate human-readable explanation for the recommendation.
        
        Args:
            recommendation (dict): AI recommendation
            
        Returns:
            str: Explanation text
        """
        action = recommendation['recommended_action']
        delay = recommendation['predicted_delay']
        conflicts = len(recommendation['conflicts'])
        
        if action == 'NoChange':
            return f"Train operating normally with minimal delay ({delay:.0f} min). No intervention required."
        elif action == 'Delay':
            base_msg = f"Predicted {delay:.0f} minute delay detected."
            if conflicts > 0:
                return f"{base_msg} Strategic holding recommended to avoid conflicts."
            else:
                return f"{base_msg} Minor schedule adjustment recommended."
        elif action == 'ShortTurn':
            return f"Significant delay ({delay:.0f} min) with high passenger impact. Short-turn to minimize disruption."
        elif action == 'Cancel':
            return f"Severe operational issues detected. Cancellation may be necessary to prevent cascade delays."
        else:
            return "Custom action required based on specific operational conditions."
    
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
    
    def _constraint_based_optimization(self, data):
        """
        Apply constraint-based optimization for section control.
        
        Args:
            data (pd.DataFrame): Train operations data to optimize
            
        Returns:
            pd.DataFrame: Optimized train operations data
        """
        optimized_data = data.copy()
        
        # Detect conflicts first
        conflicts = self._detect_conflicts(optimized_data)
        
        # Resolve conflicts using constraint programming
        for conflict in conflicts:
            resolution = self._resolve_conflict(conflict, optimized_data)
            optimized_data = self._apply_conflict_resolution(optimized_data, resolution)
        
        # Apply additional optimizations while respecting constraints
        optimized_data = self._apply_constraint_optimization(optimized_data)
        
        return optimized_data
    
    def _detect_conflicts(self, data):
        """
        Detect potential conflicts in train movements.
        
        Args:
            data (pd.DataFrame): Train operations data
            
        Returns:
            list: List of detected conflicts
        """
        conflicts = []
        
        # Group trains by section/route
        for section in data['origin_station'].unique():
            section_trains = data[data['origin_station'] == section]
            
            # Check for headway violations
            for i, train1 in section_trains.iterrows():
                for j, train2 in section_trains.iterrows():
                    if i != j:
                        # Simulate time-based conflict detection
                        time_diff = abs(hash(train1['train_id']) % 300 - hash(train2['train_id']) % 300)
                        
                        if time_diff < self.constraints['min_headway_seconds']:
                            conflicts.append({
                                'type': 'headway_violation',
                                'trains': [train1['train_id'], train2['train_id']],
                                'section': section,
                                'severity': 'high' if time_diff < 60 else 'medium',
                                'time_diff': time_diff
                            })
        
        # Check platform capacity conflicts
        platform_usage = {}
        for _, train in data.iterrows():
            platform = f"Platform_{hash(train['destination_station']) % 10}"
            if platform not in platform_usage:
                platform_usage[platform] = 0
            platform_usage[platform] += 1
            
            if platform_usage[platform] > self.constraints['max_platform_occupancy']:
                conflicts.append({
                    'type': 'platform_capacity',
                    'trains': [train['train_id']],
                    'platform': platform,
                    'severity': 'high',
                    'occupancy': platform_usage[platform]
                })
        
        return conflicts
    
    def _resolve_conflict(self, conflict, data):
        """
        Generate resolution strategy for a specific conflict.
        
        Args:
            conflict (dict): Conflict information
            data (pd.DataFrame): Current train data
            
        Returns:
            dict: Resolution strategy
        """
        if conflict['type'] == 'headway_violation':
            # Resolution strategies for headway violations
            if conflict['severity'] == 'high':
                return {
                    'action': 'hold_train',
                    'target': conflict['trains'][1],  # Hold second train
                    'duration': self.constraints['min_headway_seconds'] - conflict['time_diff'],
                    'reason': 'Safety headway violation'
                }
            else:
                return {
                    'action': 'adjust_speed',
                    'target': conflict['trains'][1],
                    'adjustment': 0.9,  # Reduce speed by 10%
                    'reason': 'Minor headway optimization'
                }
        
        elif conflict['type'] == 'platform_capacity':
            return {
                'action': 'reroute',
                'target': conflict['trains'][0],
                'alternative_platform': f"Platform_{(hash(conflict['trains'][0]) + 1) % 10}",
                'reason': 'Platform capacity exceeded'
            }
        
        return {'action': 'no_action', 'reason': 'No resolution needed'}
    
    def _apply_conflict_resolution(self, data, resolution):
        """
        Apply conflict resolution to the data.
        
        Args:
            data (pd.DataFrame): Train operations data
            resolution (dict): Resolution strategy
            
        Returns:
            pd.DataFrame: Updated data with resolution applied
        """
        updated_data = data.copy()
        
        if resolution['action'] == 'hold_train':
            # Increase delay for the target train
            mask = updated_data['train_id'] == resolution['target']
            updated_data.loc[mask, 'actual_delay'] += resolution['duration'] / 60  # Convert to minutes
            updated_data.loc[mask, 'recommended_action'] = 'Delay'
        
        elif resolution['action'] == 'reroute':
            # Mark train for rerouting
            mask = updated_data['train_id'] == resolution['target']
            updated_data.loc[mask, 'recommended_action'] = 'ShortTurn'
            updated_data.loc[mask, 'actual_delay'] *= 0.8  # Reduce delay with rerouting
        
        elif resolution['action'] == 'adjust_speed':
            # Minor delay adjustment
            mask = updated_data['train_id'] == resolution['target']
            updated_data.loc[mask, 'actual_delay'] *= resolution['adjustment']
        
        return updated_data
    
    def _apply_constraint_optimization(self, data):
        """
        Apply additional optimizations while respecting operational constraints.
        
        Args:
            data (pd.DataFrame): Train operations data
            
        Returns:
            pd.DataFrame: Optimized data
        """
        optimized_data = data.copy()
        
        # Sort by priority and delay impact
        priority_order = optimized_data.copy()
        priority_order['priority_score'] = (
            priority_order['actual_delay'] * 2 +
            priority_order['passenger_load_percentage'] / 100 * 10 +
            (priority_order['train_type'] == 'Express').astype(int) * 5
        )
        
        priority_order = priority_order.sort_values('priority_score', ascending=False)
        
        # Apply optimizations in priority order
        for idx, train in priority_order.iterrows():
            if train['actual_delay'] > 10:  # Only optimize significantly delayed trains
                # Check if optimization is feasible within constraints
                if self._is_optimization_feasible(train, optimized_data):
                    # Apply optimization
                    current_delay = optimized_data.loc[idx, 'actual_delay']
                    optimization_factor = self._calculate_optimization_factor(train)
                    
                    new_delay = max(0, current_delay * optimization_factor)
                    optimized_data.loc[idx, 'actual_delay'] = new_delay
                    
                    # Update recommended action
                    if new_delay < 5:
                        optimized_data.loc[idx, 'recommended_action'] = 'NoChange'
                    elif new_delay < 15:
                        optimized_data.loc[idx, 'recommended_action'] = 'Delay'
                    else:
                        optimized_data.loc[idx, 'recommended_action'] = 'ShortTurn'
        
        return optimized_data
    
    def _is_optimization_feasible(self, train, data):
        """
        Check if optimization is feasible within operational constraints.
        
        Args:
            train (pd.Series): Train record
            data (pd.DataFrame): Current data state
            
        Returns:
            bool: True if optimization is feasible
        """
        # Check section capacity
        same_section_trains = data[
            data['origin_station'] == train['origin_station']
        ]
        
        if len(same_section_trains) >= self.constraints['max_trains_per_section']:
            return False
        
        # Check platform availability
        if not train['platform_available']:
            return False
        
        # Check crew availability
        if not train['crew_available']:
            return False
        
        return True
    
    def _calculate_optimization_factor(self, train):
        """
        Calculate optimization factor based on train characteristics.
        
        Args:
            train (pd.Series): Train record
            
        Returns:
            float: Optimization factor (0-1, lower = more optimization)
        """
        factor = 1.0
        
        # Express trains get better optimization
        if train['train_type'] == 'Express':
            factor *= 0.7
        
        # Lower passenger load allows more aggressive optimization
        if train['passenger_load_percentage'] < 50:
            factor *= 0.8
        
        # Good weather allows better optimization
        if train['weather_severity'] == 'Clear':
            factor *= 0.85
        
        # Available resources enable optimization
        if train['platform_available'] and train['crew_available']:
            factor *= 0.9
        
        return max(0.5, factor)  # Minimum 50% of original delay
    
    def real_time_decision_support(self, current_train, network_state):
        """
        Provide real-time decision support for section controllers.
        
        Args:
            current_train (dict): Current train information
            network_state (dict): Current network state
            
        Returns:
            dict: Decision support recommendation
        """
        # Create DataFrame for prediction
        train_df = pd.DataFrame([current_train])
        
        # Predict potential delay
        predicted_delay = self.delay_predictor.predict(train_df)[0]
        
        # Get AI action recommendation
        train_df['actual_delay'] = predicted_delay
        recommended_action = self.action_classifier.predict(train_df)[0]
        
        # Analyze network conflicts
        conflicts = self._analyze_real_time_conflicts(current_train, network_state)
        
        # Generate recommendation
        recommendation = {
            'train_id': current_train['train_id'],
            'predicted_delay': predicted_delay,
            'recommended_action': recommended_action,
            'conflicts': conflicts,
            'priority': self._calculate_train_priority(current_train),
            'alternatives': self._generate_alternatives(current_train, network_state),
            'confidence': min(0.95, max(0.6, 1.0 - (predicted_delay / 60)))  # Higher confidence for lower delays
        }
        
        return recommendation
    
    def _analyze_real_time_conflicts(self, train, network_state):
        """
        Analyze real-time conflicts for a specific train.
        
        Args:
            train (dict): Train information
            network_state (dict): Current network state
            
        Returns:
            list: List of potential conflicts
        """
        conflicts = []
        
        # Simulate conflict detection based on train characteristics
        conflict_probability = 0.1  # Base probability
        
        if train.get('passenger_load_percentage', 0) > 80:
            conflict_probability += 0.15
        
        if train.get('weather_severity') == 'Severe':
            conflict_probability += 0.25
        
        if not train.get('platform_available', True):
            conflict_probability += 0.4
        
        if random.random() < conflict_probability:
            conflicts.append({
                'type': 'potential_delay',
                'severity': 'medium' if conflict_probability < 0.3 else 'high',
                'description': 'Potential conflict detected based on current conditions'
            })
        
        return conflicts
    
    def _calculate_train_priority(self, train):
        """
        Calculate priority level for a train.
        
        Args:
            train (dict): Train information
            
        Returns:
            str: Priority level ('low', 'medium', 'high')
        """
        priority_score = 0
        
        # High passenger load increases priority
        if train.get('passenger_load_percentage', 0) > 85:
            priority_score += 3
        elif train.get('passenger_load_percentage', 0) > 70:
            priority_score += 2
        
        # Express trains get higher priority
        if train.get('train_type') == 'Express':
            priority_score += 2
        
        # Peak hours increase priority
        if train.get('is_peak_hour', False):
            priority_score += 1
        
        # Current delay affects priority
        current_delay = train.get('upstream_delay', 0)
        if current_delay > 15:
            priority_score += 3
        elif current_delay > 5:
            priority_score += 1
        
        if priority_score >= 5:
            return 'high'
        elif priority_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_alternatives(self, train, network_state):
        """
        Generate alternative routing and scheduling options.
        
        Args:
            train (dict): Train information
            network_state (dict): Current network state
            
        Returns:
            list: List of alternative options
        """
        alternatives = []
        
        # Alternative 1: Hold at current location
        alternatives.append({
            'option': 'hold',
            'description': 'Hold train at current signal for optimal slot',
            'estimated_delay': '+3-8 minutes',
            'passenger_impact': 'Low',
            'feasibility': 'High'
        })
        
        # Alternative 2: Alternative routing
        if train.get('train_type') == 'Local':  # More routing flexibility for local trains
            alternatives.append({
                'option': 'reroute',
                'description': 'Use alternative route via secondary track',
                'estimated_delay': '-2 to +5 minutes',
                'passenger_impact': 'Medium',
                'feasibility': 'Medium'
            })
        
        # Alternative 3: Platform change
        if train.get('platform_available', True):
            alternatives.append({
                'option': 'platform_change',
                'description': 'Assign to alternative platform',
                'estimated_delay': '+1-3 minutes',
                'passenger_impact': 'Low',
                'feasibility': 'High'
            })
        
        # Alternative 4: Express priority (if conditions allow)
        if train.get('passenger_load_percentage', 0) > 80:
            alternatives.append({
                'option': 'priority_override',
                'description': 'Grant priority passage due to high passenger load',
                'estimated_delay': '-5 to -10 minutes',
                'passenger_impact': 'Very Low',
                'feasibility': 'Medium'
            })
        
        return alternatives
