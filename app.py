import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import time

from data_generator import TrainDataGenerator
from models import DelayPredictor, ActionClassifier
from optimizer import TrainScheduleOptimizer
from evaluator import SystemEvaluator
from integrations import integration_manager

def main():
    st.set_page_config(
        page_title="AI Train Rescheduling System",
        page_icon="🚂",
        layout="wide"
    )
    
    st.title("🚂 AI-Based Train Rescheduling System")
    st.markdown("**Machine Learning for Delay Prediction and Schedule Optimization**")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Control Center", "Data Generation", "Model Training", "Schedule Optimization", "What-If Simulation", "Evaluation & Results", "Performance Dashboard"]
    )
    
    # Initialize session state
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'optimization_done' not in st.session_state:
        st.session_state.optimization_done = False
    if 'realtime_data' not in st.session_state:
        st.session_state.realtime_data = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    
    if page == "Control Center":
        control_center_page()
    elif page == "Data Generation":
        data_generation_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Schedule Optimization":
        optimization_page()
    elif page == "What-If Simulation":
        whatif_simulation_page()
    elif page == "Evaluation & Results":
        evaluation_page()
    elif page == "Performance Dashboard":
        performance_dashboard_page()

def data_generation_page():
    st.header("📊 Data Generation")
    st.markdown("Generate synthetic train operations dataset for model training and testing.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Dataset Parameters")
        n_samples = st.slider("Number of train operations", 1000, 10000, 5000, 500)
        n_trains = st.slider("Number of trains", 50, 500, 200, 25)
        n_stations = st.slider("Number of stations", 10, 100, 50, 5)
        
        # Advanced parameters in expander
        with st.expander("Advanced Parameters"):
            delay_prob = st.slider("Delay probability", 0.1, 0.5, 0.25, 0.05)
            weather_severity_prob = st.slider("Severe weather probability", 0.05, 0.3, 0.15, 0.05)
            holiday_prob = st.slider("Holiday probability", 0.05, 0.2, 0.1, 0.01)
    
    with col2:
        st.subheader("Generate Dataset")
        
        if st.button("🎲 Generate Synthetic Data", type="primary"):
            with st.spinner("Generating synthetic train operations data..."):
                # Initialize data generator
                generator = TrainDataGenerator(
                    n_trains=n_trains,
                    n_stations=n_stations,
                    delay_probability=delay_prob,
                    weather_severity_probability=weather_severity_prob,
                    holiday_probability=holiday_prob
                )
                
                # Generate data
                data = generator.generate_dataset(n_samples)
                
                # Save to session state
                st.session_state.train_data = data
                st.session_state.data_generated = True
                
                st.success(f"✅ Generated {len(data)} train operation records!")
    
    # Display generated data if available
    if st.session_state.data_generated:
        st.subheader("📋 Generated Dataset Preview")
        data = st.session_state.train_data
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Average Delay", f"{data['actual_delay'].mean():.1f} min")
        with col3:
            st.metric("Delay Rate", f"{(data['actual_delay'] > 0).mean()*100:.1f}%")
        with col4:
            st.metric("Cancellation Rate", f"{(data['recommended_action'] == 'Cancel').mean()*100:.1f}%")
        
        # Data preview
        st.dataframe(data.head(10), use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Delay distribution
            fig = px.histogram(data, x='actual_delay', nbins=30, 
                             title="Distribution of Train Delays")
            fig.update_layout(xaxis_title="Delay (minutes)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Action distribution
            action_counts = data['recommended_action'].value_counts()
            fig = px.pie(values=action_counts.values, names=action_counts.index,
                        title="Recommended Actions Distribution")
            st.plotly_chart(fig, use_container_width=True)

def model_training_page():
    st.header("🤖 Model Training")
    st.markdown("Train machine learning models for delay prediction and action classification.")
    
    if not st.session_state.data_generated:
        st.warning("⚠️ Please generate data first in the Data Generation section.")
        return
    
    data = st.session_state.train_data
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Configuration")
        
        # Model selection
        regression_model = st.selectbox(
            "Delay Prediction Model",
            ["Random Forest", "XGBoost"],
            help="Choose the algorithm for predicting train delays"
        )
        
        classification_model = st.selectbox(
            "Action Classification Model",
            ["Random Forest", "XGBoost"],
            help="Choose the algorithm for recommending actions"
        )
        
        # Training parameters
        with st.expander("Training Parameters"):
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("Random seed", 0, 1000, 42)
    
    with col2:
        st.subheader("Train Models")
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models..."):
                # Initialize predictors
                delay_predictor = DelayPredictor(model_type=regression_model.lower().replace(" ", "_"))
                action_classifier = ActionClassifier(model_type=classification_model.lower().replace(" ", "_"))
                
                # Train models
                delay_results = delay_predictor.train(data, test_size=test_size, random_state=random_state)
                action_results = action_classifier.train(data, test_size=test_size, random_state=random_state)
                
                # Save to session state
                st.session_state.delay_predictor = delay_predictor
                st.session_state.action_classifier = action_classifier
                st.session_state.delay_results = delay_results
                st.session_state.action_results = action_results
                st.session_state.models_trained = True
                
                st.success("✅ Models trained successfully!")
    
    # Display training results if available
    if st.session_state.models_trained:
        st.subheader("📈 Training Results")
        
        delay_results = st.session_state.delay_results
        action_results = st.session_state.action_results
        
        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Delay MAE", f"{delay_results['mae']:.2f} min")
        with col2:
            st.metric("Delay R² Score", f"{delay_results['r2_score']:.3f}")
        with col3:
            st.metric("Action Accuracy", f"{action_results['accuracy']:.3f}")
        with col4:
            st.metric("Action F1-Score", f"{action_results['f1_score']:.3f}")
        
        # Detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Delay Prediction Results")
            
            # Prediction vs actual plot
            y_true = delay_results['y_test']
            y_pred = delay_results['y_pred']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                name='Predictions',
                opacity=0.6
            ))
            fig.add_trace(go.Scatter(
                x=[y_true.min(), y_true.max()],
                y=[y_true.min(), y_true.max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            fig.update_layout(
                title="Predicted vs Actual Delays",
                xaxis_title="Actual Delay (minutes)",
                yaxis_title="Predicted Delay (minutes)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics table
            metrics_df = pd.DataFrame({
                'Metric': ['MAE', 'RMSE', 'R² Score'],
                'Value': [
                    f"{delay_results['mae']:.2f}",
                    f"{delay_results['rmse']:.2f}",
                    f"{delay_results['r2_score']:.3f}"
                ]
            })
            st.dataframe(metrics_df, hide_index=True)
        
        with col2:
            st.subheader("🎯 Action Classification Results")
            
            # Confusion matrix
            conf_matrix = action_results['confusion_matrix']
            labels = action_results['labels']
            
            fig = px.imshow(
                conf_matrix,
                x=labels,
                y=labels,
                text_auto=True,
                title="Confusion Matrix"
            )
            fig.update_layout(
                xaxis_title="Predicted Action",
                yaxis_title="Actual Action"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification report
            st.text("Classification Report:")
            st.text(action_results['classification_report'])

def optimization_page():
    st.header("⚙️ Schedule Optimization")
    st.markdown("Optimize train schedules using AI predictions to minimize delays and congestion.")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the Model Training section.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Optimization Parameters")
        
        # Optimization strategy
        strategy = st.selectbox(
            "Optimization Strategy",
            ["Greedy", "Weighted Greedy"],
            help="Choose the optimization algorithm"
        )
        
        # Weights for optimization objectives
        st.subheader("Objective Weights")
        weight_delay = st.slider("Passenger Delay Weight", 0.1, 1.0, 0.5, 0.1)
        weight_cancellation = st.slider("Cancellation Weight", 0.1, 1.0, 0.3, 0.1)
        weight_congestion = st.slider("Congestion Weight", 0.1, 1.0, 0.2, 0.1)
        
        # Simulation parameters
        with st.expander("Simulation Parameters"):
            simulation_days = st.slider("Simulation Days", 1, 30, 7)
            trains_per_day = st.slider("Trains per Day", 100, 1000, 500, 50)
    
    with col2:
        st.subheader("Run Optimization")
        
        if st.button("🔧 Optimize Schedule", type="primary"):
            with st.spinner("Running schedule optimization..."):
                # Initialize optimizer
                optimizer = TrainScheduleOptimizer(
                    delay_predictor=st.session_state.delay_predictor,
                    action_classifier=st.session_state.action_classifier,
                    strategy=strategy.lower().replace(" ", "_")
                )
                
                # Set optimization weights
                optimizer.set_weights(
                    passenger_delay=weight_delay,
                    cancellations=weight_cancellation,
                    congestion=weight_congestion
                )
                
                # Generate simulation data
                data = st.session_state.train_data
                simulation_data = data.sample(n=min(len(data), simulation_days * trains_per_day)).reset_index(drop=True)
                
                # Run optimization
                results = optimizer.optimize_schedule(simulation_data)
                
                # Save results
                st.session_state.optimization_results = results
                st.session_state.optimization_done = True
                
                st.success("✅ Schedule optimization completed!")
    
    # Display optimization results
    if st.session_state.optimization_done:
        st.subheader("📊 Optimization Results")
        
        results = st.session_state.optimization_results
        
        # Key metrics comparison
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            improvement = ((results['original_metrics']['avg_delay'] - results['optimized_metrics']['avg_delay']) / 
                          results['original_metrics']['avg_delay'] * 100)
            st.metric(
                "Avg Delay Reduction",
                f"{improvement:.1f}%",
                delta=f"-{results['original_metrics']['avg_delay'] - results['optimized_metrics']['avg_delay']:.1f} min"
            )
        
        with col2:
            on_time_improvement = (results['optimized_metrics']['on_time_rate'] - 
                                 results['original_metrics']['on_time_rate']) * 100
            st.metric(
                "On-Time Rate Improvement",
                f"+{on_time_improvement:.1f}%",
                delta=f"{on_time_improvement:.1f}%"
            )
        
        with col3:
            st.metric(
                "Cancellations",
                results['optimized_metrics']['cancellations'],
                delta=int(results['optimized_metrics']['cancellations'] - results['original_metrics']['cancellations'])
            )
        
        with col4:
            congestion_change = results['optimized_metrics']['congestion_score'] - results['original_metrics']['congestion_score']
            st.metric(
                "Congestion Score",
                f"{results['optimized_metrics']['congestion_score']:.2f}",
                delta=f"{congestion_change:.2f}"
            )
        
        # Detailed comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Before vs After delay distribution
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Original Schedule', 'Optimized Schedule'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Histogram(x=results['original_delays'], name='Original', nbinsx=20),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=results['optimized_delays'], name='Optimized', nbinsx=20),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text="Delay Distribution Comparison",
                showlegend=False
            )
            fig.update_xaxes(title_text="Delay (minutes)")
            fig.update_yaxes(title_text="Frequency")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Action distribution comparison
            actions_df = pd.DataFrame({
                'Action': ['NoChange', 'Delay', 'ShortTurn', 'Cancel'],
                'Original': [
                    results['original_actions'].count('NoChange'),
                    results['original_actions'].count('Delay'),
                    results['original_actions'].count('ShortTurn'),
                    results['original_actions'].count('Cancel')
                ],
                'Optimized': [
                    results['optimized_actions'].count('NoChange'),
                    results['optimized_actions'].count('Delay'),
                    results['optimized_actions'].count('ShortTurn'),
                    results['optimized_actions'].count('Cancel')
                ]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Original',
                x=actions_df['Action'],
                y=actions_df['Original']
            ))
            fig.add_trace(go.Bar(
                name='Optimized',
                x=actions_df['Action'],
                y=actions_df['Optimized']
            ))
            
            fig.update_layout(
                title="Action Distribution Comparison",
                xaxis_title="Action Type",
                yaxis_title="Count",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)

def evaluation_page():
    st.header("📊 Evaluation & Results")
    st.markdown("Comprehensive evaluation of the AI train rescheduling system performance.")
    
    if not st.session_state.optimization_done:
        st.warning("⚠️ Please complete the schedule optimization first.")
        return
    
    # Initialize evaluator
    evaluator = SystemEvaluator(
        delay_predictor=st.session_state.delay_predictor,
        action_classifier=st.session_state.action_classifier
    )
    
    # Comprehensive evaluation
    with st.spinner("Generating comprehensive evaluation..."):
        eval_results = evaluator.comprehensive_evaluation(
            st.session_state.train_data,
            st.session_state.optimization_results
        )
    
    # Display results
    st.subheader("🎯 Overall System Performance")
    
    # Key performance indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "System Efficiency Score",
            f"{eval_results['efficiency_score']:.2f}",
            help="Overall system performance (0-1 scale)"
        )
    
    with col2:
        st.metric(
            "Passenger Satisfaction",
            f"{eval_results['passenger_satisfaction']:.1%}",
            help="Based on delay reduction and service reliability"
        )
    
    with col3:
        st.metric(
            "Network Utilization",
            f"{eval_results['network_utilization']:.1%}",
            help="Efficient use of railway network capacity"
        )
    
    with col4:
        st.metric(
            "Cost Savings",
            f"${eval_results['cost_savings']:,.0f}",
            help="Estimated operational cost savings"
        )
    
    # Detailed evaluation sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Performance Metrics",
        "🔍 Model Analysis",
        "⚡ Optimization Impact",
        "📋 Summary Report"
    ])
    
    with tab1:
        st.subheader("Performance Metrics Overview")
        
        # Model performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Delay Prediction Performance**")
            delay_metrics = pd.DataFrame({
                'Metric': ['Mean Absolute Error', 'Root Mean Square Error', 'R² Score', 'Mean Accuracy'],
                'Value': [
                    f"{st.session_state.delay_results['mae']:.2f} minutes",
                    f"{st.session_state.delay_results['rmse']:.2f} minutes",
                    f"{st.session_state.delay_results['r2_score']:.3f}",
                    f"{eval_results['delay_prediction_accuracy']:.1%}"
                ]
            })
            st.dataframe(delay_metrics, hide_index=True)
        
        with col2:
            st.write("**Action Classification Performance**")
            action_metrics = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Value': [
                    f"{st.session_state.action_results['accuracy']:.3f}",
                    f"{eval_results['action_precision']:.3f}",
                    f"{eval_results['action_recall']:.3f}",
                    f"{st.session_state.action_results['f1_score']:.3f}"
                ]
            })
            st.dataframe(action_metrics, hide_index=True)
        
        # Performance trends
        st.subheader("Performance Trends")
        
        trend_data = eval_results['performance_trends']
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(trend_data['delays']))),
            y=trend_data['delays'],
            mode='lines+markers',
            name='Average Delay',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(trend_data['on_time_rate']))),
            y=[rate * 100 for rate in trend_data['on_time_rate']],
            mode='lines+markers',
            name='On-Time Rate (%)',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="System Performance Over Time",
            xaxis_title="Time Period",
            yaxis=dict(title="Average Delay (minutes)", side="left"),
            yaxis2=dict(title="On-Time Rate (%)", side="right", overlaying="y"),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Model Analysis")
        
        # Feature importance
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Delay Prediction Feature Importance**")
            delay_importance = eval_results['delay_feature_importance']
            
            fig = px.bar(
                x=list(delay_importance.values()),
                y=list(delay_importance.keys()),
                orientation='h',
                title="Feature Importance for Delay Prediction"
            )
            fig.update_layout(xaxis_title="Importance", yaxis_title="Features")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Action Classification Feature Importance**")
            action_importance = eval_results['action_feature_importance']
            
            fig = px.bar(
                x=list(action_importance.values()),
                y=list(action_importance.keys()),
                orientation='h',
                title="Feature Importance for Action Classification"
            )
            fig.update_layout(xaxis_title="Importance", yaxis_title="Features")
            st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        st.subheader("Model Comparison")
        
        comparison_data = eval_results['model_comparison']
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    with tab3:
        st.subheader("Optimization Impact Analysis")
        
        # Before/After comparison
        results = st.session_state.optimization_results
        
        comparison_metrics = pd.DataFrame({
            'Metric': [
                'Average Delay (minutes)',
                'On-Time Rate (%)',
                'Total Cancellations',
                'Congestion Score',
                'Passenger Hours Saved',
                'Network Efficiency (%)'
            ],
            'Original': [
                f"{results['original_metrics']['avg_delay']:.1f}",
                f"{results['original_metrics']['on_time_rate']*100:.1f}",
                f"{results['original_metrics']['cancellations']}",
                f"{results['original_metrics']['congestion_score']:.2f}",
                f"{eval_results['original_passenger_hours']:.0f}",
                f"{eval_results['original_efficiency']*100:.1f}"
            ],
            'Optimized': [
                f"{results['optimized_metrics']['avg_delay']:.1f}",
                f"{results['optimized_metrics']['on_time_rate']*100:.1f}",
                f"{results['optimized_metrics']['cancellations']}",
                f"{results['optimized_metrics']['congestion_score']:.2f}",
                f"{eval_results['optimized_passenger_hours']:.0f}",
                f"{eval_results['optimized_efficiency']*100:.1f}"
            ],
            'Improvement': [
                f"{((results['original_metrics']['avg_delay'] - results['optimized_metrics']['avg_delay'])/results['original_metrics']['avg_delay']*100):+.1f}%",
                f"{(results['optimized_metrics']['on_time_rate'] - results['original_metrics']['on_time_rate'])*100:+.1f}%",
                f"{results['optimized_metrics']['cancellations'] - results['original_metrics']['cancellations']:+d}",
                f"{results['optimized_metrics']['congestion_score'] - results['original_metrics']['congestion_score']:+.2f}",
                f"{eval_results['optimized_passenger_hours'] - eval_results['original_passenger_hours']:+.0f}",
                f"{(eval_results['optimized_efficiency'] - eval_results['original_efficiency'])*100:+.1f}%"
            ]
        })
        
        st.dataframe(comparison_metrics, use_container_width=True)
        
        # ROI Analysis
        st.subheader("Return on Investment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Annual Cost Savings", f"${eval_results['financial_impact']['annual_savings']:,.0f}")
            st.metric("Implementation Cost", f"${eval_results['financial_impact']['implementation_cost']:,.0f}")
            st.metric("Payback Period", f"{eval_results['financial_impact']['payback_period']:.1f} months")
        
        with col2:
            st.metric("5-Year NPV", f"${eval_results['financial_impact']['five_year_npv']:,.0f}")
            st.metric("ROI", f"{eval_results['financial_impact']['roi']:.1f}%")
            st.metric("Break-even Point", f"{eval_results['financial_impact']['break_even_months']:.1f} months")
    
    with tab4:
        st.subheader("Executive Summary Report")
        
        st.markdown(f"""
        ## AI Train Rescheduling System - Performance Report
        
        ### 🎯 **Key Achievements**
        
        - **Delay Reduction**: {((results['original_metrics']['avg_delay'] - results['optimized_metrics']['avg_delay'])/results['original_metrics']['avg_delay']*100):.1f}% average delay reduction
        - **On-Time Performance**: Improved from {results['original_metrics']['on_time_rate']*100:.1f}% to {results['optimized_metrics']['on_time_rate']*100:.1f}%
        - **Passenger Satisfaction**: {eval_results['passenger_satisfaction']:.1%} satisfaction rate
        - **Cost Savings**: ${eval_results['financial_impact']['annual_savings']:,.0f} estimated annual savings
        
        ### 📊 **Model Performance**
        
        **Delay Prediction Model**:
        - Mean Absolute Error: {st.session_state.delay_results['mae']:.2f} minutes
        - R² Score: {st.session_state.delay_results['r2_score']:.3f}
        - Prediction Accuracy: {eval_results['delay_prediction_accuracy']:.1%}
        
        **Action Classification Model**:
        - Overall Accuracy: {st.session_state.action_results['accuracy']:.3f}
        - F1-Score: {st.session_state.action_results['f1_score']:.3f}
        - Precision: {eval_results['action_precision']:.3f}
        
        ### ⚡ **Optimization Impact**
        
        - **Passenger Hours Saved**: {eval_results['optimized_passenger_hours'] - eval_results['original_passenger_hours']:,.0f} hours per period
        - **Network Efficiency**: Improved by {(eval_results['optimized_efficiency'] - eval_results['original_efficiency'])*100:.1f}%
        - **Congestion Reduction**: {results['original_metrics']['congestion_score'] - results['optimized_metrics']['congestion_score']:.2f} point improvement
        
        ### 💰 **Financial Impact**
        
        - **Annual Savings**: ${eval_results['financial_impact']['annual_savings']:,.0f}
        - **ROI**: {eval_results['financial_impact']['roi']:.1f}% return on investment
        - **Payback Period**: {eval_results['financial_impact']['payback_period']:.1f} months
        
        ### 📈 **Recommendations**
        
        1. **Deploy the system** with current performance levels showing significant improvements
        2. **Monitor continuously** to ensure sustained performance gains
        3. **Expand gradually** to additional routes and services
        4. **Integrate real-time data** feeds for enhanced accuracy
        5. **Regular model retraining** to adapt to changing conditions
        
        ### 🔄 **Next Steps**
        
        - Replace synthetic data with real operational data
        - Implement real-time prediction capabilities
        - Add advanced optimization algorithms (ILP, Reinforcement Learning)
        - Integrate with existing railway management systems
        - Conduct pilot testing on selected routes
        """)
        
        # Download report button
        if st.button("📥 Download Full Report"):
            st.info("Report download functionality would be implemented here in a production system.")

def control_center_page():
    """Real-time decision support interface for section controllers."""
    st.header("🚦 Control Center - Real-Time Decision Support")
    st.markdown("**Intelligent decision support for section controllers with real-time train precedence and crossing optimization**")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the Model Training section.")
        return
    
    # Real-time status section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚂 Real-Time Train Status")
        
        # Generate current operational data
        if st.button("🔄 Refresh Real-Time Data", type="primary"):
            from data_generator import TrainDataGenerator
            generator = TrainDataGenerator()
            current_data = generator.generate_real_time_data(30)
            
            # Add simulated real-time fields
            current_data['current_location'] = [f"Section {i%10+1}" for i in range(len(current_data))]
            current_data['next_signal'] = [f"Signal {i%20+1}" for i in range(len(current_data))]
            current_data['conflict_detected'] = np.random.choice([True, False], len(current_data), p=[0.2, 0.8])
            current_data['priority_level'] = np.random.choice(['High', 'Medium', 'Low'], len(current_data), p=[0.3, 0.5, 0.2])
            
            st.session_state.realtime_data = current_data
        
        # Display current trains
        if 'realtime_data' in st.session_state:
            data = st.session_state.realtime_data
            
            # Status metrics
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Active Trains", len(data))
            with col_b:
                conflicts = int(data['conflict_detected'].sum())
                st.metric("Conflicts Detected", conflicts, delta=f"{conflicts} requiring attention")
            with col_c:
                high_priority = int((data['priority_level'] == 'High').sum())
                st.metric("High Priority", high_priority)
            with col_d:
                avg_load = data['passenger_load_percentage'].mean()
                st.metric("Avg Load", f"{avg_load:.0f}%")
            
            # Train status table
            display_data = data[['train_id', 'train_type', 'current_location', 'next_signal', 
                               'passenger_load_percentage', 'conflict_detected', 'priority_level']].copy()
            display_data.columns = ['Train ID', 'Type', 'Location', 'Next Signal', 'Load %', 'Conflict', 'Priority']
            
            # Color code conflicts
            st.dataframe(
                display_data,
                use_container_width=True
            )
    
    with col2:
        st.subheader("⚡ Smart Recommendations")
        
        if 'realtime_data' in st.session_state:
            data = st.session_state.realtime_data
            
            # Generate AI recommendations
            if st.button("🤖 Generate Recommendations"):
                from optimizer import TrainScheduleOptimizer
                optimizer = TrainScheduleOptimizer(
                    st.session_state.delay_predictor,
                    st.session_state.action_classifier,
                    strategy='weighted_greedy'
                )
                
                # Generate recommendations for conflicted trains
                conflict_trains = data[data['conflict_detected']]
                
                recommendations = []
                if len(conflict_trains) > 0:
                    for _, train in conflict_trains.iterrows():
                        rec = generate_controller_recommendation(train, data)
                        recommendations.append(rec)
                else:
                    recommendations = []
                
                st.session_state.recommendations = recommendations
            
            # Display recommendations
            if 'recommendations' in st.session_state:
                for i, rec in enumerate(st.session_state.recommendations):
                    with st.expander(f"🚨 {rec['train_id']} - {rec['action']}", expanded=True):
                        st.write(f"**Situation:** {rec['situation']}")
                        st.write(f"**Recommendation:** {rec['recommendation']}")
                        st.write(f"**Expected Impact:** {rec['impact']}")
                        
                        col_x, col_y = st.columns(2)
                        with col_x:
                            if st.button(f"✅ Accept", key=f"accept_{i}"):
                                st.success(f"Recommendation accepted for {rec['train_id']}")
                        with col_y:
                            if st.button(f"❌ Override", key=f"override_{i}"):
                                st.info(f"Manual override recorded for {rec['train_id']}")
    
    # Section controller tools
    st.subheader("🎛️ Section Controller Tools")
    
    tab1, tab2, tab3 = st.tabs(["Manual Controls", "Crossing Management", "Emergency Protocols"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Signal Control**")
            selected_signal = st.selectbox("Select Signal", [f"Signal {i}" for i in range(1, 21)])
            signal_action = st.radio("Signal Action", ["Clear", "Caution", "Stop"])
            if st.button("Apply Signal Command"):
                st.success(f"Signal {selected_signal} set to {signal_action}")
        
        with col2:
            st.write("**Platform Management**")
            selected_platform = st.selectbox("Select Platform", [f"Platform {i}" for i in range(1, 11)])
            platform_action = st.radio("Platform Action", ["Available", "Occupied", "Maintenance"])
            if st.button("Update Platform Status"):
                st.success(f"Platform {selected_platform} status: {platform_action}")
    
    with tab2:
        st.write("**Crossing Conflict Resolution**")
        if 'realtime_data' in st.session_state:
            conflicts = st.session_state.realtime_data[st.session_state.realtime_data['conflict_detected']]
            if len(conflicts) > 0:
                for _, conflict in conflicts.head(3).iterrows():
                    st.write(f"🚨 **Conflict:** Train {conflict['train_id']} at {conflict['current_location']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"Hold {conflict['train_id']}", key=f"hold_{conflict['train_id']}"):
                            st.info(f"Train {conflict['train_id']} held")
                    with col2:
                        if st.button(f"Reroute {conflict['train_id']}", key=f"reroute_{conflict['train_id']}"):
                            st.info(f"Rerouting {conflict['train_id']}")
                    with col3:
                        if st.button(f"Priority Pass {conflict['train_id']}", key=f"priority_{conflict['train_id']}"):
                            st.info(f"Priority given to {conflict['train_id']}")
            else:
                st.success("✅ No crossing conflicts detected")
    
    with tab3:
        st.write("**Emergency Response Protocols**")
        emergency_type = st.selectbox("Emergency Type", 
                                    ["Signal Failure", "Track Obstruction", "Medical Emergency", "Weather Alert"])
        affected_section = st.selectbox("Affected Section", [f"Section {i}" for i in range(1, 11)])
        
        if st.button("🚨 Activate Emergency Protocol", type="primary"):
            st.error(f"Emergency protocol activated for {emergency_type} in {affected_section}")
            st.info("All trains in affected section will be automatically held and alternative routes calculated.")

def whatif_simulation_page():
    """What-if simulation and scenario analysis interface."""
    st.header("🎯 What-If Simulation & Scenario Analysis")
    st.markdown("**Evaluate alternative strategies and analyze potential outcomes before implementation**")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the Model Training section.")
        return
    
    # Scenario setup
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Scenario Configuration")
        
        scenario_type = st.selectbox(
            "Scenario Type",
            ["Normal Operations", "Weather Disruption", "Signal Failure", "Track Maintenance", "Peak Hour Rush", "Emergency Situation"]
        )
        
        # Scenario-specific parameters
        if scenario_type == "Weather Disruption":
            weather_severity = st.slider("Weather Severity Impact", 1.0, 3.0, 1.5, 0.1)
            affected_routes = st.multiselect("Affected Routes", [f"Route {i}" for i in range(1, 11)], ["Route 1", "Route 3"])
        
        elif scenario_type == "Signal Failure":
            failed_signals = st.multiselect("Failed Signals", [f"Signal {i}" for i in range(1, 21)], ["Signal 5"])
            failure_duration = st.slider("Estimated Repair Time (hours)", 0.5, 8.0, 2.0, 0.5)
        
        elif scenario_type == "Track Maintenance":
            maintenance_sections = st.multiselect("Maintenance Sections", [f"Section {i}" for i in range(1, 11)], ["Section 4"])
            maintenance_duration = st.slider("Maintenance Duration (hours)", 1, 12, 4)
        
        simulation_duration = st.slider("Simulation Duration (hours)", 1, 24, 8)
        num_trains = st.slider("Number of Trains", 50, 500, 200, 25)
        
    with col2:
        st.subheader("⚙️ Alternative Strategies")
        
        # Strategy options
        holding_strategy = st.selectbox(
            "Holding Strategy",
            ["Minimal Holding", "Strategic Holding", "Dynamic Holding"]
        )
        
        rerouting_enabled = st.checkbox("Enable Automatic Rerouting", True)
        if rerouting_enabled:
            rerouting_aggressiveness = st.slider("Rerouting Aggressiveness", 0.1, 1.0, 0.6, 0.1)
        
        priority_override = st.checkbox("Allow Priority Override", False)
        if priority_override:
            priority_trains = st.multiselect("Priority Trains", [f"Train {i}" for i in range(1, 21)])
        
        capacity_management = st.selectbox(
            "Capacity Management",
            ["Standard", "Load Balancing", "Express Priority"]
        )
    
    # Run simulation
    if st.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Running scenario simulation..."):
            # Generate simulation data
            from data_generator import TrainDataGenerator
            generator = TrainDataGenerator()
            sim_data = generator.generate_dataset(num_trains)
            
            # Apply scenario modifications
            sim_data = apply_scenario_modifications(sim_data, scenario_type, locals())
            
            # Run optimization with different strategies
            from optimizer import TrainScheduleOptimizer
            
            # Baseline optimization
            optimizer_baseline = TrainScheduleOptimizer(
                st.session_state.delay_predictor,
                st.session_state.action_classifier,
                strategy='greedy'
            )
            baseline_results = optimizer_baseline.optimize_schedule(sim_data)
            
            # Alternative strategy optimization
            optimizer_alternative = TrainScheduleOptimizer(
                st.session_state.delay_predictor,
                st.session_state.action_classifier,
                strategy='weighted_greedy'
            )
            optimizer_alternative.set_weights(
                passenger_delay=0.4 if holding_strategy == "Strategic Holding" else 0.6,
                cancellations=0.4,
                congestion=0.2
            )
            alternative_results = optimizer_alternative.optimize_schedule(sim_data)
            
            # Store results
            st.session_state.simulation_results = {
                'scenario_type': scenario_type,
                'baseline': baseline_results,
                'alternative': alternative_results,
                'parameters': {
                    'holding_strategy': holding_strategy,
                    'rerouting_enabled': rerouting_enabled,
                    'capacity_management': capacity_management
                }
            }
            
            st.success("✅ Simulation completed!")
    
    # Display simulation results
    if 'simulation_results' in st.session_state:
        st.subheader("📈 Simulation Results")
        
        results = st.session_state.simulation_results
        baseline = results['baseline']
        alternative = results['alternative']
        
        # Comparison metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            baseline_delay = baseline['original_metrics']['avg_delay']
            alternative_delay = alternative['optimized_metrics']['avg_delay']
            improvement = ((baseline_delay - alternative_delay) / baseline_delay * 100) if baseline_delay > 0 else 0
            st.metric(
                "Delay Improvement",
                f"{improvement:.1f}%",
                delta=f"-{baseline_delay - alternative_delay:.1f} min"
            )
        
        with col2:
            baseline_ontime = baseline['original_metrics']['on_time_rate']
            alternative_ontime = alternative['optimized_metrics']['on_time_rate']
            ontime_improvement = (alternative_ontime - baseline_ontime) * 100
            st.metric(
                "On-Time Rate",
                f"{alternative_ontime*100:.1f}%",
                delta=f"+{ontime_improvement:.1f}%"
            )
        
        with col3:
            baseline_cancellations = baseline['original_metrics']['cancellations']
            alternative_cancellations = alternative['optimized_metrics']['cancellations']
            st.metric(
                "Cancellations",
                int(alternative_cancellations),
                delta=int(alternative_cancellations - baseline_cancellations)
            )
        
        with col4:
            baseline_congestion = baseline['original_metrics']['congestion_score']
            alternative_congestion = alternative['optimized_metrics']['congestion_score']
            st.metric(
                "Congestion Score",
                f"{alternative_congestion:.2f}",
                delta=f"{alternative_congestion - baseline_congestion:.2f}"
            )
        
        # Strategy comparison
        st.subheader("📊 Strategy Comparison")
        
        comparison_df = pd.DataFrame({
            'Metric': ['Average Delay (min)', 'On-Time Rate (%)', 'Cancellations', 'Congestion Score', 'Throughput'],
            'Baseline Strategy': [
                f"{baseline['original_metrics']['avg_delay']:.1f}",
                f"{baseline['original_metrics']['on_time_rate']*100:.1f}",
                baseline['original_metrics']['cancellations'],
                f"{baseline['original_metrics']['congestion_score']:.2f}",
                baseline['original_metrics']['total_trains']
            ],
            'Alternative Strategy': [
                f"{alternative['optimized_metrics']['avg_delay']:.1f}",
                f"{alternative['optimized_metrics']['on_time_rate']*100:.1f}",
                alternative['optimized_metrics']['cancellations'],
                f"{alternative['optimized_metrics']['congestion_score']:.2f}",
                alternative['optimized_metrics']['total_trains']
            ],
            'Improvement': [
                f"{improvement:.1f}%",
                f"+{ontime_improvement:.1f}%",
                f"{alternative_cancellations - baseline_cancellations:+d}",
                f"{alternative_congestion - baseline_congestion:+.2f}",
                "0"
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Simulation Insights")
        
        if improvement > 10:
            st.success(f"✅ Alternative strategy shows significant improvement ({improvement:.1f}% delay reduction)")
            st.info("**Recommendation:** Implement the alternative strategy for this scenario type.")
        elif improvement > 5:
            st.info(f"⚠️ Alternative strategy shows moderate improvement ({improvement:.1f}% delay reduction)")
            st.info("**Recommendation:** Consider implementing with close monitoring.")
        else:
            st.warning(f"❗ Alternative strategy shows minimal improvement ({improvement:.1f}% delay reduction)")
            st.info("**Recommendation:** Stick with baseline strategy or explore other alternatives.")

def performance_dashboard_page():
    """Performance monitoring dashboard with KPIs and audit trails."""
    st.header("📊 Performance Dashboard")
    st.markdown("**Real-time KPIs, audit trails, and continuous improvement metrics**")
    
    # KPI Overview
    st.subheader("🎯 Key Performance Indicators")
    
    # Generate sample KPI data
    if 'kpi_data' not in st.session_state:
        st.session_state.kpi_data = generate_sample_kpi_data()
    
    kpi_data = st.session_state.kpi_data
    
    # Current KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "System Availability",
            f"{kpi_data['availability']:.1f}%",
            delta=f"+{np.random.uniform(0.1, 0.5):.1f}%"
        )
    
    with col2:
        st.metric(
            "Average Punctuality",
            f"{kpi_data['punctuality']:.1f}%",
            delta=f"+{np.random.uniform(1, 3):.1f}%"
        )
    
    with col3:
        st.metric(
            "Throughput (trains/hour)",
            f"{kpi_data['throughput']:.0f}",
            delta=f"+{np.random.randint(2, 8)}"
        )
    
    with col4:
        st.metric(
            "Network Utilization",
            f"{kpi_data['utilization']:.1f}%",
            delta=f"+{np.random.uniform(0.5, 2.0):.1f}%"
        )
    
    with col5:
        st.metric(
            "Customer Satisfaction",
            f"{kpi_data['satisfaction']:.1f}",
            delta=f"+{np.random.uniform(0.1, 0.3):.1f}"
        )
    
    # Performance trends
    tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Audit Trail", "System Health", "Reports"])
    
    with tab1:
        st.subheader("📈 Performance Trends")
        
        # Generate trend data
        dates = pd.date_range(start='2025-08-01', end='2025-09-05', freq='D')
        trend_data = pd.DataFrame({
            'Date': dates,
            'Punctuality': np.random.normal(85, 5, len(dates)).clip(70, 100),
            'Throughput': np.random.normal(45, 8, len(dates)).clip(20, 70),
            'Delays': np.random.normal(12, 4, len(dates)).clip(5, 25),
            'Satisfaction': np.random.normal(4.2, 0.3, len(dates)).clip(3.0, 5.0)
        })
        
        # Punctuality trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_data['Date'],
            y=trend_data['Punctuality'],
            mode='lines+markers',
            name='Punctuality (%)',
            line=dict(color='green')
        ))
        fig.update_layout(
            title="Punctuality Trend (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Punctuality (%)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Multi-metric dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_data['Date'],
                y=trend_data['Throughput'],
                mode='lines+markers',
                name='Throughput',
                line=dict(color='blue')
            ))
            fig.update_layout(
                title="Daily Throughput",
                xaxis_title="Date",
                yaxis_title="Trains/Hour"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_data['Date'],
                y=trend_data['Delays'],
                mode='lines+markers',
                name='Average Delay',
                line=dict(color='red')
            ))
            fig.update_layout(
                title="Average Daily Delays",
                xaxis_title="Date",
                yaxis_title="Minutes"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📋 Audit Trail")
        
        # Generate sample audit data
        audit_data = generate_sample_audit_data()
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            action_filter = st.selectbox("Filter by Action", ["All"] + audit_data['Action'].unique().tolist())
        with col2:
            user_filter = st.selectbox("Filter by User", ["All"] + audit_data['User'].unique().tolist())
        with col3:
            date_filter = st.date_input("From Date", value=(pd.Timestamp.now() - pd.Timedelta(days=7)).date())
        
        # Apply filters
        filtered_data = audit_data.copy()
        if action_filter != "All":
            filtered_data = filtered_data[filtered_data['Action'] == action_filter]
        if user_filter != "All":
            filtered_data = filtered_data[filtered_data['User'] == user_filter]
        
        st.dataframe(filtered_data, use_container_width=True)
        
        # Audit statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Actions Today", len(filtered_data[pd.to_datetime(filtered_data['Timestamp']).dt.date == pd.Timestamp.now().date()]))
        with col2:
            st.metric("Manual Overrides", len(filtered_data[filtered_data['Action'] == 'Manual Override']))
        with col3:
            st.metric("System Actions", len(filtered_data[filtered_data['User'] == 'System']))
    
    with tab3:
        st.subheader("🔧 System Health Monitoring")
        
        # System components status
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**System Components Status**")
            components = [
                {"name": "AI Models", "status": "Operational", "uptime": "99.8%"},
                {"name": "Data Pipeline", "status": "Operational", "uptime": "99.9%"},
                {"name": "Optimization Engine", "status": "Operational", "uptime": "99.7%"},
                {"name": "API Gateway", "status": "Operational", "uptime": "99.6%"},
                {"name": "Database", "status": "Operational", "uptime": "99.9%"}
            ]
            
            for comp in components:
                col_a, col_b, col_c = st.columns([2, 1, 1])
                with col_a:
                    st.write(comp["name"])
                with col_b:
                    st.success(comp["status"])
                with col_c:
                    st.write(comp["uptime"])
        
        with col2:
            st.write("**Performance Metrics**")
            
            # Response time chart
            response_times = np.random.normal(250, 50, 24)
            hours = list(range(24))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours,
                y=response_times,
                mode='lines+markers',
                name='Response Time (ms)',
                line=dict(color='purple')
            ))
            fig.update_layout(
                title="24-Hour Response Time",
                xaxis_title="Hour",
                yaxis_title="Response Time (ms)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("📄 Performance Reports")
        
        report_type = st.selectbox(
            "Report Type",
            ["Daily Summary", "Weekly Analysis", "Monthly Report", "Incident Analysis", "ROI Report"]
        )
        
        report_period = st.date_input("Report Period", value=pd.Timestamp.now().date())
        
        if st.button("📊 Generate Report"):
            with st.spinner("Generating report..."):
                time.sleep(2)  # Simulate report generation
                
                if report_type == "Daily Summary":
                    st.markdown(f"""
                    ## Daily Performance Summary - {report_period}
                    
                    ### 🎯 Key Metrics
                    - **Trains Processed:** 1,247
                    - **Average Delay:** 8.3 minutes (↓ 15% from yesterday)
                    - **On-Time Performance:** 87.2% (↑ 3.1% from yesterday)
                    - **Cancellations:** 12 (↓ 4 from yesterday)
                    - **System Uptime:** 99.8%
                    
                    ### 📈 Performance Highlights
                    - Peak hour efficiency improved by 12%
                    - Weather impact mitigation reduced delays by 23%
                    - 3 manual interventions prevented potential conflicts
                    
                    ### ⚠️ Issues & Resolutions
                    - Signal malfunction at Junction 5 (resolved in 45 minutes)
                    - Track maintenance caused 8-minute average delay on Line 3
                    - Emergency protocol activated once for medical incident
                    """)
                
                elif report_type == "Weekly Analysis":
                    st.markdown(f"""
                    ## Weekly Performance Analysis - Week ending {report_period}
                    
                    ### 📊 Weekly Trends
                    - **Total Trains:** 8,731 (↑ 2.3% from last week)
                    - **Average Weekly Delay:** 9.1 minutes (↓ 8% from last week)
                    - **Weekly On-Time Rate:** 85.7% (↑ 2.8% from last week)
                    - **Customer Satisfaction:** 4.3/5.0 (↑ 0.2 from last week)
                    
                    ### 🎯 Achievement Summary
                    - Exceeded punctuality target for 5 out of 7 days
                    - Implemented 23 AI-recommended optimizations
                    - Zero safety incidents reported
                    - Passenger complaints reduced by 18%
                    
                    ### 🔍 Areas for Improvement
                    - Morning peak hour delays still 12% above target
                    - Weekend service reliability needs attention
                    - Integration with external weather services required
                    """)
                
                st.success("✅ Report generated successfully!")
                
                # Export options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Export as PDF"):
                        st.info("PDF export functionality would be implemented here")
                with col2:
                    if st.button("📧 Email Report"):
                        st.info("Email functionality would be implemented here")

def generate_controller_recommendation(train, all_data):
    """Generate AI-powered recommendation for section controller."""
    situations = {
        'High': 'Critical delay detected with high passenger load',
        'Medium': 'Moderate delay with potential for escalation',
        'Low': 'Minor delay requiring attention'
    }
    
    recommendations = {
        'High': 'Immediate priority routing with alternative path calculation',
        'Medium': 'Strategic holding at next signal for optimal slot',
        'Low': 'Continue with current path, monitor closely'
    }
    
    impacts = {
        'High': 'Reduce delay by 15-25 minutes, improve passenger satisfaction',
        'Medium': 'Reduce delay by 8-15 minutes, maintain schedule integrity',
        'Low': 'Prevent delay escalation, maintain current performance'
    }
    
    priority = train['priority_level']
    
    return {
        'train_id': train['train_id'],
        'action': f"{priority} Priority Action",
        'situation': situations[priority],
        'recommendation': recommendations[priority],
        'impact': impacts[priority]
    }

def apply_scenario_modifications(data, scenario_type, params):
    """Apply scenario-specific modifications to simulation data."""
    modified_data = data.copy()
    
    if scenario_type == "Weather Disruption":
        weather_impact = params.get('weather_severity', 1.5)
        modified_data['actual_delay'] *= weather_impact
        modified_data['weather_severity'] = 'Severe'
    
    elif scenario_type == "Signal Failure":
        # Increase delays for affected areas
        modified_data['actual_delay'] += np.random.uniform(10, 30, len(modified_data))
    
    elif scenario_type == "Peak Hour Rush":
        # Increase passenger load and delays
        modified_data['passenger_load_percentage'] *= 1.3
        modified_data['passenger_load_percentage'] = modified_data['passenger_load_percentage'].clip(0, 100)
        modified_data['actual_delay'] *= 1.2
    
    return modified_data

def generate_sample_kpi_data():
    """Generate sample KPI data for dashboard."""
    return {
        'availability': np.random.uniform(98, 100),
        'punctuality': np.random.uniform(80, 95),
        'throughput': np.random.uniform(40, 60),
        'utilization': np.random.uniform(75, 90),
        'satisfaction': np.random.uniform(3.8, 4.8)
    }

def generate_sample_audit_data():
    """Generate sample audit trail data."""
    actions = ['Manual Override', 'Schedule Optimization', 'Signal Change', 'Platform Assignment', 'Emergency Protocol']
    users = ['Controller_A', 'Controller_B', 'System', 'Supervisor_1', 'Maintenance']
    
    data = []
    for i in range(50):
        timestamp = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 30), 
                                                    hours=np.random.randint(0, 24))
        data.append({
            'Timestamp': timestamp,
            'Action': np.random.choice(actions),
            'User': np.random.choice(users),
            'Target': f'Train_{np.random.randint(1, 200)}',
            'Result': np.random.choice(['Success', 'Success', 'Success', 'Failed']),
            'Details': f'Action performed with parameter set {np.random.randint(1, 10)}'
        })
    
    return pd.DataFrame(data).sort_values('Timestamp', ascending=False)

if __name__ == "__main__":
    main()
