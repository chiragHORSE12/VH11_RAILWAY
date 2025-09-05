import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

from data_generator import TrainDataGenerator
from models import DelayPredictor, ActionClassifier
from optimizer import TrainScheduleOptimizer
from evaluator import SystemEvaluator

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
        ["Data Generation", "Model Training", "Schedule Optimization", "Evaluation & Results"]
    )
    
    # Initialize session state
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'optimization_done' not in st.session_state:
        st.session_state.optimization_done = False
    
    if page == "Data Generation":
        data_generation_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Schedule Optimization":
        optimization_page()
    elif page == "Evaluation & Results":
        evaluation_page()

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
                delta=results['optimized_metrics']['cancellations'] - results['original_metrics']['cancellations']
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
            st.metric("Annual Cost Savings", f"${eval_results['annual_savings']:,.0f}")
            st.metric("Implementation Cost", f"${eval_results['implementation_cost']:,.0f}")
            st.metric("Payback Period", f"{eval_results['payback_period']:.1f} months")
        
        with col2:
            st.metric("5-Year NPV", f"${eval_results['five_year_npv']:,.0f}")
            st.metric("ROI", f"{eval_results['roi']:.1f}%")
            st.metric("Break-even Point", f"{eval_results['break_even_months']:.1f} months")
    
    with tab4:
        st.subheader("Executive Summary Report")
        
        st.markdown(f"""
        ## AI Train Rescheduling System - Performance Report
        
        ### 🎯 **Key Achievements**
        
        - **Delay Reduction**: {((results['original_metrics']['avg_delay'] - results['optimized_metrics']['avg_delay'])/results['original_metrics']['avg_delay']*100):.1f}% average delay reduction
        - **On-Time Performance**: Improved from {results['original_metrics']['on_time_rate']*100:.1f}% to {results['optimized_metrics']['on_time_rate']*100:.1f}%
        - **Passenger Satisfaction**: {eval_results['passenger_satisfaction']:.1%} satisfaction rate
        - **Cost Savings**: ${eval_results['annual_savings']:,.0f} estimated annual savings
        
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
        
        - **Annual Savings**: ${eval_results['annual_savings']:,.0f}
        - **ROI**: {eval_results['roi']:.1f}% return on investment
        - **Payback Period**: {eval_results['payback_period']:.1f} months
        
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

if __name__ == "__main__":
    main()
