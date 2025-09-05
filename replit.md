# Overview

This is an AI-Based Train Rescheduling System that uses machine learning to predict train delays and optimize schedule decisions. The system generates synthetic train operational data, trains predictive models for delays and scheduling actions, and provides optimization algorithms to minimize passenger impact and operational costs. It features a Streamlit web interface for interactive use and comprehensive evaluation metrics including ROI analysis.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The system uses **Streamlit** as the web framework, providing an interactive dashboard with multiple pages for data generation, model training, optimization, and evaluation. The interface is organized with a sidebar navigation system and maintains session state across different operations.

## Machine Learning Architecture
The system implements a **dual-model approach**:
- **Delay Predictor**: Uses Random Forest or XGBoost regression to predict train delays based on operational features
- **Action Classifier**: Uses Random Forest classification to recommend scheduling actions (NoChange, Delay, ShortTurn, Cancel)

Both models support feature encoding, standardization, and comprehensive performance evaluation with metrics like MAE, RMSE, and accuracy scores.

## Data Generation System
A **synthetic data generator** creates realistic train operational scenarios with configurable parameters including:
- Number of trains and stations
- Delay and weather probabilities
- Train types (Express/Local) with different characteristics
- Seasonal and holiday effects

## Optimization Engine
The system implements **multiple optimization strategies**:
- Greedy optimization for quick decisions
- Weighted optimization balancing passenger delays, cancellations, and network congestion
- Configurable objective weights and action costs

## Evaluation Framework
A **comprehensive evaluation system** provides:
- Model performance metrics (accuracy, precision, recall)
- Optimization impact analysis
- Financial ROI calculations with configurable cost parameters
- Before/after comparisons of scheduling decisions

## Data Processing
The architecture includes utility classes for data handling, file operations (CSV, JSON, pickle formats), and feature preprocessing with label encoding and standardization.

# External Dependencies

## Core ML Libraries
- **scikit-learn**: Primary machine learning framework for Random Forest models, preprocessing, and evaluation metrics
- **XGBoost**: Alternative gradient boosting algorithm for delay prediction
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing operations

## Visualization and UI
- **Streamlit**: Web application framework for the interactive dashboard
- **plotly**: Interactive plotting library for charts and graphs (express and graph_objects modules)

## Data Persistence
- **pickle**: Model serialization and data storage
- Built-in **json** and **csv** modules for data export/import

## Utility Libraries
- **datetime**: Date and time operations for scheduling
- **random**: Random number generation for data synthesis
- **copy**: Deep copying for optimization operations
- **os**: File system operations

The system is designed to be self-contained with no external APIs or databases, relying on synthetic data generation for demonstration and testing purposes.