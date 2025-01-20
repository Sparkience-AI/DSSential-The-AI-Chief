import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import google.generativeai as genai
from scipy.stats import norm, stats
import optuna
from SALib.sample import saltelli
from SALib.analyze import sobol
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta


class TimeSeriesModel:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.last_train_date = None
        
    def prepare_data_for_prophet(self, data, target_col):
        """
        Prepare data for Prophet model with improved date handling.
        
        Args:
            data (pd.DataFrame or pd.Series): Input time series data
            target_col (str): Name of the target column
            
        Returns:
            pd.DataFrame: DataFrame formatted for Prophet
        """
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Handle datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            df['ds'] = df.index
        else:
            # Try to find a date column
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                df['ds'] = df[date_cols[0]]
            else:
                # Create artificial dates
                df['ds'] = pd.date_range(
                    start=datetime.now() - timedelta(days=len(df)),
                    periods=len(df),
                    freq='D'
                )
        
        # Ensure target column is properly handled
        if target_col in df.columns:
            df['y'] = df[target_col]
        else:
            df['y'] = df.iloc[:, 0]  # Take the first column if target_col not found
        
        # Convert to numeric, handle any non-numeric values
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna(subset=['ds', 'y'])
        
        # Ensure ds column is datetime
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Sort by date
        df = df.sort_values('ds')
        
        # Scale the target variable
        df['y'] = self.scaler.fit_transform(df[['y']].values.reshape(-1, 1)).ravel()
        
        # Select only required columns
        prophet_data = df[['ds', 'y']].copy()
        
        return prophet_data
    
    def train(self, data, target_col, **kwargs):
        """
        Train Prophet model on the data.
        
        Args:
            data (pd.DataFrame or pd.Series): Input time series data
            target_col (str): Name of the target column
            **kwargs: Additional arguments for Prophet model
        """
        try:
            # Prepare data
            prophet_data = self.prepare_data_for_prophet(data, target_col)
            
            # Store the last training date
            self.last_train_date = prophet_data['ds'].max()
            
            # Initialize and train Prophet model with additional parameters
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                changepoint_range=0.9,  # Allow changepoints up to 90% of the time series
                n_changepoints=25,      # Number of potential changepoints
                **kwargs
            )
            
            # Add additional seasonality if data spans multiple years
            date_range = (prophet_data['ds'].max() - prophet_data['ds'].min()).days
            if date_range > 730:  # More than 2 years
                self.model.add_seasonality(name='biannual', period=365*2, fourier_order=5)
            
            self.model.fit(prophet_data)
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def forecast(self, steps=30, confidence_interval=0.95, future_periods=365):
        """
        Generate forecasts with confidence intervals.
        
        Args:
            steps (int): Number of steps to forecast for initial period
            confidence_interval (float): Confidence interval (0-1)
            future_periods (int): Number of days to forecast into the future
            
        Returns:
            dict: Dictionary containing forecast results
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            # Create future dataframe for both near-term and long-term forecasts
            future = self.model.make_future_dataframe(periods=max(steps, future_periods))
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Prepare arrays for inverse transform
            yhat = forecast[['yhat']].values.reshape(-1, 1)
            yhat_lower = forecast[['yhat_lower']].values.reshape(-1, 1)
            yhat_upper = forecast[['yhat_upper']].values.reshape(-1, 1)
            
            # Inverse transform the scaled values
            forecast_values = pd.DataFrame({
                'ds': forecast['ds'],
                'yhat': self.scaler.inverse_transform(yhat).ravel(),
                'yhat_lower': self.scaler.inverse_transform(yhat_lower).ravel(),
                'yhat_upper': self.scaler.inverse_transform(yhat_upper).ravel()
            })
            
            # Add flags for historical vs forecast periods
            forecast_values['is_forecast'] = forecast_values['ds'] > self.last_train_date
            
            return {
                'forecast': forecast_values,
                'components': self.model.plot_components(forecast)
            }
            
        except Exception as e:
            print(f"Error generating forecast: {str(e)}")
            return None
    

    def create_forecast_plot(self, original_data, forecast_results, target_col):
        """
        Create an interactive plot of the forecast with all lines visible.
        
        Args:
            original_data (pd.DataFrame): Original time series data
            forecast_results (dict): Results from forecast method
            target_col (str): Name of the target column
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()
        
        # Ensure original data has datetime index
        if not isinstance(original_data.index, pd.DatetimeIndex):
            # Try to find a date column
            date_cols = original_data.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                original_data.set_index(date_cols[0], inplace=True)
            else:
                # If no date column, create artificial dates
                original_data.set_index(pd.date_range(
                    start=datetime.now() - timedelta(days=len(original_data)),
                    periods=len(original_data),
                    freq='D'
                ), inplace=True)
        
        # Plot original data
        fig.add_trace(go.Scatter(
            x=original_data.index,
            y=original_data[target_col],
            name='Historical Data',
            line=dict(color='blue', width=2),
            mode='lines'
        ))
        
        forecast = forecast_results['forecast']
        
        # Split forecast into historical and future periods
        historical_forecast = forecast[~forecast['is_forecast']]
        future_forecast = forecast[forecast['is_forecast']]
        
        # Plot historical forecast (fitted values)
        fig.add_trace(go.Scatter(
            x=historical_forecast['ds'],
            y=historical_forecast['yhat'],
            name='Historical Forecast',
            line=dict(color='red', width=2, dash='dot'),
            mode='lines'
        ))
        
        # Plot future forecast
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat'],
            name='Future Forecast',
            line=dict(color='red', width=3),
            mode='lines'
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(231,234,241,0.5)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
        
        # Add vertical line at the forecast start point
        fig.add_shape(
            type="line",
            x0=self.last_train_date,
            x1=self.last_train_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="gray", dash="dash"),
        )
        
        # Add annotation for the forecast start
        fig.add_annotation(
            x=self.last_train_date,
            y=1,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            yshift=10
        )
        
        # Update layout with better defaults
        fig.update_layout(
            title='Time Series Forecast with Future Predictions',
            xaxis_title='Date',
            yaxis_title='Value',
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Add range slider and ensure proper date formatting
        fig.update_xaxes(
            rangeslider_visible=True,
            type='date'
        )
        
        # Ensure y-axis accommodates all values
        y_values = pd.concat([
            pd.Series(original_data[target_col]),
            forecast['yhat'],
            forecast['yhat_upper'],
            forecast['yhat_lower']
        ])
        
        y_min = y_values.min()
        y_max = y_values.max()
        y_range = y_max - y_min
        
        fig.update_yaxes(
            range=[y_min - 0.1 * y_range, y_max + 0.1 * y_range]
        )
        
        return fig
    # Inside the TimeSeriesModel class, add this new method:
    def get_forecast_insights(self, original_data, forecast_results, target_col):
        """
        Prepare forecast insights for AI analysis.
        
        Args:
            original_data (pd.DataFrame): Original time series data
            forecast_results (dict): Results from forecast method
            target_col (str): Name of the target column
            
        Returns:
            dict: Dictionary containing forecast insights
        """
        forecast = forecast_results['forecast']
        
        # Calculate trend
        trend_direction = "increasing" if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[0] else "decreasing"
        
        # Calculate forecast statistics
        original_stats = {
            "mean": original_data[target_col].mean(),
            "std": original_data[target_col].std(),
            "min": original_data[target_col].min(),
            "max": original_data[target_col].max()
        }
        
        forecast_stats = {
            "mean": forecast['yhat'].mean(),
            "std": forecast['yhat'].std(),
            "min": forecast['yhat'].min(),
            "max": forecast['yhat'].max()
        }
        
        # Calculate uncertainty metrics
        uncertainty_width = (forecast['yhat_upper'] - forecast['yhat_lower']).mean()
        uncertainty_growth = (
            (forecast['yhat_upper'] - forecast['yhat_lower']).iloc[-1] /
            (forecast['yhat_upper'] - forecast['yhat_lower']).iloc[0]
        )
        
        # Calculate trend strength
        trend_strength = abs(forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[0]) / original_stats['std']
        
        # Calculate seasonality metrics if available
        seasonality_metrics = {}
        try:
            components = self.model.plot_components(forecast_results['forecast'])
            if 'yearly' in components:
                seasonality_metrics['yearly'] = components['yearly'].std() / original_stats['std']
            if 'weekly' in components:
                seasonality_metrics['weekly'] = components['weekly'].std() / original_stats['std']
            if 'daily' in components:
                seasonality_metrics['daily'] = components['daily'].std() / original_stats['std']
        except:
            pass
        
        # Calculate changepoints
        significant_changes = []
        try:
            changepoints = self.model.changepoints
            change_indices = [forecast['ds'][forecast['ds'] == cp].index[0] for cp in changepoints]
            for idx in change_indices:
                if idx + 1 < len(forecast):
                    change = (forecast['yhat'].iloc[idx + 1] - forecast['yhat'].iloc[idx]) / original_stats['std']
                    if abs(change) > 0.1:  # Threshold for significant changes
                        significant_changes.append({
                            'date': forecast['ds'].iloc[idx].strftime('%Y-%m-%d'),
                            'change': change
                        })
        except:
            pass
        
        return {
            'trend': {
                'direction': trend_direction,
                'strength': trend_strength,
                'significant_changes': significant_changes
            },
            'statistics': {
                'original': original_stats,
                'forecast': forecast_stats
            },
            'uncertainty': {
                'average_width': uncertainty_width,
                'growth_ratio': uncertainty_growth
            },
            'seasonality': seasonality_metrics
        }

class DecisionSupportSystem:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        
        # Initialize Gemini
        try:
            if 'GOOGLE_API_KEY' in st.secrets:
                genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
                self.llm = genai.GenerativeModel('gemini-pro')
            else:
                st.warning("Google API Key not found in secrets. AI insights will be disabled.")
                self.llm = None
        except Exception as e:
            st.error(f"Error initializing Gemini: {str(e)}")
            self.llm = None

    
    def preprocess_data(self, df):
        """Enhanced data preprocessing with automated feature detection"""
        try:
            # Handle missing values with more sophisticated imputation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            # Numeric imputation
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
            
            # Categorical imputation
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            
            # Convert date columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        # Check if column contains dates
                        pd.to_datetime(df[col], errors='raise')
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            
            # Scale numeric features while preserving DataFrame structure
            if len(numeric_cols) > 0:
                df[numeric_cols] = pd.DataFrame(
                    self.scaler.fit_transform(df[numeric_cols]),
                    columns=numeric_cols,
                    index=df.index
                )
            
            return df
        except Exception as e:
            st.error(f"Error preprocessing data: {str(e)}")
            return None

    def perform_what_if_analysis(self, base_value, percentage_change, n_scenarios=1000):
        """Enhanced What-If analysis with comprehensive statistics and risk metrics"""
        try:
            # Calculate parameters for the normal distribution
            std_dev = abs(percentage_change / 100 * base_value / 2)
            
            # Generate scenarios using Monte Carlo simulation
            simulated_values = norm.rvs(loc=base_value, scale=std_dev, size=n_scenarios)
            
            # Calculate comprehensive statistics
            results = {
                'mean': np.mean(simulated_values),
                'median': np.median(simulated_values),
                'std_dev': np.std(simulated_values),
                'skewness': stats.skew(simulated_values),
                'kurtosis': stats.kurtosis(simulated_values),
                'var_95': np.percentile(simulated_values, 5),  # 95% Value at Risk
                'cvar_95': np.mean(simulated_values[simulated_values <= np.percentile(simulated_values, 5)]),  # Conditional VaR
                'percentiles': {
                    '1%': np.percentile(simulated_values, 1),
                    '5%': np.percentile(simulated_values, 5),
                    '25%': np.percentile(simulated_values, 25),
                    '75%': np.percentile(simulated_values, 75),
                    '95%': np.percentile(simulated_values, 95),
                    '99%': np.percentile(simulated_values, 99)
                },
                'simulated_values': simulated_values,
                'probability_loss': np.mean(simulated_values < base_value)
            }
            
            return results
        except Exception as e:
            st.error(f"Error performing What-If analysis: {str(e)}")
            return None

    def evaluate_sensitivity(self, params, baseline):
        """
        Evaluate the sensitivity of output to input parameter changes.
        
        Args:
            params (dict): Dictionary of parameter values
            baseline (float): Baseline value for comparison
            
        Returns:
            float: Sensitivity score
        """
        try:
            # Calculate weighted impact of parameter changes
            impact = 0
            for param, value in params.items():
                if param in self.data.columns:
                    # Calculate correlation-weighted impact
                    correlation = abs(self.data[param].corr(self.data[self.target_col]))
                    param_impact = (value - self.data[param].mean()) / self.data[param].std()
                    impact += correlation * param_impact
            
            # Normalize impact relative to baseline
            relative_impact = impact / baseline if baseline != 0 else impact
            
            return relative_impact
            
        except Exception as e:
            st.error(f"Error in sensitivity evaluation: {str(e)}")
            return 0.0

    def perform_sensitivity_analysis(self, data, target_col, input_cols):
        """Enhanced sensitivity analysis with multiple methods"""
        try:
            # Store target column for use in evaluate_sensitivity
            self.target_col = target_col
            self.data = data
            
            # Define the problem for Sobol analysis
            problem = {
                'num_vars': len(input_cols),
                'names': input_cols,
                'bounds': [[data[col].min(), data[col].max()] for col in input_cols]
            }
            
            # Generate samples using Saltelli's method
            param_values = saltelli.sample(problem, 1024)
            
            # Calculate outputs for each sample
            Y = np.zeros([param_values.shape[0]])
            baseline = data[target_col].mean()
            
            for i, X in enumerate(param_values):
                sample = dict(zip(input_cols, X))
                Y[i] = self.evaluate_sensitivity(sample, baseline)
            
            # Perform Sobol analysis
            Si = sobol.analyze(problem, Y)
            
            # Calculate additional sensitivity metrics
            correlations = {col: np.corrcoef(data[col], data[target_col])[0,1] 
                          for col in input_cols}
            
            # Calculate elasticities
            elasticities = {}
            for col in input_cols:
                delta_x = data[col].std()
                delta_y = data[target_col].std()
                if delta_x != 0 and delta_y != 0:
                    elasticities[col] = (delta_y / data[target_col].mean()) / (delta_x / data[col].mean())
            
            results = {
                'first_order': dict(zip(input_cols, Si['S1'])),
                'total_order': dict(zip(input_cols, Si['ST'])),
                'second_order': Si['S2'],
                'confidence': Si['S1_conf'],
                'correlations': correlations,
                'elasticities': elasticities
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error performing sensitivity analysis: {str(e)}")
            return None

    def perform_scenario_analysis(self, base_scenario, scenarios, monte_carlo=True):
        """Enhanced scenario analysis with risk metrics and correlations"""
        try:
            results = {}
            if monte_carlo:
                n_sims = 1000
                correlations = np.random.uniform(-0.5, 0.5, (len(base_scenario), len(base_scenario)))
                np.fill_diagonal(correlations, 1)
                
                for scenario_name, adjustments in scenarios.items():
                    scenario_results = []
                    
                    # Generate correlated random noise
                    for _ in range(n_sims):
                        noise = np.random.multivariate_normal(
                            mean=np.zeros(len(base_scenario)),
                            cov=correlations,
                            size=1
                        )[0]
                        
                        modified_scenario = base_scenario.copy()
                        for i, (variable, adjustment) in enumerate(adjustments.items()):
                            # Apply adjustment with correlated noise
                            modified_scenario[variable] *= (1 + adjustment + noise[i] * 0.1)
                        scenario_results.append(modified_scenario)
                    
                    # Calculate comprehensive statistics
                    results[scenario_name] = {
                        'mean': {k: np.mean([s[k] for s in scenario_results]) for k in base_scenario.keys()},
                        'std': {k: np.std([s[k] for s in scenario_results]) for k in base_scenario.keys()},
                        'var_95': {k: np.percentile([s[k] for s in scenario_results], 5) for k in base_scenario.keys()},
                        'cvar_95': {k: np.mean([s[k] for s in scenario_results if s[k] <= np.percentile([s[k] for s in scenario_results], 5)])
                                  for k in base_scenario.keys()},
                        'correlations': {k: {j: np.corrcoef([s[k] for s in scenario_results], 
                                                          [s[j] for s in scenario_results])[0,1]
                                           for j in base_scenario.keys()}
                                       for k in base_scenario.keys()},
                        'percentiles': {
                            k: {
                                '5%': np.percentile([s[k] for s in scenario_results], 5),
                                '25%': np.percentile([s[k] for s in scenario_results], 25),
                                '75%': np.percentile([s[k] for s in scenario_results], 75),
                                '95%': np.percentile([s[k] for s in scenario_results], 95)
                            } for k in base_scenario.keys()
                        }
                    }
            else:
                for scenario_name, adjustments in scenarios.items():
                    modified_scenario = base_scenario.copy()
                    for variable, adjustment in adjustments.items():
                        modified_scenario[variable] *= (1 + adjustment)
                    results[scenario_name] = {'mean': modified_scenario}
            
            return results
        except Exception as e:
            st.error(f"Error performing scenario analysis: {str(e)}")
            return None

    def optimize_goal(self, objective_func, bounds, n_trials=100):
        """Enhanced goal optimization with multi-objective support and advanced algorithms"""
        try:
            # Create study with multiple optimization directions
            study = optuna.create_study(
                directions=["maximize"],
                sampler=optuna.samplers.TPESampler(multivariate=True)
            )
            
            # Define objective with regularization and constraints
            def objective(trial):
                params = {
                    key: trial.suggest_float(key, bound[0], bound[1])
                    for key, bound in bounds.items()
                }
                
                # Add regularization term
                regularization = sum(value ** 2 for value in params.values()) * 0.01
                
                # Calculate main objective
                main_objective = objective_func(params)
                
                # Apply soft constraints
                constraint_violation = 0
                for key, bound in bounds.items():
                    if params[key] < bound[0] or params[key] > bound[1]:
                        constraint_violation += 1000
                
                return main_objective - regularization - constraint_violation
            
            # Optimize with callbacks for tracking
            study.optimize(objective, n_trials=n_trials, callbacks=[])
            
            # Analyze optimization results
            results = {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'all_trials': [{
                    'params': trial.params,
                    'value': trial.value,
                    'datetime_start': trial.datetime_start,
                    'datetime_complete': trial.datetime_complete
                } for trial in study.trials],
                'importance': optuna.importance.get_param_importances(study),
                'pareto_front': study.best_trials,
                'optimization_history': {
                    'values': [trial.value for trial in study.trials],
                    'params': [trial.params for trial in study.trials]
                }
            }
            
            return results
        except Exception as e:
            st.error(f"Error performing goal optimization: {str(e)}")
            return None

    def get_ai_insights(self, prompt, context=None):
        """Enhanced AI insights with structured output and error handling"""
        try:
            if self.llm is not None:
                # Create structured prompt
                structured_prompt = f"""
                Context: {context if context else 'No additional context provided'}
                
                Analysis Request: {prompt}
                
                Please provide insights in the following structure:
                1. Key Findings
                2. Trends and Patterns
                3. Potential Risks
                4. Recommendations
                5. Additional Considerations
                """
                
                response = self.llm.generate_content(structured_prompt)
                return response.text
            return "AI insights not available (API key not configured)"
        except Exception as e:
            return f"Error generating AI insights: {str(e)}"
        

def main():
    st.set_page_config(page_title="Advanced Decision Support System", layout="wide")
    
    # Initialize DSS
    if 'dss' not in st.session_state:
        st.session_state['dss'] = DecisionSupportSystem()
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tomorrow:wght@400;500;600&display=swap');
    
    /* Main theme colors */
    :root {
        --background-color: #1E201E;
        --text-color: #76ABAE;
        --accent-color: #EEEEEE;
        --secondary-color: #31363F;
        --error-color: #CF6679;
    }
    
    /* Override Streamlit's default styles */
    .stApp {
        background-color: var(--background-color);
    }
    
    .stMarkdown, .stText, p, span {
        color: var(--text-color) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--accent-color) !important;
        font-family: 'Tomorrow', sans-serif;
    }
    
    /* Buttons */
    .stButton > button:hover {
        background-color: var(--accent-color);
        color: black;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
        font-family: 'Tomorrow', sans-serif;
    }
    
    .stButton > button {
        background-color: var(--secondary-color);
        color: black;
    }
    
    /* Additional styles remain the same as in original CSS */
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Data Upload & Overview", "Forecasting", "What-If Analysis", 
         "Sensitivity Analysis", "Scenario Analysis", "Goal Setting"]
    )
    
    # Main title
    st.title("DSSential: The AI Chief")
    st.subheader("Advanced Decision Support System for Enterprises with LLM Enhanced Efficiency")
    
    if page == "Data Upload & Overview":
        st.header("Data Upload & Overview")
        
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        
        if uploaded_file is not None:
            with st.spinner("Processing your data..."):
                try:
                    # Read data with multiple encoding attempts
                    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                    for encoding in encodings:
                        try:
                            data = pd.read_csv(uploaded_file, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    # Process and store data
                    processed_data = st.session_state['dss'].preprocess_data(data)
                    if processed_data is not None:
                        st.session_state['data'] = processed_data
                        
                        # Display data overview
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Dataset Information")
                            st.write(f"Number of rows: {data.shape[0]}")
                            st.write(f"Number of columns: {data.shape[1]}")
                            st.write("Columns:", list(data.columns))
                        
                        with col2:
                            st.subheader("Data Preview")
                            st.write(data.head())
                        
                        # Data quality metrics
                        st.subheader("Data Quality Analysis")
                        quality_metrics = {
                            "Missing Values": data.isnull().sum().to_dict(),
                            "Data Types": data.dtypes.to_dict(),
                            "Unique Values": data.nunique().to_dict()
                        }
                        st.json(quality_metrics)
                        
                        # Basic visualizations
                        st.subheader("Data Visualizations")
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        
                        if len(numeric_cols) > 0:
                            # Correlation heatmap
                            corr = data[numeric_cols].corr()
                            fig = px.imshow(corr, title="Correlation Heatmap")
                            st.plotly_chart(fig)
                            
                            # Distribution plots
                            selected_col = st.selectbox("Select column for distribution plot", numeric_cols)
                            fig = px.histogram(data, x=selected_col, title=f"Distribution of {selected_col}")
                            st.plotly_chart(fig)
                        
                        # Get AI insights
                        st.subheader("AI Insights")
                        data_description = f"""
                        Dataset Overview:
                        - Shape: {data.shape}
                        - Columns: {', '.join(data.columns)}
                        - Numeric columns: {', '.join(numeric_cols)}
                        - Data types: {data.dtypes.to_dict()}
                        """
                        prompt = "Analyze this dataset and provide key insights about the data quality, patterns, and potential analysis opportunities."
                        ai_insights = st.session_state['dss'].get_ai_insights(prompt, data_description)
                        st.write(ai_insights)
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    elif page == "Forecasting" and 'data' in st.session_state:
        st.header("Time Series Forecasting")
        
        # Model parameters
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox(
                "Select target column for forecasting",
                st.session_state['data'].select_dtypes(include=[np.number]).columns
            )
            sequence_length = st.slider("Sequence Length", 5, 30, 10)
        
        with col2:
            forecast_steps = st.slider("Forecast Steps", 1, 100, 30)
            confidence_interval = st.slider("Confidence Interval", 0.8, 0.99, 0.95)
        
        # In the DSS class initialization
        st.session_state['dss'].time_series_model = TimeSeriesModel()

        # In the Forecasting page
        forecast_steps = st.slider("Near-term Forecast Steps", 1, 100, 30)
        future_periods = st.slider("Long-term Forecast Days", 30, 730, 365)

        if st.button("Generate Forecast"):
            with st.spinner("Training model and generating forecast..."):
                # Train model
                success = st.session_state['dss'].time_series_model.train(
                    st.session_state['data'], target_col
                )
                
                if success:
                    # Generate forecast with both near-term and future predictions
                    forecast_results = st.session_state['dss'].time_series_model.forecast(
                        steps=forecast_steps,
                        confidence_interval=confidence_interval,
                        future_periods=future_periods
                    )
                    

                    if forecast_results is not None:
                        # Plot results
                        fig = st.session_state['dss'].time_series_model.create_forecast_plot(
                            st.session_state['data'],
                            forecast_results,
                            target_col
                        )
                        st.plotly_chart(fig)
                        
                        # Get AI insights
                        st.subheader("AI Analysis")
                        forecast_insights = st.session_state['dss'].time_series_model.get_forecast_insights(
                            st.session_state['data'],
                            forecast_results,
                            target_col
                        )
                        
                        prompt = f"""
                        Analyze this forecast for {target_col} with the following insights:
                        
                        Trend Analysis:
                        - Direction: {forecast_insights['trend']['direction']}
                        - Strength: {forecast_insights['trend']['strength']:.2f} standard deviations
                        - Significant changes detected: {len(forecast_insights['trend']['significant_changes'])}
                        
                        Statistical Summary:
                        - Original data mean: {forecast_insights['statistics']['original']['mean']:.2f}
                        - Forecast mean: {forecast_insights['statistics']['forecast']['mean']:.2f}
                        - Forecast range: {forecast_insights['statistics']['forecast']['min']:.2f} to {forecast_insights['statistics']['forecast']['max']:.2f}
                        
                        Uncertainty Analysis:
                        - Average prediction interval width: {forecast_insights['uncertainty']['average_width']:.2f}
                        - Uncertainty growth ratio: {forecast_insights['uncertainty']['growth_ratio']:.2f}
                        
                        Seasonality Pattern Strength:
                        {json.dumps(forecast_insights['seasonality'], indent=2)}
                        
                        Please provide:
                        1. A detailed analysis of the overall trend and its significance
                        2. An assessment of the forecast reliability based on uncertainty metrics
                        3. Key observations about seasonal patterns and significant changes
                        4. Specific recommendations for actions based on these forecasting results
                        5. Potential risks and areas that need monitoring
                        """
                        
                        ai_analysis = st.session_state['dss'].get_ai_insights(prompt)
                        st.write(ai_analysis)
                        
                        # Display additional forecast components if available
                        if 'components' in forecast_results:
                            st.subheader("Forecast Components")
                            st.pyplot(forecast_results['components'])
                    
    
    elif page == "What-If Analysis" and 'data' in st.session_state:
        st.header("What-If Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox(
                "Select target variable",
                st.session_state['data'].select_dtypes(include=[np.number]).columns
            )
            base_value = st.session_state['data'][target_col].mean()
            st.write(f"Base value (mean): {base_value:.2f}")
        
        with col2:
            percentage_change = st.slider("Percentage change to analyze", -100, 100, 0)
            n_scenarios = st.slider("Number of scenarios", 100, 10000, 1000)
        
        if st.button("Run What-If Analysis"):
            with st.spinner("Performing What-If Analysis..."):
                results = st.session_state['dss'].perform_what_if_analysis(
                    base_value, percentage_change, n_scenarios
                )
                
                if results is not None:
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Summary Statistics")
                        metrics = {
                            "Mean": results['mean'],
                            "Median": results['median'],
                            "Standard Deviation": results['std_dev']
                        }
                        for name, value in metrics.items():
                            st.metric(name, f"{value:.2f}")
                    
                    with col2:
                        st.subheader("Percentiles")
                        for p, value in results['percentiles'].items():
                            st.metric(p, f"{value:.2f}")
                    
                    # Plot distribution
                    fig = px.histogram(
                        x=results['simulated_values'],
                        title="Distribution of Simulated Outcomes",
                        labels={'x': 'Value', 'y': 'Count'}
                    )
                    st.plotly_chart(fig)
                    
                    # Get AI insights
                    prompt = f"""
                    Analyze these What-If results:
                    - Base value: {base_value}
                    - Percentage change: {percentage_change}%
                    - Results: {str(results)}
                    
                    Provide insights about:
                    1. The impact of the change
                    2. Risk assessment
                    3. Recommendations
                    """
                    ai_analysis = st.session_state['dss'].get_ai_insights(prompt)
                    st.subheader("AI Analysis")
                    st.write(ai_analysis)
    
    elif page == "Sensitivity Analysis" and 'data' in st.session_state:
        st.header("Sensitivity Analysis")
        
        # Parameter selection
        target_col = st.selectbox(
            "Select target variable",
            st.session_state['data'].select_dtypes(include=[np.number]).columns
        )
        
        input_cols = st.multiselect(
            "Select input variables",
            [col for col in st.session_state['data'].select_dtypes(include=[np.number]).columns 
             if col != target_col]
        )
        
        if input_cols and st.button("Run Sensitivity Analysis"):
            with st.spinner("Performing Sensitivity Analysis..."):
                results = st.session_state['dss'].perform_sensitivity_analysis(
                    st.session_state['data'], target_col, input_cols
                )
                
                if results is not None:
                    # Display results
                    st.subheader("First Order Sensitivity Indices")
                    fig = px.bar(
                        x=list(results['first_order'].keys()),
                        y=list(results['first_order'].values()),
                        title="First Order Sensitivity Indices",
                        labels={'x': 'Variable', 'y': 'Sensitivity Index'}
                    )
                    st.plotly_chart(fig)
                    
                    st.subheader("Total Order Sensitivity Indices")
                    fig = px.bar(
                        x=list(results['total_order'].keys()),
                        y=list(results['total_order'].values()),
                        title="Total Order Sensitivity Indices",
                        labels={'x': 'Variable', 'y': 'Sensitivity Index'}
                    )
                    st.plotly_chart(fig)
                    
                    # Get AI insights
                    prompt = f"""
                    Analyze these sensitivity results:
                    - Target variable: {target_col}
                    - Input variables: {', '.join(input_cols)}
                    - Results: {str(results)}
                    
                    Provide insights about:
                    1. Most influential variables
                    2. Variable interactions
                    3. Recommendations for decision-making
                    """
                    ai_analysis = st.session_state['dss'].get_ai_insights(prompt)
                    st.subheader("AI Analysis")
                    st.write(ai_analysis)
    
    elif page == "Scenario Analysis" and 'data' in st.session_state:
        st.header("Scenario Analysis")
        
        # Base scenario
        st.subheader("Base Scenario")
        numeric_cols = st.session_state['data'].select_dtypes(include=[np.number]).columns
        base_scenario = {}
        for col in numeric_cols:
            base_scenario[col] = st.session_state['data'][col].mean()
        
        # Scenario definition
        st.subheader("Define Scenarios")
        n_scenarios = st.number_input("Number of scenarios", 1, 5, 3)
        
        scenarios = {}
        for i in range(n_scenarios):
            st.write(f"Scenario {i+1}")
            scenario_name = st.text_input(f"Scenario {i+1} name", f"Scenario {i+1}")
            adjustments = {}
            
            for col in numeric_cols:
                adjustment = st.slider(
                    f"Adjustment for {col} (%)",
                    -100, 100, 0,
                    key=f"scenario_{i}_{col}"
                )
                adjustments[col] = adjustment / 100
            
            scenarios[scenario_name] = adjustments
        
        use_monte_carlo = st.checkbox("Use Monte Carlo simulation", value=True)
        
        if st.button("Run Scenario Analysis"):
            with st.spinner("Performing Scenario Analysis..."):
                results = st.session_state['dss'].perform_scenario_analysis(
                    base_scenario, scenarios, monte_carlo=use_monte_carlo
                )
                
                if results is not None:
                    # Display results
                    for scenario_name, scenario_results in results.items():
                        st.subheader(scenario_name)
                        
                        # Display mean values
                        fig = px.bar(
                            x=list(scenario_results['mean'].keys()),
                            y=list(scenario_results['mean'].values()),
                            title=f"{scenario_name} - Mean Values"
                        )
                        st.plotly_chart(fig)
                        
                        if use_monte_carlo:
                            # Display uncertainty
                            st.write("Uncertainty Analysis")
                            for var in scenario_results['percentiles'].keys():
                                st.metric(
                                    var,
                                    f"{scenario_results['mean'][var]:.2f}",
                                    f"±{scenario_results['std'][var]:.2f}"
                                )
                    
                    # Compare scenarios
                    st.subheader("Scenario Comparison")
                    comparison_data = []
                    for scenario_name, scenario_results in results.items():
                        for var, value in scenario_results['mean'].items():
                            comparison_data.append({
                                'Scenario': scenario_name,
                                'Variable': var,
                                'Value': value
                            })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    fig = px.bar(
                        comparison_df,
                        x='Variable',
                        y='Value',
                        color='Scenario',
                        barmode='group',
                        title='Scenario Comparison'
                    )
                    st.plotly_chart(fig)
                    
                    # Get AI insights
                    prompt = f"""
                    Analyze these scenario analysis results:
                    Base scenario: {str(base_scenario)}
                    Scenarios: {str(scenarios)}
                    Results: {str(results)}
                    
                    Provide insights about:
                    1. Key differences between scenarios
                    2. Risk and opportunity assessment
                    3. Recommended scenario selection
                    4. Implementation considerations
                    """
                    ai_analysis = st.session_state['dss'].get_ai_insights(prompt)
                    st.subheader("AI Analysis")
                    st.write(ai_analysis)
    
    elif page == "Goal Setting" and 'data' in st.session_state:
        st.header("Goal Setting Optimization")
        
        # Select target and variables
        target_col = st.selectbox(
            "Select target variable to optimize",
            st.session_state['data'].select_dtypes(include=[np.number]).columns
        )
        
        optimization_vars = st.multiselect(
            "Select variables for optimization",
            [col for col in st.session_state['data'].select_dtypes(include=[np.number]).columns 
             if col != target_col]
        )
        
        if optimization_vars:
            # Define bounds
            st.subheader("Optimization Bounds")
            bounds = {}
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Minimum Values")
                min_values = {
                    col: st.number_input(
                        f"Min {col}",
                        value=float(st.session_state['data'][col].min()),
                        key=f"min_{col}"
                    )
                    for col in optimization_vars
                }
            
            with col2:
                st.write("Maximum Values")
                max_values = {
                    col: st.number_input(
                        f"Max {col}",
                        value=float(st.session_state['data'][col].max()),
                        key=f"max_{col}"
                    )
                    for col in optimization_vars
                }
            
            for col in optimization_vars:
                bounds[col] = (min_values[col], max_values[col])
            
            n_trials = st.slider("Number of optimization trials", 10, 1000, 100)
            
            if st.button("Optimize Goals"):
                with st.spinner("Performing Goal Optimization..."):
                    # Define objective function
                    def objective_func(params):
                        # Simple objective function based on correlation with target
                        return sum(
                            value * st.session_state['data'][key].corr(
                                st.session_state['data'][target_col]
                            )
                            for key, value in params.items()
                        )
                    
                    results = st.session_state['dss'].optimize_goal(
                        objective_func, bounds, n_trials=n_trials
                    )
                    
                    if results is not None:
                        # Display results
                        st.subheader("Optimization Results")
                        
                        # Best parameters
                        st.write("Best Parameters:")
                        for param, value in results['best_params'].items():
                            st.metric(param, f"{value:.2f}")
                        
                        st.write(f"Best Objective Value: {results['best_value']:.2f}")
                        
                        # Visualization of optimization history
                        trial_data = pd.DataFrame(results['all_trials'])
                        fig = px.line(
                            x=range(len(trial_data)),
                            y=trial_data['value'],
                            title='Optimization History'
                        )
                        st.plotly_chart(fig)
                        
                        # Parameter importance
                        st.subheader("Parameter Importance")
                        importance_data = []
                        for param in results['best_params'].keys():
                            values = [trial['params'][param] for trial in results['all_trials']]
                            correlation = np.corrcoef(values, [t['value'] for t in results['all_trials']])[0, 1]
                            importance_data.append({
                                'Parameter': param,
                                'Importance': abs(correlation)
                            })
                        
                        importance_df = pd.DataFrame(importance_data)
                        fig = px.bar(
                            importance_df,
                            x='Parameter',
                            y='Importance',
                            title='Parameter Importance'
                        )
                        st.plotly_chart(fig)
                        
                        # Get AI insights
                        prompt = f"""
                        Analyze these optimization results:
                        Target variable: {target_col}
                        Optimization variables: {', '.join(optimization_vars)}
                        Best parameters: {str(results['best_params'])}
                        Best value: {results['best_value']}
                        
                        Provide insights about:
                        1. Optimal parameter settings
                        2. Expected impact on target variable
                        3. Implementation recommendations
                        4. Potential risks and considerations
                        """
                        ai_analysis = st.session_state['dss'].get_ai_insights(prompt)
                        st.subheader("AI Analysis")
                        st.write(ai_analysis)
    
    else:
        st.info("""
        ### Getting Started
        1. Upload your dataset using the 'Data Upload & Overview' page
        2. Explore different analysis types:
            - Forecasting: Predict future values using BiLSTM
            - What-If Analysis: Simulate different scenarios
            - Sensitivity Analysis: Understand variable importance
            - Scenario Analysis: Compare different business scenarios
            - Goal Setting: Optimize for specific targets
        
        ### Dataset Requirements
        - File format: CSV
        - Must contain at least one numeric column
        - For time series analysis, data should be in chronological order
        """)

if __name__ == "__main__":
    main()