import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelEvaluator:
    """
    Class for evaluating forecast models and tracking performance.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_history = []
    
    def calculate_metrics(self, actual, predicted):
        """
        Calculate various evaluation metrics for regression.
        
        Args:
            actual (numpy.ndarray): Actual values
            predicted (numpy.ndarray): Predicted values
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Ensure inputs are flattened arrays
        actual = actual.flatten()
        predicted = predicted.flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        
        # Mean Absolute Percentage Error (MAPE)
        # Handle zeros in actual values to avoid division by zero
        mask = actual != 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        else:
            mape = np.nan
        
        # Weighted Mean Absolute Percentage Error (WMAPE)
        wmape = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
        
        # Bias (Mean Error)
        bias = np.mean(predicted - actual)
        
        # Tracking Signal
        if np.sum(np.abs(actual - predicted)) != 0:
            tracking_signal = np.sum(predicted - actual) / np.sum(np.abs(predicted - actual))
        else:
            tracking_signal = 0
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'WMAPE': wmape,
            'R2': r2,
            'Bias': bias,
            'Tracking Signal': tracking_signal
        }
        
        # Store in history
        self.evaluation_history.append({
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics,
            'n_samples': len(actual)
        })
        
        return metrics
    
    def get_evaluation_history(self):
        """
        Get the evaluation history.
        
        Returns:
            pd.DataFrame: Evaluation history as a DataFrame
        """
        if not self.evaluation_history:
            return pd.DataFrame()
        
        # Flatten the history
        flat_history = []
        for entry in self.evaluation_history:
            history_entry = {
                'timestamp': entry['timestamp'],
                'n_samples': entry['n_samples']
            }
            
            # Add each metric
            for metric, value in entry['metrics'].items():
                history_entry[metric] = value
            
            flat_history.append(history_entry)
        
        # Convert to DataFrame
        history_df = pd.DataFrame(flat_history)
        
        return history_df
    
    def compare_models(self, actuals, predictions_list, model_names):
        """
        Compare multiple models side by side.
        
        Args:
            actuals (numpy.ndarray): Actual values
            predictions_list (list): List of prediction arrays from different models
            model_names (list): List of model names
            
        Returns:
            pd.DataFrame: Comparison of models
        """
        if len(predictions_list) != len(model_names):
            raise ValueError("Number of prediction arrays must match number of model names")
        
        # Calculate metrics for each model
        comparison = []
        
        for i, predictions in enumerate(predictions_list):
            metrics = self.calculate_metrics(actuals, predictions)
            metrics['Model'] = model_names[i]
            comparison.append(metrics)
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison)
        
        # Reorder columns to put Model first
        cols = comparison_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('Model')))
        comparison_df = comparison_df[cols]
        
        return comparison_df
    
    def residual_analysis(self, actual, predicted):
        """
        Perform residual analysis.
        
        Args:
            actual (numpy.ndarray): Actual values
            predicted (numpy.ndarray): Predicted values
            
        Returns:
            dict: Residual analysis results
        """
        # Flatten arrays
        actual = actual.flatten()
        predicted = predicted.flatten()
        
        # Calculate residuals
        residuals = actual - predicted
        
        # Calculate statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Test for normality of residuals (Shapiro-Wilk)
        from scipy import stats
        shapiro_test = None
        if len(residuals) >= 3 and len(residuals) <= 5000:  # Shapiro-Wilk works best for this range
            shapiro_test = stats.shapiro(residuals)
        
        # Test for autocorrelation (Durbin-Watson)
        # DW ranges from 0 to 4, with 2 meaning no autocorrelation
        # Closer to 0: positive autocorrelation
        # Closer to 4: negative autocorrelation
        dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
        
        # Test for heteroscedasticity
        # Simple test: correlation between absolute residuals and predicted values
        # A high correlation suggests heteroscedasticity
        abs_resid = np.abs(residuals)
        hetero_corr = np.corrcoef(predicted, abs_resid)[0, 1]
        
        analysis = {
            'Mean Residual': mean_residual,
            'Std Residual': std_residual,
            'Shapiro-Wilk Test': shapiro_test,
            'Durbin-Watson Statistic': dw_stat,
            'Heteroscedasticity Correlation': hetero_corr
        }
        
        return analysis
    
    def calculate_threshold_accuracy(self, actual, predicted, threshold=0.1):
        """
        Calculate threshold accuracy: percentage of predictions within threshold of actual values.
        
        Args:
            actual (numpy.ndarray): Actual values
            predicted (numpy.ndarray): Predicted values
            threshold (float): Threshold as a proportion (e.g., 0.1 = 10%)
            
        Returns:
            float: Threshold accuracy (0-1)
        """
        # Flatten arrays
        actual = actual.flatten()
        predicted = predicted.flatten()
        
        # Calculate absolute percentage errors
        abs_perc_errors = np.abs((actual - predicted) / np.maximum(1e-10, actual))
        
        # Calculate threshold accuracy
        threshold_accuracy = np.mean(abs_perc_errors <= threshold)
        
        return threshold_accuracy
