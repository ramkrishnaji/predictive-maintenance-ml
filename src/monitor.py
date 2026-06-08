import os
import pandas as pd
import numpy as np

class DriftMonitor:
    def __init__(self, reference_path="data/reference_data.csv", report_path="assets/drift_report.html"):
        """
        Initializes the Drift Monitor.
        Args:
            reference_path: Path to the reference training dataset.
            report_path: Path where the HTML report will be saved.
        """
        self.reference_path = reference_path
        self.report_path = report_path
        
        # NASA CMAPSS features (excluding constant/low variance sensors)
        self.sensor_cols = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
        self.setting_cols = ['setting_1', 'setting_2', 'setting_3']
        self.features = self.sensor_cols + self.setting_cols

    def load_reference(self):
        """Loads and returns the reference dataset."""
        if not os.path.exists(self.reference_path):
            raise FileNotFoundError(f"Reference data file not found at {self.reference_path}. Please run training first.")
        
        df = pd.read_csv(self.reference_path)
        return df

    def detect_drift(self, current_df, current_predictions=None, reference_predictions=None, threshold=0.3):
        """
        Runs Evidently AI Data Drift preset on features and predictions.
        """
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        ref_df = self.load_reference()
        
        # Ensure current_df has all required columns
        missing_cols = [c for c in self.features if c not in current_df.columns]
        if missing_cols:
            raise ValueError(f"Current dataset is missing required feature columns: {missing_cols}")
            
        # Select features to compare
        ref_data = ref_df[self.features].copy()
        cur_data = current_df[self.features].copy()
        
        # Include predictions if provided
        if current_predictions is not None and reference_predictions is not None:
            ref_data['predicted_RUL'] = reference_predictions
            cur_data['predicted_RUL'] = current_predictions
            
        # Initialize and run Evidently Report
        report = Report(metrics=[
            DataDriftPreset()
        ])
        
        print(f"Running Evidently drift report comparing {len(ref_data)} reference rows vs {len(cur_data)} current rows.")
        report.run(reference_data=ref_data, current_data=cur_data)
        
        # Ensure save folder exists and save report
        os.makedirs(os.path.dirname(os.path.abspath(self.report_path)), exist_ok=True)
        report.save_html(self.report_path)
        print(f"Evidently HTML report saved to {self.report_path}")
        
        # Parse result dictionary
        result_dict = report.as_dict()
        metrics = result_dict['metrics'][0]['result']
        
        number_of_features = metrics['number_of_columns']
        number_of_drifted_features = metrics['number_of_drifted_columns']
        drift_share = metrics['share_of_drifted_columns']
        
        # Determine if warning should be triggered
        drift_detected = drift_share >= threshold
        
        # Build individual column drift report
        column_drift = {}
        for col, col_metrics in metrics['drift_by_columns'].items():
            column_drift[col] = {
                "drift_detected": bool(col_metrics['drift_detected']),
                "drift_score": round(float(col_metrics['drift_score']), 4),
                "metric_name": col_metrics['metric_name']
            }
            
        return {
            "drift_detected": bool(drift_detected),
            "drift_ratio": round(float(drift_share), 4),
            "number_of_drifted_features": int(number_of_drifted_features),
            "total_features_checked": int(number_of_features),
            "html_report_path": self.report_path,
            "column_drift": column_drift
        }

if __name__ == "__main__":
    # Test execution
    print("Testing DriftMonitor initialization...")
    try:
        monitor = DriftMonitor()
        ref = monitor.load_reference()
        print(f"Reference shape: {ref.shape}")
        
        # Create a synthetic "drifted" test set by adding noise
        drifted = ref.copy()
        drifted['s_2'] += np.random.normal(5.0, 1.0, size=len(drifted))
        
        # Test detection
        res = monitor.detect_drift(drifted)
        print("Test Complete. Results:")
        print(f"Drift Detected: {res['drift_detected']}")
        print(f"Drift Ratio: {res['drift_ratio']}")
        print(f"Drifted Columns Count: {res['number_of_drifted_features']}/{res['total_features_checked']}")
    except Exception as e:
        print(f"Dry run failed: {e}. (Normal if data/reference_data.csv hasn't been created yet)")
