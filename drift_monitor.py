from scipy.stats import ks_2samp

class DriftMonitor:
    """Monitors statistical data drift and model performance drift."""
    
    def __init__(self, p_value_threshold=0.05, accuracy_drop_threshold=0.05):
        self.p_value_threshold = p_value_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold

    def detect_data_drift(self, reference_df, current_df, features):
        """
        Uses the Kolmogorov-Smirnov test to detect distribution changes.
        Returns a dictionary detailing which features have drifted.
        """
        drift_report = {}
        is_drifting = False
        
        for feature in features:
            stat, p_value = ks_2samp(reference_df[feature], current_df[feature])
            drift_detected = p_value < self.p_value_threshold
            
            if drift_detected:
                is_drifting = True
                
            drift_report[feature] = {
                'p_value': round(p_value, 4),
                'drift_detected': drift_detected
            }
            
        return {'dataset_drift': is_drifting, 'feature_details': drift_report}
