import os
import joblib
import logging
import yaml
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class Predictor:
    def __init__(self, config_path='insurance_sale/config.yml'):
        self.config = self.load_config(config_path)
        self.model_name = self.config['model']['name']
        self.model_path = self.config['model']['store_path']
        self.pipeline = self.load_model()

    def load_config(self, path):
        with open(path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def load_model(self):
        model_file = f"{self.model_name}.pkl"
        model_file_path = os.path.join(self.model_path, model_file)
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found at {model_file_path}")
        logging.info(f"Loading model from {model_file_path}")
        return joblib.load(model_file_path)

    def feature_target_separator(self, data):
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return x, y

    def evaluate_model(self, x_test, y_test):
        y_pred = self.pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"ROC AUC: {roc_auc:.4f}")
        logging.info("Classification Report:")
        logging.info(f"\n{class_report}")

        return accuracy, roc_auc, class_report, y_pred
    
if __name__ == "__main__":
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s : %(message)s')
    
    predictor = Predictor()

    # Load test data
    test_path = predictor.config['data']['test_path']
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
    
    test_data = pd.read_csv(test_path)
    x_test, y_test = predictor.feature_target_separator(test_data)

    # Evaluate model
    accuracy, roc_auc, class_report, _ = predictor.evaluate_model(x_test, y_test)

    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {predictor.model_name}")
    print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc:.4f}")
    print(f"\n{class_report}")
    print("=====================================================\n")

