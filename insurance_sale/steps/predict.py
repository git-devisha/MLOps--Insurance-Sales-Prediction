import os
import joblib
from sklearn.metrics import accuracy_score, classification_report,  roc_auc_score

class predictor:
    def __init__(self):
        self.model_path= self.load_config()['model']['store_path']
        self.pipeline = self.load_model()
    def load_config(self):
        import yaml
        with open('config.yaml', 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def load_model(self):
        model_file_path= os.path.join(self.model_path, 'model.pkl')
        return joblib.load(model_file_path)
    
    def feature_target_separator(self, data):
        x= data.iloc[:, :-1]
        y= data.iloc[:, -1]
        return x, y
    
    def evaluate_model(self, x_test, y_test):
        y_pred= self.pipeline.predict(x_test)
        accuracy= accuracy_score(y_test,  y_pred)
        class_report= classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        return accuracy, class_report, roc_auc