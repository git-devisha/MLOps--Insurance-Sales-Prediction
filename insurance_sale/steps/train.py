import os
import joblib
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

class Trainer:
    def __init__(self, model_config):
        self.model_name = model_config['name']
        self.model_params = model_config['params']
        self.model_path = model_config['store_path']
        
        # Handle 'null' in YAML config
        for key, val in self.model_params.items():
            if val is None or str(val).lower() == "null":
                self.model_params[key] = None

        self.pipeline = self.create_pipeline()
        logging.info(f"Initialized Trainer with model: {self.model_name}")

    def create_pipeline(self):
        preprocessor = ColumnTransformer(transformers=[
            ('minmax', MinMaxScaler(), ['AnnualPremium']),
            ('standardize', StandardScaler(), ['Age', 'RegionID']),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'PastAccident']),
        ])

        smote = SMOTE(sampling_strategy=1.0)

        model_map = {
            'RandomForestClassifier': RandomForestClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier
        }

        if self.model_name not in model_map:
            raise ValueError(f"Model {self.model_name} is not supported")

        model = model_map[self.model_name](**self.model_params)

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('smote', smote),
            ('model', model)
        ])

        logging.info(f"Pipeline created with model: {self.model_name}")
        return pipeline

    def feature_target_separator(self, data):
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return x, y

    def train_model(self, x_train, y_train):
        logging.info("Training started...")
        self.pipeline.fit(x_train, y_train)
        logging.info("Training completed successfully.")

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        # model_file = f"{self.model_name}.pkl"
        model_file_path = os.path.join(self.model_path, f'{self.model_name}.pkl')
        joblib.dump(self.pipeline, model_file_path)
        logging.info(f"Model saved at {model_file_path}")
