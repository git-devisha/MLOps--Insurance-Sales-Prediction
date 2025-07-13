# import logging
# import yaml
# import mlflow
# import mlflow.sklearn
# from steps.ingest import ingestion
# from steps.clean import cleaner
# from steps.train import Trainer
# from steps.predict import predictor
# from sklearn.metrics import classification_report

# logging.basicConfig(level = logging.INFO, format= '%(asctime)s: %(levelname)s : %(message)s')

# def main():
#     # load data
#     Ingestion= ingestion()
#     train, test= Ingestion.load_data()
#     logging.info("data ingestion completed")

#     # data cleaning
#     Cleaner= cleaner()
#     train_data = Cleaner.clean_data(train)
#     test_data = Cleaner.clean_data(test)
#     logging.info("data cleaning completed")

#     # prepare and train the model
#     trainer= Trainer()
#     x_train, y_train = trainer.feature_target_separator(train_data)
#     trainer.train_model(x_train, y_train)
#     trainer.save_model()
#     logging.info("Model trained successfully")

#     # evaluate model
#     Predictor= predictor()
#     x_test, y_test = Predictor.feature_target_separator(x_test, y_test)
#     logging.info("model evaluation completed")

#     # Print evaluation results
#     from sklearn.metrics import accuracy_score, roc_auc_score
#     y_pred = trainer.model.predict(x_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     try:
#         roc_auc = roc_auc_score(y_test, y_pred)
#     except Exception:
#         roc_auc = float('nan')
#     class_report = classification_report(y_test, y_pred)
#     print("\n============= Model Evaluation Results ==============")
#     print(f"Model: {trainer.model_name}")
#     print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc:.4f}")
#     print(f"\n{class_report}")
#     print("=====================================================\n")

# def train_with_mlflow():
#     with open('config.yml', 'r') as file:
#         config = yaml.safe_load(file)
#     mlflow.set_experiment("model training experiment")

#     with mlflow.start_run() as run:
#         # load data
#         Ingestion= ingestion()
#     train, test= Ingestion.load_data()
#     logging.info("data ingestion completed")

#     # data cleaning
#     Cleaner= cleaner()
#     train_data = Cleaner.clean_data(train)
#     test_data = Cleaner.clean_data(test)
#     logging.info("data cleaning completed")

#     # prepare and train the model
#     trainer= Trainer()
#     x_train, y_train = trainer.feature_target_separator(train_data)
#     trainer.train_model(x_train, y_train)
#     trainer.save_model()
#     logging.info("Model trained successfully")

#     # evaluate model
#     Predictor= predictor()
#     x_test, y_test = Predictor.feature_target_separator(test_data)
#     y_pred = trainer.model.predict(x_test)
#     class_report = classification_report(y_test, y_pred)
#     logging.info("model evaluation completed")

#     # log metrics
#     model_params = config['model']['params']
#     model_uri = f"runs:/{run.info.run_id}/model"
#     # mlflow.register_model(model_params, model_name)  # This line may need correction based on your mlflow usage
#     logging.info("mlflow tracking completed")

#     # Print evaluation results
#     print("\n============= Model Evaluation Results ==============")
#     print(f"Model: {trainer.model_name}")
#     # print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")  # Uncomment and define accuracy/roc_auc_score if needed
#     print(f"\n{class_report}")
#     print("=====================================================\n")
        
# if __name__ == "__main__":
#     # main()
#     train_with_mlflow()


import logging
import yaml
import mlflow
import mlflow.sklearn
from steps.ingest import ingestion
from steps.clean import cleaner
from steps.train import Trainer
from steps.predict import Predictor
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s : %(message)s')

def run_pipeline(log_mlflow=False, config=None):
    # Data Ingestion
    ingestion_step = ingestion()
    train, test = ingestion_step.load_data()
    logging.info("Data ingestion completed successfully")

    # Data Cleaning
    cleaning_step = cleaner()
    train_data = cleaning_step.clean_data(train)
    test_data = cleaning_step.clean_data(test)
    logging.info("Data cleaning completed successfully")

    # Model Training
    # trainer = Trainer(config['model']['params'] if config else None)
    trainer = Trainer(config['model'])
    x_train, y_train = trainer.feature_target_separator(train_data)
    trainer.train_model(x_train, y_train)
    trainer.save_model()
    logging.info("Model training completed successfully")

    # Model Evaluation
    predictor_step = Predictor()
    x_test, y_test = predictor_step.feature_target_separator(test_data)
    y_pred = trainer.model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_pred)
    except Exception as e:
        logging.warning(f"ROC AUC calculation failed: {e}")
        roc_auc = float('nan')

    class_report = classification_report(y_test, y_pred)
    
    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {trainer.model_name}")
    print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc:.4f}")
    print(f"\n{class_report}")
    print("=====================================================\n")

    # MLflow Logging
    if log_mlflow:
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(trainer.model, "model")
        logging.info("Model and metrics logged to MLflow")

def main():
    run_pipeline(log_mlflow=False)

def train_with_mlflow():
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment("insurance-sales-prediction")
    with mlflow.start_run():
        run_pipeline(log_mlflow=True, config=config)

if __name__ == "__main__":
    # main()  # For non-MLflow run
    train_with_mlflow()  # For MLflow run
