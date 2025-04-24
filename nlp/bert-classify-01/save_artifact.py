# Train the model
import mlflow

mlflow.set_tracking_uri("https://mlflow.yellowday.day")
mlflow.set_experiment("bert-classify-smsspam")
mlflow.enable_system_metrics_logging() # Logs CPU, RAM, GPU usage

mlflow.start_run(run_id="38f11b05a1514aae823e35ddee8d49b1")
mlflow.log_artifact("/home/yisheng/playground/nlp/bert-classify-smsspam/data/smsspam-train.csv")
mlflow.end_run()