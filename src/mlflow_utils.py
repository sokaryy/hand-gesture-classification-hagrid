import os
import time
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

def setup_mlflow(experiment_name, tracking_uri = "http://localhost:5000"):
    """
    Setup MLflow tracking and set the experiment.
    
    Parameters:
    -----------
    experiment_name : str
        Name of the MLflow experiment
    tracking_uri : str
        URI of the MLflow tracking server (default: http://localhost:5000)
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow Active | Tracking URI: {tracking_uri} | Experiment: {experiment_name}")


def start_run(run_name=None):
    """
    Start a new MLflow run.
    
    Parameters:
    -----------
    run_name : str, optional
        Name for the current run
    
    Returns:
    --------
    run_id : str
        The ID of the started run
    """
    run = mlflow.start_run(run_name=run_name)
    print(f"Run started: {run.info.run_id}")
    return run.info.run_id


def log_parameters(params_dict):
    """
    Log a dictionary of parameters to the current run.
    
    Parameters:
    -----------
    params_dict : dict
        Dictionary of parameters to log
    """
    for key, value in params_dict.items():
        mlflow.log_param(key, value)
    print(f"Logged {len(params_dict)} parameters")


def log_metrics(y_true, y_pred, prefix=""):
    """
    Calculate and log classification metrics (accuracy, precision, recall, f1).
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    prefix : str, optional
        Prefix for metric names (e.g., 'val_' for validation metrics)
    
    Returns:
    --------
    metrics : dict
        Dictionary of calculated metrics
    """
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        f"{prefix}f1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    print(f"Logged metrics: {metrics}")
    return metrics

def log_confusion_matrix(y_true, y_pred, labels=None, figure_name="confusion_matrix"):
    """
    Create and log confusion matrix as an artifact.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        List of label names
    figure_name : str
        Name for the saved figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues')
    
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save locally and log
    artifact_path = f"./artifacts/{figure_name}.png"
    os.makedirs("./artifacts", exist_ok=True)
    plt.savefig(artifact_path, dpi=100, bbox_inches='tight')
    mlflow.log_artifact(artifact_path)
    
    plt.close()
    print(f"Logged confusion matrix: {artifact_path}")


def log_model_info(model,X,y ,model_name="model"):
    """
    Log a scikit-learn model to MLflow.
    
    Parameters:
    -----------
    model : sklearn model
        Trained scikit-learn model
    model_name : str
        Name for the model artifact
    """
    signature = mlflow.models.infer_signature(X, y)
    mlflow.sklearn.log_model(model,artifact_path= model_name,signature=signature,input_example=X.iloc[[0]])
    print(f"Logged model: {model_name}")


def log_artifact_file(file_path, artifact_path=None):
    """
    Log a local file as an artifact.
    
    Parameters:
    -----------
    file_path : str
        Path to the local file to log
    artifact_path : str, optional
        Artifact path in MLflow (if None, uses file name)
    """
    if os.path.exists(file_path):
        mlflow.log_artifact(file_path, artifact_path=artifact_path)
        print(f"Logged artifact: {file_path}")
    else:
        print(f"File not found: {file_path}")


def log_plot(figure, plot_name="plot"):
    """
    Save and log a matplotlib figure as an artifact.
    
    Parameters:
    -----------
    figure : matplotlib.figure.Figure
        Matplotlib figure object
    plot_name : str
        Name for the plot
    """
    artifact_path = f"./artifacts/{plot_name}.png"
    os.makedirs("./artifacts", exist_ok=True)
    figure.savefig(artifact_path, dpi=100, bbox_inches='tight')
    mlflow.log_artifact(artifact_path)
    plt.close(figure)
    print(f"Logged plot: {artifact_path}")


def log_tags(tags_dict):
    """
    Log a dictionary of tags to the current run.
    
    Parameters:
    -----------
    tags_dict : dict
        Dictionary of tags to log
    """
    for key, value in tags_dict.items():
        mlflow.set_tag(key, value)
    print(f"Logged {len(tags_dict)} tags")


def end_run(status="FINISHED"):
    """
    End the current MLflow run.
    
    Parameters:
    -----------
    status : str
        Status of the run (FINISHED, FAILED, KILLED)
    """
    mlflow.end_run(status=status)
    print(f"Run ended with status: {status}")


def load_model(run_id, model_name="model"):
    """
    Load a logged model from MLflow.
    
    Parameters:
    -----------
    run_id : str
        The run ID containing the model
    model_name : str
        Name of the model artifact
    
    Returns:
    --------
    model : sklearn model
        Loaded scikit-learn model
    """
    model_uri = f"runs:/{run_id}/{model_name}"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded model from run: {run_id}")
    return model



def log_model_summary(model, X_train_shape, X_test_shape, num_features):
    """
    Log model and dataset summary information.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_train_shape : tuple
        Shape of training data
    X_test_shape : tuple
        Shape of test data
    num_features : int
        Number of features
    """
    mlflow.log_param("model_type", type(model).__name__)
    mlflow.log_param("num_features", num_features)
    mlflow.log_param("train_samples", X_train_shape[0])
    mlflow.log_param("test_samples", X_test_shape[0])
    print("Logged model summary")





 

def log_everything_(run_name,model,best_model,X_train,y_train,X_val,y_val,y_pred,prefix,training_time):
    try:
        start_run(run_name)
        log_parameters(model.best_params_)
        log_model_summary(best_model, X_train.shape, X_val.shape, X_val.shape[1])
        log_metrics(y_val, y_pred, prefix=f'{prefix}_validation_')
        mlflow.log_metric("training_time_seconds", training_time)
        log_confusion_matrix(
            y_val, 
            y_pred, 
            labels=y_train.unique(), 
            figure_name=f"{prefix}_optimal_conf_matrix"
        )
        log_model_info(best_model,X_val,y_val ,prefix)
    except Exception as e:
        print(f"Run failed with error: {e}")
        end_run('FAILED')
    else:
        end_run('FINISHED')

