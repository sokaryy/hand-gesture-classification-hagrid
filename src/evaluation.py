from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
def evaluate_metrics(y_true,y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    print(f'Model accuracy: {metrics['accuracy']}')
    print(f'Model precision: {metrics['precision']}')
    print(f'Model recall: {metrics['recall']}')
    print(f'Model f1: {metrics['f1']}')

    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_true.unique())
    disp.plot(ax=ax, cmap='Blues')
    
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return metrics