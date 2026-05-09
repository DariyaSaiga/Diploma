from sklearn.metrics import accuracy_score, f1_score, classification_report

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def compute_metrics(preds, labels, label_names=None):
    """Return accuracy, macro F1, and per-class classification report."""
    label_names = label_names or EMOTIONS
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    report = classification_report(labels, preds, target_names=label_names,
                                   digits=3, zero_division=0)
    return {"accuracy": acc, "macro_f1": f1, "report": report}
