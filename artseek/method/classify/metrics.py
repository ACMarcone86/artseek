from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
import torch


class MulticlassClassificationMetrics(MetricCollection):
    def __init__(self, num_classes: int):
        super().__init__(
            {
                "accuracy": Accuracy(
                    task="multiclass", num_classes=num_classes, ignore_index=-100
                ),
                "accuracy_top_2": Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                    top_k=2,
                    ignore_index=-100,
                ),
                "precision": Precision(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                    ignore_index=-100,
                ),
                "recall": Recall(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                    ignore_index=-100,
                ),
                "f1": F1Score(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                    ignore_index=-100,
                ),
            }
        )


class MultilabelClassificationMetrics(MetricCollection):
    def __init__(self, num_labels: int):
        # thresholds is a list with 0.3, 0.4, up to 0.9
        thresholds = [i / 10 for i in range(3, 10)]
        metrics = {}
        for threshold in thresholds:
            threshold_str = str(threshold).replace(".", "_")
            metrics[f"accuracy@{threshold_str}"] = Accuracy(
                task="multilabel", num_labels=num_labels, threshold=threshold
            )

            metrics[f"accuracy_top_2@{threshold_str}"] = Accuracy(
                task="multilabel",
                num_labels=num_labels,
                top_k=2,
                threshold=threshold,
            )

            metrics[f"precision@{threshold_str}"] = Precision(
                task="multilabel",
                num_labels=num_labels,
                average="macro",
                threshold=threshold,
            )

            metrics[f"recall@{threshold_str}"] = Recall(
                task="multilabel",
                num_labels=num_labels,
                average="macro",
                threshold=threshold,
            )
            metrics[f"f1@{threshold_str}"] = F1Score(
                task="multilabel",
                num_labels=num_labels,
                average="macro",
                threshold=threshold,
            )
        super().__init__(metrics)
