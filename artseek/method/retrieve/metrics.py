from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalRecall


class RetrievalMetrics(MetricCollection):
    def __init__(self):
        metrics = {
            "ndcg@5": RetrievalNormalizedDCG(top_k=5),
            "r@1": RetrievalRecall(top_k=1),
        }
        super().__init__(metrics)
