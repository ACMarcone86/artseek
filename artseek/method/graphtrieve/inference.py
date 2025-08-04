import torch
import torch.nn.functional as F
from langchain_community.graphs import Neo4jGraph
from PIL import Image
from torchvision import transforms

from ...data.graph.ops import get_nodes_by_type
from ...utils.dirutils import get_data_dir
from .ops import entities_to_ids, load_triples_factory


class GraphRetrieverInference:
    def __init__(
        self,
        model: torch.nn.Module,
        image_encoder: torch.nn.Module,
        graph: Neo4jGraph,
        transform,
    ):
        self.model = model
        self.image_encoder = image_encoder
        self.graph = graph
        self.transform = transform
        self.training, _, _ = load_triples_factory(
            get_data_dir() / "graph" / "graph.tsv"
        )

    @torch.no_grad()
    def predict_link(
        self, images: list[Image.Image], node_type: str, relation: str
    ) -> torch.FloatTensor:
        images = [self.transform(image) for image in images]
        images = torch.stack(images).to(self.image_encoder.device)
        image_embs = self.image_encoder.get_image_features(images)
        image_embs = F.normalize(image_embs, p=2, dim=1)

        # get the useful nodes by type
        candidates = get_nodes_by_type(self.graph, node_type)

        # get the candidates ids in the model
        candidates2ids = entities_to_ids(
            [str(candidate["id"]) for candidate in candidates], self.training
        )[0]

        # get the embeddings of the candidates
        candidate_embs = self.model.entity_representations[0](indices=torch.tensor(list(candidates2ids.values())))

        # relation embs
        relation_id = self.training.relation_to_id[relation]
        relation_emb = []
        for i in range(len(self.model.relation_representations)):
            relation_emb.append(self.model.relation_representations[i](indices=torch.tensor([relation_id])))
        if len(relation_emb) == 1:
            relation_emb = relation_emb[0]

        # compute the scores
        scores = []
        for candidate_emb in candidate_embs:
            tails = candidate_emb.unsqueeze(0).repeat(image_embs.shape[0], 1)
            score = self.model.interaction(image_embs, relation_emb, tails)
            scores.append(score)

        # get the best
        scores = torch.stack(scores, dim=1)
        best = torch.argmax(scores, dim=-1)

        # best is the index, now go from index to candidate id to candidate id in the graph
        best_candidates = [candidates[i]["id"] for i in best]

        return best_candidates



