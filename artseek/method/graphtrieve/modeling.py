from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import pykeen.models
from langchain_community.graphs import Neo4jGraph
from pykeen.triples import CoreTriplesFactory
from safetensors.torch import load_file, save_file
from functools import partial

from ...utils.dirutils import get_data_dir, get_model_checkpoints_dir


def entity_constrainer(
    entity_embeddings: torch.FloatTensor,
    artwork_embeddings: torch.FloatTensor,
    ids: list[int],
    **kwargs,
) -> torch.FloatTensor:
    """A constraint function that constrains the embeddings of the artworks.

    Artwork embeddings are fixed and should not be changed, while we apply normalization to the entity embeddings.

    Args:
        entity_embeddings (torch.FloatTensor): The entity embeddings.
        artwork_embeddings (torch.FloatTensor): The artwork embeddings.
        ids (list[int]): The indexes of the artworks in the entity embeddings.

    Returns:
        torch.FloatTensor: The entity embeddings.
    """
    entity_embeddings = nn.functional.normalize(entity_embeddings)
    entity_embeddings[ids] = artwork_embeddings.to(entity_embeddings.device)
    return entity_embeddings


def get_kge_model(
    model_name: str,
    triples_factory: CoreTriplesFactory,
    artwork_embeddings_path: str | Path = None,
    graph: Neo4jGraph = None,
    embs_init_path: str | Path = None,
    model_kwargs: dict[str, Any] = {},
):
    """Get a knowledge graph embedding model.

    Args:
        model_name (str): The name of the model.
        triples_factory (CoreTriplesFactory): The triples factory.
        graph (Neo4jGraph): The graph.
        embs_init_path (str | Path): The path to the embeddings initialization file.
        artwork_embeddings_path (str | Path): The path to the artwork embeddings file.
        model_kwargs (dict[str, Any]): The model keyword arguments.
    """
    assert embs_init_path is not None or (
        artwork_embeddings_path is not None and graph is not None
    ), "Either embs_init_path or artwork_embeddings_path and graph must be provided."

    # instantiate class with model_name from pykeen.models
    model_class = getattr(pykeen.models, model_name)

    # read artworks from the graph
    query = """
    MATCH (a:Artwork)
    RETURN ID(a) as id, a.name as name
    ORDER BY id
    """
    result = graph.query(query)
    artworks_id2name = {row["id"]: row["name"] for row in result}

    # keep the neo4j ids of the artworks
    keys = list(artworks_id2name.keys())
    keys = [str(key) for key in keys]

    # make list of ids (in the triples factory) of the artworks
    # make list of indexes (in the embeddings matrix) of the artworks
    ids = []
    indexes = []
    for i, key in enumerate(keys):
        try:
            ids.append(triples_factory.entity_to_id[key])
            indexes.append(i)
        except:
            pass

    # read the artwork embeddings
    artwork_embeddings = load_file(artwork_embeddings_path)["features"]
    artwork_embeddings = nn.functional.normalize(artwork_embeddings)
    
    # if model was already initialized
    if embs_init_path is None:
        # create a fake model
        model = model_class(triples_factory=triples_factory, **model_kwargs)

        # fill the model embeddings with the artwork embeddings
        kge_embeddings = model.entity_representations[0]._embeddings.weight.detach()
        kge_embeddings[ids] = artwork_embeddings[indexes]
        # normalize the embeddings
        kge_embeddings = nn.functional.normalize(kge_embeddings)

        # save the embeddings initialization file for future use
        embs_init_path = (
            get_model_checkpoints_dir()
            / "graph_retriever"
            / "init"
            / f"{model_name}.safetensors"
        )
        save_file({"entity_representations.0._embeddings.weight": kge_embeddings}, embs_init_path)

        # delete the fake model
        del model

    entity_constrainer_partial = partial(entity_constrainer, artwork_embeddings=artwork_embeddings[indexes], ids=ids)
    model = model_class(triples_factory=triples_factory, entity_constrainer=entity_constrainer_partial, **model_kwargs)
    model.load_state_dict(load_file(embs_init_path), strict=False)

    return model
