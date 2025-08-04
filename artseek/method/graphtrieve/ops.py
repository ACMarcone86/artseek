from pathlib import Path
import torch
import torch.nn as nn
from pykeen.triples import CoreTriplesFactory, TriplesFactory


def load_triples_factory(triples_path: str | Path) -> CoreTriplesFactory:
    triples_factory = TriplesFactory.from_path(triples_path)
    training, testing, validation = triples_factory.split([0.8, 0.1, 0.1], random_state=42)
    return training, testing, validation


def load_model(model_dir: str | Path) -> nn.Module:
    """Load a model from a directory.

    Args:
        model_dir (str | Path): The model directory.

    Returns:
        nn.Module: The model.
    """
    return torch.load(model_dir / "trained_model.pkl")


def entities_to_ids(
    entities: list[str],
    triples_factory: CoreTriplesFactory,
    return_missing_dict: bool = False,
    return_index_dict: bool = False,
) -> list[str]:
    """Get the IDs (indexes in the model) of the entities.

    Args:
        entities (list[str]): The entities.
        triples_factory (CoreTriplesFactory): The triples factory.
        return_missing_dict (bool): Whether to return a list with the missing entities.
        return_index_dict (bool): Whether to return a dictionary with the indexes of the entities in the original list.

    Returns:
        dict[str, int]: The entities to IDs dictionary.
        list[str]: The missing entities.
        dict[str, int]: The entities to indexes dictionary.
    """
    entities2ids = {}
    missing_entities = []
    entities2indexes = {}

    for i, entity in enumerate(entities):
        entities2indexes[entity] = i
        try:
            entities2ids[entity] = triples_factory.entities_to_ids([entity])[0]
        except:
            missing_entities.append(entity)

    output = (entities2ids,)
    if return_missing_dict:
        output += (missing_entities,)
    if return_index_dict:
        output += (entities2indexes,)
    return output
