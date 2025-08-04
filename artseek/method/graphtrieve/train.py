from functools import partial

import torch
import torch.nn as nn
import click
import torch
import yaml
from hydra.utils import instantiate
from langchain_community.graphs import Neo4jGraph
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from .modeling import get_kge_model
from .ops import load_triples_factory


@click.command()
@click.option(
    "--config", type=click.Path(exists=True), help="Path to the YAML config file"
)
def train(config: str):
    try:
        config_dict = yaml.safe_load(open(config))
    except Exception as e:
        print(e)
        return
    config = instantiate(config_dict)

    seed = config["seed"]
    
    training, testing, validation = load_triples_factory(config["data"]["triples_path"])

    graph = Neo4jGraph()
    model = get_kge_model(**config["model"], triples_factory=training, graph=graph)

    pipeline_result = pipeline(
        model=model,
        training=training,
        testing=testing,
        validation=validation,
        **config["pipeline"],
    )
    pipeline_result.save_to_directory(config["output_dir"])


if __name__ == "__main__":
    train()
