from pathlib import Path
from typing import Any, Optional

import pandas as pd
from langchain_community.graphs import Neo4jGraph
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def add_field_to_node(
    graph: Neo4jGraph, node_id: int, field_name: str, field_value: Any
):
    """Add a field to a node in the graph database. The field is changed if it already exists.

    Args:
        graph (Neo4jGraph): The graph database.
        node_id (int): The ID of the node.
        field_name (str): The name of the field.
        field_value (Any): The value of the field.
    """
    query = f"MATCH (n) WHERE ID(n) = $node_id SET n.$field_name = $field_value"
    query = query.replace("$field_name", field_name)
    graph.query(
        query,
        {"node_id": node_id, "field_value": field_value},
    )


def write_csv_to_graph(graph: Neo4jGraph, file_path: Path | str) -> None:
    """Write a CSV file to the graph database.

    The CSV file should have a column named "id" that contains the ID of the node.
    The other columns are added as fields to the node.

    Args:
        graph (Neo4jGraph): The graph database.
        file_path (Path | str): The path to the CSV file.
    """
    df = pd.read_csv(file_path)

    for _, row in df.iterrows():
        for field_name, field_value in row.items():
            if field_name == "id":
                continue
            if pd.isna(field_value):
                continue
            add_field_to_node(graph, row["id"], field_name, field_value)


def make_graph_split(
    graph: Neo4jGraph, out_dir: str | Path, split: float = 0.85
) -> None:
    """Split the artworks in the graph database.

    Args:
        graph (Neo4jGraph): The graph database.
        split (float): The fraction of artworks to include in the train split.
        out_dir (str | Path): The directory where the splits will be saved.
    """
    query = """
    MATCH (a: Artwork)-[:hasGenre]->(g: Genre)
    RETURN ID(a) as artwork_id, ID(g) as genre_id
    """
    result = graph.query(query)
    artwork_ids = [r["artwork_id"] for r in result]
    genre_ids = [r["genre_id"] for r in result]

    artwork_train, artwork_test = train_test_split(
        artwork_ids, train_size=split, stratify=genre_ids, random_state=42
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train.txt", "w") as f:
        for artwork_id in artwork_train:
            f.write(f"{artwork_id}\n")

    with open(out_dir / "test.txt", "w") as f:
        for artwork_id in artwork_test:
            f.write(f"{artwork_id}\n")


def write_graph_as_tsv(
    graph: Neo4jGraph, out_data_path: Path | str, anti_split_path: Path | str
) -> None:
    """Write the content of a Neo4j graph as a set of triples in a TSV file.

    The TSV file will have tuples of the form (ID(head), relationship, ID(tail)).

    Args:
        graph (Neo4jGraph): The graph database.
        out_data_path (Path | str): The path to the TSV file.
        anti_split_path (Path | str): The path to the file containing the split with artworks to exclude.
    """
    out_data_path = Path(out_data_path)
    anti_split_path = Path(anti_split_path)

    with anti_split_path.open("r") as f:
        anti_artwork_ids = set(int(line.strip()) for line in f)

    query = "MATCH ()-[r]->() RETURN DISTINCT type(r) AS relationship_name"
    result = graph.query(query)
    relationships = [record["relationship_name"] for record in result]

    full_graph = []
    for relationship in tqdm(relationships):
        query = f"MATCH (n)-[r:{relationship}]->(m) RETURN ID(n) AS head, ID(m) AS tail"
        result = graph.query(query)
        for record in result:
            head, tail = record["head"], record["tail"]
            if head in anti_artwork_ids or tail in anti_artwork_ids:
                continue
            triple = (head, relationship, tail)
            full_graph.append(triple)

    with open(out_data_path, "w") as f:
        for head, relationship, tail in full_graph:
            f.write(f"{head}\t{relationship}\t{tail}\n")


def get_nodes_connections(
    graph: Neo4jGraph, node_ids: list[int], relationship_name: Optional[str] = None
) -> list[int]:
    """Get the IDs of the nodes connected to a given node.

    Args:
        graph (Neo4jGraph): The graph database.
        node_id (int): The ID of the node.

    Returns:
        list[int]: The IDs of the connected nodes.
    """
    result = {}

    if relationship_name is None:
        query = f"MATCH (n)-[r]->(m) WHERE ID(n) IN {node_ids} RETURN ID(n) AS idn, ID(m) AS idm ORDER BY idn"
    else:
        query = f"MATCH (n)-[r:{relationship_name}]->(m) WHERE ID(n) IN {node_ids} RETURN ID(n) as idn, ID(m) AS idm ORDER BY idn"
    records = graph.query(query)

    for record in records:
        if not record["idn"] in result:
            result[record["idn"]] = []
        result[record["idn"]].append(record["idm"])

    return result


def get_nodes_by_id(graph: Neo4jGraph, node_ids: list[int]) -> list[dict]:
    """Get the nodes by their IDs.

    Args:
        graph (Neo4jGraph): The graph database.
        node_ids (list[int]): The IDs of the nodes.

    Returns:
        list[dict]: The nodes.
    """
    result = []

    for node_id in node_ids:
        query = f"MATCH (n) WHERE ID(n) = {node_id} RETURN n"
        records = graph.query(query)
        result.extend(records)

    return result


def get_nodes_by_type(graph: Neo4jGraph, node_type: str) -> list[dict]:
    """Get the nodes by their type.

    Args:
        graph (Neo4jGraph): The graph database.
        node_type (str): The type of the nodes.

    Returns:
        list[dict]: The nodes.
    """
    query = f"MATCH (n:{node_type}) RETURN ID(n) as id"
    records = graph.query(query)
    return records
