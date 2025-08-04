import re
from pathlib import Path

import pandas as pd
import torch
from langchain_community.graphs import Neo4jGraph
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from unidecode import unidecode
from w3lib.html import replace_entities
from wikidata.client import Client
import concurrent.futures

from ...utils.dirutils import get_data_dir
from ...utils.lookups import ARTIST_NAMES_LOOKUP, ARTWORK_ARTIST_NAMES_LOOKUP


@torch.no_grad()
def connect_linked_artworks_wikidata_urls(
    graph: Neo4jGraph,
    src_img_dir: Path | str,
    dest_img_dir: Path | str,
    file_path: Path | str,
) -> tuple[dict, list, list]:
    """Connect linked artworks with their Wikidata URLs.

    If the source image is available, the function will use the CLIP model to find the best match.

    Args:
        graph (Neo4jGraph): The graph database.
        src_img_dir (Path | str): Directory containing the source images (Wikidata images).
        dest_img_dir (Path | str): Directory containing the destination images (ArtGraph images).
        file_path (Path | str): Path to the CSV file containing the linked artworks.

    Returns:
        dict: A dictionary containing the links between the artworks and their Wikidata URLs.
        list: A list of unmatched titles.
        list: A list of unmatched artists.
    """

    # define functions to clean strings
    def clear_title(title: str) -> str:
        return unidecode(re.sub(r"\W+", "", replace_entities(title)).lower())

    def clear_artist(artist: str) -> str:
        return unidecode(artist.lower()).replace(" ", "-")

    # read the csv file
    df = pd.read_csv(file_path)

    # load the CLIP model
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # read all artworks with their artists
    query = """
    MATCH (artwork: Artwork) -[createdBy]-> (artist: Artist)
    RETURN ID(artwork) as id, artwork.title as title, artwork.name as file_name, artist.name as artist
    """
    result = graph.query(query)

    # make a dict with the results
    result_dict = {}
    for record in result:
        # clean string by removing special characters and converting to lowercase
        title = clear_title(record["title"])
        if not title in result_dict:
            result_dict[title] = []
        result_dict[title].append(
            {
                "id": record["id"],
                "file_name": record["file_name"],
                "artist": record["artist"],
            }
        )

    links = {}
    processed = set()
    # iterate over the dataframe
    for i, row in tqdm(df.iterrows()):
        title = clear_title(row["title"])
        artist = clear_artist(row["artist"])

        try:
            src_image = Image.open(
                src_img_dir / f'{row["wikidata_url"].split("/")[-1]}.jpg'
            )
            src_features = model.get_image_features(
                **processor(images=src_image, return_tensors="pt")
            )
        except FileNotFoundError as e:
            src_image, src_features = None, None

        # human-defined edge cases
        if artist in ARTIST_NAMES_LOOKUP:
            artist = ARTIST_NAMES_LOOKUP[artist]
        if (title, artist) in ARTWORK_ARTIST_NAMES_LOOKUP:
            artist = ARTWORK_ARTIST_NAMES_LOOKUP[(title, artist)]

        unmatched_titles = []
        unmatched_artists = []
        if (title, artist) in processed:
            print(f"Already processed: {title} - {artist}")
        processed.add((title, artist))

        if title in result_dict:
            max_score = 0
            max_score_id = None
            found = False
            for artwork in result_dict[title]:
                if artist == artwork["artist"]:
                    found = True
                    if src_image is None:
                        links[artwork["id"]] = row["wikidata_url"]
                        break
                    else:
                        artwork_image = Image.open(dest_img_dir / artwork["file_name"])
                        dest_features = model.get_image_features(
                            **processor(images=artwork_image, return_tensors="pt")
                        )
                        score = (
                            (src_features @ dest_features.T)
                            / (src_features.norm(dim=-1) * dest_features.norm(dim=-1))
                        ).item()
                        if score > max_score:
                            max_score = score
                            max_score_id = artwork["id"]
            if max_score_id is not None:
                links[max_score_id] = row["wikidata_url"]
            if not found:
                unmatched_artists.append(artist)
        else:
            unmatched_titles.append(title)

    with open(get_data_dir() / "graph" / "wikidata_links.csv", "w") as f:
        f.write("id,wikidata_url\n")
        for key, value in links.items():
            f.write(f"{key},{value}\n")

    return links, unmatched_titles, unmatched_artists


def extract_urls_from_wikidata(file_path: Path | str):
    """Extract Wikipedia and "described at" URLs from Wikidata.

    Args:
        file_path (Path | str): Path to the CSV file containing the Wikidata URLs.
    """
    client = Client()
    described_at_prop = client.get("P973")
    df = pd.read_csv(file_path)

    # Function to process each URL and fetch data
    def process_wikidata_url(url):
        entity_id = url.split("/")[-1]
        entity = client.get(entity_id, load=True)
        
        # Try to fetch the Wikipedia URL and described at URL
        wikipedia_url = None
        described_at_url = None
        try:
            wikipedia_url = entity.attributes["sitelinks"]["enwiki"]["url"]
        except KeyError:
            pass
        try:
            described_at_url = entity[described_at_prop]
        except KeyError:
            pass

        return wikipedia_url, described_at_url

    # Use ThreadPoolExecutor to process each Wikidata URL in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_wikidata_url, url) for url in df["wikidata_url"]]
        
        wikipedia_urls = []
        described_at_urls = []
        
        # Process results and update progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(df)):
            wikipedia_url, described_at_url = future.result()
            wikipedia_urls.append(wikipedia_url)
            described_at_urls.append(described_at_url)

    # Add the collected data to the DataFrame
    df["wikipedia_url"] = wikipedia_urls
    df["described_at_url"] = described_at_urls

    # Save the updated DataFrame
    df.to_csv(file_path, index=False)

    # Print the summary
    print(f"Found Wikipedia URLs for {len(df[df['wikipedia_url'].notnull()])} entities.")
    print(f"Found described at URLs for {len(df[df['described_at_url'].notnull()])} entities.")
