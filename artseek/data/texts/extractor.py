import re

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

from ...utils.dirutils import get_data_dir

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def extract_described_at_texts(graph: Neo4jGraph, llm) -> None:
    """Extract text from the "described at" URLs of the artworks.

    This algorithm obtains all the paragraphs from the "described at" URLs of the artworks and filters out the paragraphs
    that are not about artistic commentary on the artwork. The filtered paragraphs are saved in a CSV file in a single
    text column.
    Estimated time: 4 hours.

    Args:
        graph (Neo4jGraph): The graph database.
        llm: The language model.
    """
    query = """
    MATCH (a:Artwork)
    WHERE EXISTS(a.described_at_url)
    RETURN ID(a) as id, a.described_at_url AS url
    """
    result = graph.query(query)
    id2url = {r["id"]: r["url"] for r in result}

    messages = [
        (
            "system",
            "You are a helpful AI paragraph checker. Check if the paragraph is a clean paragraph about artistic commentary on an artwork. If it is, answer 'yes'. If not, answer 'no'. If the paragraph contains metadata, answer 'no'. If it is empty, answer 'no'. If it is about museum information, website information, events, products, downloads, exhibitions, answer 'no'. Anything not concerning the artwork and the artist is 'no'. If the text is not in English, answer 'no'. Don't add anything after 'yes' or 'no'.",
        ),
        ("user", "{paragraph}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | llm | StrOutputParser()

    def get_paragraphs_from_url(url: str) -> list[str]:
        response = requests.get(url, timeout=10, verify=False)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        return [p.get_text() for p in paragraphs]

    def clean_string(input_string: str) -> str:
        cleaned_string = input_string.replace("\n", " ")
        cleaned_string = re.sub(r"\s+", " ", cleaned_string)
        return cleaned_string

    try:
        df = pd.read_csv(get_data_dir() / "texts" / "described_at.csv")
        largest_id = df["id"].max()
    except FileNotFoundError:
        largest_id = 0
    texts = {}
    for i, (id, url) in enumerate(tqdm(id2url.items())):
        if id <= largest_id:
            continue
        try:
            paragraphs = get_paragraphs_from_url(url)
        except Exception as e:
            print(f"Error with {url}: {e}")
            continue
        text = ""
        for paragraph in paragraphs:
            # paragraph max 1024 words to analyze
            tokens = paragraph.split()[:1024]
            # at least 10 words
            if len(tokens) < 10:
                continue
            p = " ".join(tokens)
            response = chain.invoke({"paragraph": p}).strip()
            start_str = "<|start_header_id|>assistant<|end_header_id|>"
            response = response[response.index(start_str) + len(start_str) :]
            if response.strip().lower().startswith("yes"):
                text += " " + paragraph
        text = text.strip()
        if text:
            texts[id] = clean_string(text)
        if i % 100 == 0:
            new_df = pd.DataFrame(texts.items(), columns=["id", "text"])
            df = pd.concat([df, new_df]).drop_duplicates(subset=["id"], keep="first")
            df.to_csv(get_data_dir() / "texts" / "described_at.csv", index=False)

    new_df = pd.DataFrame(texts.items(), columns=["id", "text"])
    df = pd.concat([df, new_df]).drop_duplicates(subset=["id"], keep="first")
    df.to_csv(get_data_dir() / "texts" / "described_at.csv", index=False)
