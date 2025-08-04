# 2. `find-wiki-styles`: Searches Wikipedia for the styles in ArtGraph, and saves the Wikipedia URL and summary of the style in a CSV file and as properties of the nodes in the graph.

# 3. `download-wikidata-images` and `connect-artworks-to-wikidata`: these two operations are used to connect the results obtained in ["Enhancing Entity Alignment Between Wikidata and ArtGraph using LLMs"](https://cris.unibo.it/handle/11585/963722), where they used LLaMA to attach ArtGraph artworks to Wikidata items. Unfortunately, they were able only to attach 13625 elements, and the results only contain the title of the artwork together with the Wikidata ID. The `download-wikidata-images` operation downloads the images from Wikidata, resulting in 8731 images, and the `connect-artworks-to-wikidata` operation connects the artworks to the Wikidata items. In the second operation, we search for the title inside ArtGraph, and if we find a match, we connect the artwork to the Wikidata item. If there are multiple matches, we attach the ID to the node whose image is the most similar to the one in Wikidata, using CLIP. This process reconnects 13533 artworks to Wikidata.

# 4. `extract-urls-from-wikidata`: this operation extracts Wikipedia URLs or external URLs attached to a Wikidata item, and saves them by updating the CSV file created in the previous step. We find Wikipedia entries for 1618 artworks, while the count of external URLs is 4867.

# 5. `extract-described-at-texts`: while linked pages at Wikipedia URLs usually follow a known structure, external URLs can point to any website. This operation collects all the paragraphs of text from the external URLs, uses LLaMA 3.1 as a binary classifier to be applied on each paragraph to determine if it is about artistic commentary or if contains useless information. The paragraphs that are classified as artistic commentary are concatenated and saved in a CSV file where the key is the artwork ID and the value is the text. This operation extracts 1348 texts.

# @cli.command()
# @click.pass_context
# def find_wiki_styles(ctx):
#     logger = ctx.obj["logger"]
#     graph = ctx.obj["graph"]

#     graph_wikipedia.find_wiki_styles(graph)
#     logger.info(
#         f"Wikipedia styles found and saved at {get_data_dir() /  'graph' / 'styles.csv'}"
#     )

#     graph_ops.write_csv_to_graph(graph, get_data_dir() / "graph" / "styles.csv")
#     logger.info("Wikipedia styles written to the graph")


# @cli.command()
# @click.pass_context
# def download_wikidata_images(ctx):
#     logger = ctx.obj["logger"]

#     df = pd.read_csv(get_data_dir() / "graph" / "linked_artworks.csv")
#     entity_ids = df["wikidata_url"].str.split("/").str[-1].tolist()
#     output_dir = get_data_dir() / "wikidata_images"
#     existing_images = set([img.stem for img in output_dir.glob("*.jpg")])
#     entity_ids = [
#         entity_id for entity_id in entity_ids if entity_id not in existing_images
#     ]
#     images_downloader.download_and_save_images_wikidata(
#         entity_ids, output_dir, img_res=224
#     )
#     logger.info(f"Images downloaded and saved at {output_dir}")


# @cli.command()
# @click.pass_context
# def connect_artworks_to_wikidata(ctx):
#     logger = ctx.obj["logger"]
#     graph = ctx.obj["graph"]

#     links, unmatched_titles, unmatched_artists = (
#         graph_wikidata.connect_linked_artworks_wikidata_urls(
#             graph,
#             get_data_dir() / "wikidata_images",
#             get_data_dir() / "images",
#             get_data_dir() / "graph" / "linked_artworks.csv",
#         )
#     )

#     logger.info(f"Links saved at {get_data_dir() / 'graph' / 'wikidata_links.csv'}")
#     logger.info(f"No. of unmatched titles: {len(unmatched_titles)}")
#     logger.info(f"No. of unmatched artists: {len(unmatched_artists)}")
#     logger.info(f"No. of links: {len(links)}")


# @cli.command()
# @click.pass_context
# def extract_urls_from_wikidata(ctx):
#     logger = ctx.obj["logger"]

#     graph_wikidata.extract_urls_from_wikidata(
#         get_data_dir() / "graph" / "wikidata_links.csv"
#     )
#     logger.info(
#         f"Extracted URLs saved at {get_data_dir() / 'graph' / 'wikidata_links.csv'}"
#     )
#     graph_ops.write_csv_to_graph(
#         ctx.obj["graph"], get_data_dir() / "graph" / "wikidata_links.csv"
#     )
#     logger.info("Wikidata links written to the graph")


# @cli.command()
# @click.pass_context
# def extract_described_at_texts(ctx):
#     logger = ctx.obj["logger"]
#     graph = ctx.obj["graph"]

#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )
#     llm = HuggingFacePipeline.from_model_id(
#         model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
#         task="text-generation",
#         model_kwargs={
#             "torch_dtype": torch.bfloat16,
#             "quantization_config": quantization_config,
#         },
#         device_map="auto",
#         pipeline_kwargs=dict(
#             max_new_tokens=8,
#         ),
#     )
#     chat_model = ChatHuggingFace(llm=llm)
#     chat_model.llm.pipeline.tokenizer.pad_token_id = (
#         chat_model.llm.pipeline.tokenizer.eos_token_id
#     )

#     texts_extractor.extract_described_at_texts(graph, chat_model)
#     logger.info(
#         f"Extracted texts saved at {get_data_dir() / 'texts'/ 'described_at.csv'}"
#     )
# 
# @cli.command()
# @click.pass_context
# def make_graph_split(ctx):
#     # TODO: remove
#     logger = ctx.obj["logger"]
#     graph = ctx.obj["graph"]

#     graph_ops.make_graph_split(graph, get_data_dir() / "graph" / "splits", split=0.85)
#     logger.info(f"Graph split saved at {get_data_dir() / 'graph' / 'splits'}")


# @cli.command()
# @click.pass_context
# def write_graph_as_tsv(ctx):
#     # TODO: remove
#     logger = ctx.obj["logger"]
#     graph = ctx.obj["graph"]

#     graph_ops.write_graph_as_tsv(
#         graph,
#         get_data_dir() / "graph" / "graph.tsv",
#         get_data_dir() / "graph" / "splits" / "test.txt",
#     )
#     logger.info(f"Graph saved as TSV at {get_data_dir() / 'graph' / 'graph.tsv'}")