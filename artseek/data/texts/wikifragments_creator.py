import json
import re
from functools import partial
from pathlib import Path

from datasets import Dataset
from datasets import Image as DatasetsImage
from datasets import Sequence, Value, Features
from PIL import Image
from tqdm import tqdm


def _make_wikipedia_dataset(text_dir: str | Path, images_dir: str | Path):
    text_dir = Path(text_dir)
    images_dir = Path(images_dir)

    id = 0
    for file in tqdm(list(text_dir.glob("**/wiki*"))):
        with open(file, "r") as f:
            pages = f.readlines()
        for page in pages:
            images = []
            page = json.loads(page)
            text = page["text"]
            links = re.findall(r'<a href="([^"]+)">(.*?)</a>', text)

            # read all images from the page (infobox and thumb)
            for url, cap in links:
                if url.startswith("File"):
                    chars_to_skip = 7
                elif url.startswith("InfoboxHeader"):
                    chars_to_skip = 16
                else:
                    continue
                filename = url[chars_to_skip:].replace("%20", "_")
                # check if image is already saved
                try:
                    if (images_dir / f"{filename}.jpg").exists():
                        image_type = "thumb" if url.startswith("File") else "infobox"
                        images.append(
                            {
                                "image": Image.open(images_dir / f"{filename}.jpg"),
                                "type": image_type,
                                "caption": cap.strip(),
                                "url": url,
                            }
                        )
                except:
                    continue

            # split the text on the file image urls
            text_parts = re.split(r'<a href="File[^"]+">(?:.*?)</a>', text)

            # split the text parts into paragraphs
            paragraphs = []
            for i in range(len(text_parts)):
                ps = text_parts[i].split("\n")
                paragraphs.append(ps)

            # remove empty and bad paragraphs
            bad_ps_start = ["BULLET::::", "Section::::", "<templatestyles"]
            clean_paragraphs = []
            for ps in paragraphs:
                new_ps = []
                for p in ps:
                    if (
                        p
                        and not p.isspace()
                        and not any(p.strip().startswith(b) for b in bad_ps_start)
                    ):
                        new_ps.append(p.strip())
                clean_paragraphs.append(new_ps)
            paragraphs = clean_paragraphs

            # paragraphs[0][0] is the title, take it and remove it from the paragraphs, by sliding paragraphs[0]
            page_title = paragraphs[0][0]
            paragraphs[0] = paragraphs[0][1:]

            # if paragraphs[0] is not empty, remove a InfoboxHeader from it (if present)
            if paragraphs[0]:
                # find <a href="InfoboxHeader[^"]+">[^<]+</a> in paragraphs[0][0]
                infobox = re.findall(
                    r'<a href="InfoboxHeader[^"]+">(?:.*?)</a>', paragraphs[0][0]
                )
                if infobox:
                    paragraphs[0][0] = paragraphs[0][0].replace(infobox[0], "")
                    paragraphs[0][0] = paragraphs[0][0].strip()
                    # if it is empty, remove it
                    if not paragraphs[0][0]:
                        paragraphs[0] = paragraphs[0][1:]
                    page_infobox = infobox[0]
                else:
                    page_infobox = None

            # transform paragraphs into dicts and give them an index
            i = 0
            new_paragraphs = []
            for ps in paragraphs:
                new_ps = []
                for p in ps:
                    # replace any space sequence with a single space
                    p = re.sub(r"\s+", " ", p)
                    if len(p) < 100:
                        continue
                    new_ps.append({"id": id, "paragraph_id": i, "text": p, "images": []})
                    id += 1
                    i += 1
                new_paragraphs.append(new_ps)
            paragraphs = new_paragraphs

            # associate the images with the paragraphs
            imgs_stack = []
            if images and images[0]["type"] != "infobox":
                # add None at the beginning of images
                images = [None] + images
            for i, img in enumerate(images):
                imgs_stack.append(img)
                if paragraphs[i]:
                    if imgs_stack[0] is None:
                        imgs_stack = imgs_stack[1:]
                    paragraphs[i][0]["images"] = imgs_stack
                    imgs_stack = []

            # unnest the paragraphs
            paragraphs_unnested = [p for ps in paragraphs for p in ps]
            for p in paragraphs_unnested:
                p["title"] = page["title"]
                p["url"] = page["url"]
                p["wiki_id"] = int(p["url"][p["url"].rfind("curid=") + 6:])
            paragraphs = paragraphs_unnested

            # yield the paragraphs
            for p in paragraphs:
                yield p


def make_wikipedia_dataset(
    text_dir: str | Path, images_dir: str | Path, output_dir: str | Path
):
    partial_make_wikipedia_dataset = partial(
        _make_wikipedia_dataset, text_dir=text_dir, images_dir=images_dir
    )
    features = {
        "id": Value(dtype="int32", id=None),
        "title": Value(dtype="string", id=None),
        "text": Value(dtype="string", id=None), 
        "url": Value(dtype="string", id=None),
        "wiki_id": Value(dtype="int32", id=None),
        "paragraph_id": Value(dtype="int32", id=None),
        "images": Sequence(
            {
                "caption": Value(dtype="string", id=None),
                "image": DatasetsImage(decode=True, id=None),
                "type": Value(dtype="string", id=None),
                "url": Value(dtype="string", id=None),
            }
        ),
    }
    features = Features(features)
    ds = Dataset.from_generator(partial_make_wikipedia_dataset, features=features)
    ds.save_to_disk(output_dir)
