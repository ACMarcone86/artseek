import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import click
from pathlib import Path
import yaml
from hydra.utils import instantiate
import json
import time

# Import the needed functions from eval.py
from .eval import (
    load_config,
    load_predictions,
    extract_messages,
    extract_json_from_message,
    process_explain_me_dataset,
    process_painting_form_dataset,
)

model_id = "meta-llama/Llama-3.1-8B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = quantized_model.generate(**input_ids, max_new_tokens=10)

print(tokenizer.decode(output[0], skip_special_tokens=True))

ASPECTS = [
    "Semantic Accuracy",
    "Fluency and Grammar",
    "Relevance and Saliency",
    "Specificity",
    "Creativity and Style",
]


def build_chat_messages(reference, hypothesis):
    system_message = {
        "role": "system",
        "content": (
            "You are an impartial and helpful evaluator for natural language generation (NLG). "
            "You will be given a reference text and a hypothesis (model prediction). "
            "Your task is to evaluate the quality of the hypothesis strictly based on the following aspects:\n"
            + "\n".join([f"- {aspect}" for aspect in ASPECTS])
            + "\n\nReturn your evaluation as a JSON object with keys: explanation (one sentence), "
            "and one key for each aspect with the score (1-5, higher is better)."
        ),
    }
    user_message = {
        "role": "user",
        "content": (
            f"Reference:\n{reference}\n\n"
            f"Hypothesis:\n{hypothesis}\n\n"
            "Please provide your evaluation in the requested JSON format."
        ),
    }
    return [system_message, user_message]


def llama_evaluate_batch(references, hypotheses, model, tokenizer, device="cuda"):
    results = []
    for hyp, refs in zip(hypotheses, references):
        reference = refs[0] if isinstance(refs, list) else refs
        messages = build_chat_messages(reference, hyp)
        # Use the chat template for input formatting
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        output = model.generate(
            input_ids,
            max_new_tokens=256,
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        print(decoded)
        # Extract only the JSON part from the output
        json_start = decoded.find("{")
        json_str = decoded[json_start:] if json_start != -1 else decoded
        try:
            result = json.loads(json_str)
        except Exception:
            result = {"raw_output": decoded}
        results.append(result)
        # Optional: sleep to avoid OOM or throttling
        time.sleep(0.1)
    return results


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the config file.",
)
def eval_llama(config_path):
    config = load_config(config_path)
    preds = load_predictions(config.out_dir)
    messages = extract_messages(preds)

    ds = config.dataset
    if hasattr(ds, "__class__") and ds.__class__.__name__ == "ExplainMeDataset":
        jsons = [extract_json_from_message(message["content"]) for message in messages]
        hypotheses, references = process_explain_me_dataset(ds, jsons)
    elif hasattr(ds, "__class__") and ds.__class__.__name__ == "PaintingFormDataset":
        hypotheses, references = process_painting_form_dataset(ds, messages)
    else:
        raise ValueError("Unknown dataset type")

    # LLaMA evaluation
    results = llama_evaluate_batch(
        references, hypotheses, quantized_model, tokenizer, device="cuda"
    )
    for res in results[:5]:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    cli()
