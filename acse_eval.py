"""ACSE-Eval: Can LLMs threat model real-world cloud infrastructure?

Sarthak Munshi, Swapnil Pathak, Sonam Ghatode, Thenuga Priyadharshini,
Ashutosh Rana, Dhivya Chandramouleeswaran 

<PREPRINT URL>

# Eval
inspect eval inspect_evals/acse_eval
"""

import json
import re
from typing import Any, Callable
from pathlib import Path
import asyncio
import threading

# Added import for Hugging Face datasets
from datasets import load_dataset as hf_load_dataset

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset
from huggingface_hub import snapshot_download
from inspect_ai.scorer import choice, match
from inspect_ai.solver import generate, prompt_template
from platformdirs import user_cache_dir
from gitingest import ingest_async
from inspect_evals.acse_eval.prompt import TEMPLATE_WITHOUT_RELATIONSHIP_CONTEXT, TEMPLATE_WITH_RELATIONSHIP_CONTEXT
from inspect_evals.acse_eval.scorer import stride_false_positive_scorer, stride_fraction_identified_scorer, affected_components_fp_scorer, affected_components_scorer, semantic_similarity_scorer, rouge_l_scorer, mitre_recall_scorer, cwe_recall_scorer, owasp_recall_scorer

ACSE_EVAL_DATASET_LOCATION = (
    Path(user_cache_dir("inspect_evals")) / "acse_eval_dataset" / "ACSE-Eval"
)

SCORERS = [
    rouge_l_scorer(),
    semantic_similarity_scorer(),
    stride_false_positive_scorer(),
    stride_fraction_identified_scorer(),
    affected_components_fp_scorer(),
    affected_components_scorer(),
    mitre_recall_scorer(),
    cwe_recall_scorer(),
    owasp_recall_scorer()
]

def ingest_in_thread(x):
    result = {}
    def runner():
        result['value'] = asyncio.run(ingest_async(x))
    t = threading.Thread(target=runner)
    t.start()
    t.join()

    return result['value']


def _prepare_shot_example(record: dict[str, Any], diagrams: bool) -> dict[str, str]:
    architecture_description_path = ACSE_EVAL_DATASET_LOCATION / record["architecture_description"]
    infra_as_code_path = ACSE_EVAL_DATASET_LOCATION / record["infra_as_code"]
    threat_model_path = ACSE_EVAL_DATASET_LOCATION / record["threat_model"]

    architecture_description = ""
    if architecture_description_path.exists(): # Check if file exists before reading
        with open(architecture_description_path, 'r') as f:
            architecture_description = f.read()
    
    summary, _, content = ingest_in_thread(str(infra_as_code_path))
    
    threat_model = ""
    if threat_model_path.exists(): # Check if file exists
        with open(threat_model_path, 'r') as f:
            threat_model = f.read()

    input_str = ""
    if diagrams:
        input_str = f"""
        Architecture Description Start:
        ```
        {architecture_description}
        ```
        Architecture Description End

        ==========================================

        Infrastructure as Code (CDK) Start:
        ```
        {summary}
        
        {content}
        ```
        Infrastructure as Code (CDK) End"""
    else:
        input_str = f"""
        Infrastructure as Code (CDK) Start:
        ```
        {summary}
        
        {content}
        ```
        Infrastructure as Code (CDK) End"""
    
    return {"input": input_str, "target": threat_model}


# Helper function to create a few-shot prompt template
def _create_few_shot_prompt(base_user_prompt_template: str, shot_examples: list[dict[str, str]]) -> str:
    few_shot_block = ""
    for ex in shot_examples:
        few_shot_block += f"User:\n{ex['input']}\n\nAssistant:\n{ex['target']}\n\n---\n\n" # Ensure newlines are correctly formatted
    # The base_user_prompt_template already contains {input} for the current query
    return few_shot_block + base_user_prompt_template


# Load the first 3 records for few-shot examples
# Note: hf_load_dataset might require trust_remote_code=True if the dataset has custom loading scripts.
# For ACSE-Eval, it's typically simple JSONL files, so it might not be strictly necessary.


# --- End of new code for 3-shot evaluation ---

@task
def acse_eval() -> Task:
    """Inspect task implementing the ACSE-Eval benchmark without the relationship context (0-shot)."""
    return Task(
        dataset=get_acse_eval_dataset(diagrams=False),
        solver=[prompt_template(template=TEMPLATE_WITHOUT_RELATIONSHIP_CONTEXT), generate()],
        scorer=SCORERS, 
    )

@task
def acse_eval_relationship_context() -> Task:
    """Inspect task implementing the ACSE-Eval benchmark with the relationship context (0-shot)."""
    return Task(
        dataset=get_acse_eval_dataset(diagrams=True),
        solver=[prompt_template(template=TEMPLATE_WITH_RELATIONSHIP_CONTEXT), generate()],
        scorer=SCORERS, 
    )

# --- New tasks for 3-shot evaluation ---
@task
def acse_eval_3shot() -> Task:
    """Inspect task implementing the ACSE-Eval benchmark without relationship context (3-shot)."""
    return Task(
        dataset=get_acse_eval_dataset_3shot(diagrams=False), # Dataset remains the same
        solver=[prompt_template(template=TEMPLATE_WITHOUT_RELATIONSHIP_CONTEXT), generate()],
        scorer=SCORERS, 
    )

@task
def acse_eval_relationship_context_3shot() -> Task:
    """Inspect task implementing the ACSE-Eval benchmark with relationship context (3-shot)."""
    return Task(
        dataset=get_acse_eval_dataset_3shot(diagrams=True), # Dataset remains the same
        solver=[prompt_template(template=TEMPLATE_WITH_RELATIONSHIP_CONTEXT), generate()],
        scorer=SCORERS,
    )
# --- End of new tasks for 3-shot evaluation ---

def record_to_sample(diagrams: bool) -> Callable[[dict[str, Any]], Sample]:
    def record_to_sample(record: dict[str, Any]) -> Sample:
        architecture_description = open(ACSE_EVAL_DATASET_LOCATION / record["architecture_description"], 'r').read()
        summary, _, content = ingest_in_thread(str(ACSE_EVAL_DATASET_LOCATION / record["infra_as_code"]))
        threat_model = open(ACSE_EVAL_DATASET_LOCATION / record["threat_model"], 'r').read()
        input_str = ""

        if diagrams:
            input_str = f"""
            Architecture Description Start:
            ```
            {architecture_description}
            ```
            Architecture Description End

            ==========================================

            Infrastructure as Code (CDK) Start:
            ```
            {summary}
            
            {content}
            ```
            Infrastructure as Code (CDK) End"""
        else:
            input_str = f"""
            Infrastructure as Code (CDK) Start:
            ```
            {summary}
            
            {content}
            ```
            Infrastructure as Code (CDK) End"""

        sample = {
            "id": record["name"],
            "input": input_str,
            "target": threat_model
        }

        return Sample(**sample)

    return record_to_sample


def record_to_sample_3shot(diagrams: bool, template: str) -> Callable[[dict[str, Any]], Sample]:
    def record_to_sample(record: dict[str, Any]) -> Sample:
        architecture_description = open(ACSE_EVAL_DATASET_LOCATION / record["architecture_description"], 'r').read()
        summary, _, content = ingest_in_thread(str(ACSE_EVAL_DATASET_LOCATION / record["infra_as_code"]))
        threat_model = open(ACSE_EVAL_DATASET_LOCATION / record["threat_model"], 'r').read()
        input_str = ""

        if diagrams:
            input_str = f"""
             You are given 3 examples to look at:
            {template}

            Architecture Description Start:
            ```
            {architecture_description}
            ```
            Architecture Description End

            ==========================================

            Infrastructure as Code (CDK) Start:
            ```
            {summary}
            
            {content}
            ```
            Infrastructure as Code (CDK) End"""
        else:
            input_str = f"""
             You are given 3 examples to look at:
            {template}

            Infrastructure as Code (CDK) Start:
            ```
            {summary}
            
            {content}
            ```
            Infrastructure as Code (CDK) End"""

        sample = {
            "id": record["name"],
            "input": input_str,
            "target": threat_model
        }

        return Sample(**sample)

    return record_to_sample


def get_acse_eval_dataset(diagrams: bool) -> Dataset:
    """Get ACSE-Eval dataset."""

    snapshot_download(
        repo_id="ACSE-Eval/ACSE-Eval",
        repo_type="dataset",
        local_dir=ACSE_EVAL_DATASET_LOCATION,
    )

    dataset = hf_dataset(
        path=ACSE_EVAL_DATASET_LOCATION.as_posix(),
        split="train",
        trust=True,
        sample_fields=record_to_sample(diagrams=diagrams),
    )

    return dataset


def get_acse_eval_dataset_3shot(diagrams: bool) -> Dataset:
    # Ensure dataset is downloaded (this will run once when the module is loaded)
    snapshot_download(
        repo_id="ACSE-Eval/ACSE-Eval",
        repo_type="dataset",
        local_dir=ACSE_EVAL_DATASET_LOCATION,
    )

    try:
        full_dataset = hf_load_dataset(
            ACSE_EVAL_DATASET_LOCATION.as_posix(),
            split="train",
            trust_remote_code=True  # Added for safety with local/custom datasets
        )
        # Shuffle the dataset and then select the first 3 for few-shot examples
        shuffled_dataset = full_dataset.shuffle(seed=42) # Using a fixed seed for reproducibility
        raw_few_shot_records_full = shuffled_dataset.select(range(3)) # Get first 3 records from shuffled set
    except Exception as e:
        print(f"Error loading or shuffling dataset for few-shot examples: {e}")
        print(f"Attempting to load from specific file: {ACSE_EVAL_DATASET_LOCATION / 'ACSE-Eval.jsonl'}")
        raise e 

    FEW_SHOT_EXAMPLES = [_prepare_shot_example(record, diagrams=diagrams) for record in raw_few_shot_records_full]
    TEMPLATE_3SHOT = ""

    if diagrams:
        TEMPLATE_3SHOT = _create_few_shot_prompt(TEMPLATE_WITH_RELATIONSHIP_CONTEXT, FEW_SHOT_EXAMPLES)
    else:
        TEMPLATE_3SHOT = _create_few_shot_prompt(TEMPLATE_WITHOUT_RELATIONSHIP_CONTEXT, FEW_SHOT_EXAMPLES)
    

    dataset = hf_dataset(
        path=ACSE_EVAL_DATASET_LOCATION.as_posix(),
        split="train",
        trust=True,
        sample_fields=record_to_sample_3shot(diagrams=diagrams, template=TEMPLATE_3SHOT),
    )

    return dataset
