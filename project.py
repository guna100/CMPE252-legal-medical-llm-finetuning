##Install dependancies
import os, re, torch, gc, matplotlib.pyplot as plt, numpy as np
## Copy and paste this for Colab, use as your first cell
"""
%%capture
import os, re, torch
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth  # Do this in local & cloud setups
else:
    import torch; v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
"""
## Other wise run this block in terminal and install
"""
pip install unsloth
pip install transformers==4.56.2
pip install --no-deps trl==0.22.2
xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
pip install datasets
"""

##Prep the data sets
from datasets import load_dataset, concatenate_datasets

pubmedqa_dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
medquad_dataset = load_dataset("lavita/MedQuAD", split="train")
legalqaeval_dataset = load_dataset("isaacus/LegalQAEval", split= ["val", "test"])
# cap_dataset = load_dataset("free-law/Caselaw_Access_Project", split="train")
def prepare_pubmedqa_columns(example):
    return {

        #merge final_decision and long)answer into a single column
        "output": f"{example['final_decision'].capitalize()}. {example['long_answer']}"
    }

def prepare_legalqaeval_columns(example):
    # Safely extract the fields, defaulting to empty strings
    ctx = example.get("text") or example.get("context") or ""
    q = example.get("question") or ""
    ans = example.get("answer") or example.get("answers") or ""

    # Handle the nested dictionary or list formats found in LegalQA
    if isinstance(ans, dict) and "text" in ans:
        ans = ans["text"]
    if isinstance(ans, list):
        ans = ans[0] if ans else ""

    return {
        "question": str(q),
        "context": str(ctx),
        "answer": str(ans).strip()
    }

pubmedqa_dataset = pubmedqa_dataset.map(prepare_pubmedqa_columns)
legalqaeval_dataset = concatenate_datasets(legalqaeval_dataset)
legalqaeval_dataset = legalqaeval_dataset.map(prepare_legalqaeval_columns)
# Filter out any rows where the answer/output is missing or literally says "None"
pubmedqa_dataset = pubmedqa_dataset.filter(lambda x: x["output"] is not None and str(x["output"]).strip() != "None" and str(x["output"]).strip() != "")
medquad_dataset = medquad_dataset.filter(lambda x: x["answer"] is not None and str(x["answer"]).strip() != "None" and str(x["answer"]).strip() != "")
legalqaeval_dataset = legalqaeval_dataset.filter(lambda x: x["answer"] is not None and str(x["answer"]).strip() not in ["None", ""])

#Merge columns into 2 columns, question and answer
from unsloth import to_sharegpt

def merge_prompt(dataset, merger, output):
  return to_sharegpt(
    dataset,
    merged_prompt = merger,
    output_column_name = output,
    conversation_extension = 0,  # Select more to handle longer conversations
)
pubmedqa_dataset = merge_prompt(pubmedqa_dataset, "[[ Question: {question}, Context: {context}]]", "output")
medquad_dataset = merge_prompt(medquad_dataset,"[[Question: {question}; Question Focus and Type: {question_focus}, {question_type}; Context URL: {document_url}]]" , "answer" )
legalqaeval_dataset = merge_prompt(legalqaeval_dataset, "[[ Question: {question}, Context: {context} ]]", "answer")

##Standardize the dataset with sharegpt
from unsloth import standardize_sharegpt

pubmedqa_dataset = standardize_sharegpt(pubmedqa_dataset)
medquad_dataset = standardize_sharegpt(medquad_dataset)
legalqaeval_dataset = standardize_sharegpt(legalqaeval_dataset)

##Apply chat template
chat_template = """Below is a question. Write a response that correctly answers the question.

### Question:
{INPUT}

### Response:
{OUTPUT}"""

from unsloth import apply_chat_template

def apply_chat_template_dynamic(dataset, tokenizer, chat_template):
  return apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
    # default_system_message = "You are a helpful assistant", << [OPTIONAL]
)