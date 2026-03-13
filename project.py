##Install dependancies
import os, re, torch, gc, matplotlib.pyplot as plt, numpy as np, collections, evaluate
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
pip install evaluate
pip install rouge_score
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

## Functions to evaluate F1 and ROUGE score
from tqdm import tqdm


def compute_token_f1(prediction, truth):
    """Calculates Token-level F1 score between two strings."""
    pred_tokens = prediction.lower().split()
    truth_tokens = truth.lower().split()

    # If both are empty, perfect match
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    # If one is empty and the other isn't, 0 match
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0

    # Count common words
    common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common_tokens.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def run_f1_evaluation(model, tokenizer, dataset, is_unsloth):
    """Generates answers for a dataset and calculates average F1 using a custom template."""

    if is_unsloth:
        FastLanguageModel.for_inference(model)

    total_f1 = 0.0

    print("Generating answers for evaluation...")
    for item in tqdm(dataset):
        # Extract the question and answer from your dataset.
        # (Adjust these keys if your dataset columns are named differently)
        user_question = item["conversations"][0]["content"]
        ground_truth = item["conversations"][1]["content"]

        # 1. Format the prompt EXACTLY as your custom template dictates,
        # but leave the response area empty!
        inference_prompt = f"""Below is a question. Write a a response that correctly answers the question.

### Question:
{user_question}

### Response:
"""

        # 2. Tokenize the raw string directly (no need for apply_chat_template here!)
        inputs = tokenizer(inference_prompt, return_tensors="pt").to("cuda")

        # 3. Generate the answer
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # 4. Decode the output and slice off the prompt tokens to get just the new words
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 5. Calculate F1 for this specific question
        f1 = compute_token_f1(generated_text, ground_truth)
        total_f1 += f1

    avg_f1 = total_f1 / len(dataset)
    return avg_f1

def run_rouge_evaluation(model, tokenizer, dataset_name, num_samples=50):
    """
    Evaluates a model on summarization datasets (billsum or scientific_papers)
    using ROUGE metrics.
    """
    # 1. Load Metric and Data
    rouge = evaluate.load("rouge")

    if dataset_name == "billsum":
        ds = load_dataset("billsum", split=f"test[:{num_samples}]")
        text_key, target_key = "text", "summary"
        instr = "Summarize the following bill."
    elif dataset_name == "scientific_papers":
        ds = load_dataset("ccdv/pubmed-summarization", split=f"test[:{num_samples}]")
        text_key, target_key = "article", "abstract"
        instr = "Summarize the following paper."
    else:
        raise ValueError("Unsupported dataset. Choose 'billsum' or 'scientific_papers'.")

    # 2. Set model to inference mode
    FastLanguageModel.for_inference(model)

    preds = []
    refs = [item[target_key] for item in ds]

    print(f"Generating {num_samples} summaries for {dataset_name}...")

    for item in tqdm(ds):
        # 3. Format the Prompt
        prompt = f"### Instruction:\n{instr}\n\n### Input:\n{item[text_key]}\n\n### Response:\n"

        # 4. Tokenize and Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=256, # Summaries usually need more than 100 tokens
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # 5. Decode and Extract Response
        # Slicing inputs.input_ids.shape[1] ensures we don't include the prompt in the score
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        preds.append(generated_text.strip())

    # 6. Compute ROUGE
    results = rouge.compute(predictions=preds, references=refs)
    return results

##Set up training and evaluation loop

##Functions to print metrics and charts
def print_metrics(metrics, model, ft_type, dataset):
  print(f"Metrics for {model} and {ft_type} trained on {dataset} dataset")

  print(f"\n\nBaseline Average F1 Score:")
  print(f"\nPubMedQA F1: {metrics['bl_f1_pubmedqa']:.4f}")
  print(f"\nMedQUAD F1: {metrics['bl_f1_medquad']:.4f}")
  print(f"\nLegalQAEval F1: {metrics['bl_f1_legalqaeval']:.4f}")
  print(f"\nPubMedQA ROUGE (L): {metrics['bl_rouge_pubmedqa']['rougeL']:.4f}")
  print(f"\nMedQUAD ROUGE (L): {metrics['bl_rouge_medquad']['rougeL']:.4f}")
  print(f"\nLegalQAEval ROUGE (L): {metrics['bl_rouge_legalqaeval']['rougeL']:.4f}")

  print(f"\n\nFinetuned Average F1 Score:")
  print(f"\nEPubMedQA F1: {metrics['ft_f1_pubmedqa']:.4f}")
  print(f"\nMedQUAD F1: {metrics['ft_f1_medquad']:.4f}")
  print(f"\nLegalQAEval F1: {metrics['ft_f1_legalqaeval']:.4f}")
  print(f"\nEPubMedQA ROUGE (L): {metrics['ft_rouge_pubmedqa']['rougeL']:.4f}")
  print(f"\nMedQUAD ROUGE (L): {metrics['ft_rouge_medquad']['rougeL']:.4f}")
  print(f"\nLegalQAEval ROUGE (L): {metrics['ft_rouge_legalqaeval']['rougeL']:.4f}")

  print(f"\n\nImprovements after Supervised Fine Tuning:")
  print(f"\n PubMedQA F1: {(metrics['ft_f1_pubmedqa'] - metrics['bl_f1_pubmedqa'])/ metrics['bl_f1_pubmedqa']*100:.2f}%")
  print(f"\n MedQUAD F1: {(metrics['ft_f1_medquad'] - metrics['bl_f1_medquad'])/ metrics['bl_f1_medquad']*100:.2f}%")
  print(f"\n LegalQAEval F1: {(metrics['ft_f1_legalqaeval'] - metrics['bl_f1_legalqaeval'])/ metrics['bl_f1_legalqaeval']*100:.2f}%")
  print(f"\n PubMedQA ROUGE (L): {(metrics['ft_rouge_pubmedqa']['rougeL'] - metrics['bl_rouge_pubmedqa']['rougeL'])/ metrics['bl_rouge_pubmedqa']['rougeL']*100:.2f}%")
  print(f"\n MedQUAD ROUGE (L): {(metrics['ft_rouge_medquad']['rougeL'] - metrics['bl_rouge_medquad']['rougeL'])/ metrics['bl_rouge_medquad']['rougeL']*100:.2f}%")
  print(f"\n LegalQAEval ROUGE (L): {(metrics['ft_rouge_legalqaeval']['rougeL'] - metrics['bl_rouge_legalqaeval']['rougeL'])/ metrics['bl_rouge_legalqaeval']['rougeL']*100:.2f}%")

def plot_metrics_comparison(metrics, model, ft_type, dataset):
    # The datasets we evaluated
    datasets = ['PubMedQA', 'MedQUAD', 'LegalQAEval']

    # Extract F1 scores
    bl_f1 = [metrics['bl_f1_pubmedqa'], metrics['bl_f1_medquad'], metrics['bl_f1_legalqaeval']]
    ft_f1 = [metrics['ft_f1_pubmedqa'], metrics['ft_f1_medquad'], metrics['ft_f1_legalqaeval']]

    # Extract ROUGE-L scores (remembering to grab the 'rougeL' key from the evaluate dict)
    bl_rouge = [
        metrics['bl_rouge_pubmedqa']['rougeL'],
        metrics['bl_rouge_medquad']['rougeL'],
        metrics['bl_rouge_legalqaeval']['rougeL']
    ]
    ft_rouge = [
        metrics['ft_rouge_pubmedqa']['rougeL'],
        metrics['ft_rouge_medquad']['rougeL'],
        metrics['ft_rouge_legalqaeval']['rougeL']
    ]

    # Chart setup
    x = np.arange(len(datasets))
    width = 0.35  # width of the bars

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Performance Comparison: {model} with {ft_type} trained on {dataset}", fontsize=16, fontweight='bold')

    # --- Plot 1: F1 Scores ---
    rects1 = ax1.bar(x - width/2, bl_f1, width, label='Baseline', color='#4C72B0')
    rects2 = ax1.bar(x + width/2, ft_f1, width, label='Finetuned', color='#55A868')
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score: Baseline vs Finetuned', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=11)
    ax1.legend()
    ax1.set_ylim(0, max(max(bl_f1), max(ft_f1)) * 1.2) # Give headroom for labels

    # --- Plot 2: ROUGE-L Scores ---
    rects3 = ax2.bar(x - width/2, bl_rouge, width, label='Baseline', color='#4C72B0')
    rects4 = ax2.bar(x + width/2, ft_rouge, width, label='Finetuned', color='#55A868')
    ax2.set_ylabel('ROUGE-L Score', fontsize=12)
    ax2.set_title('ROUGE-L Score: Baseline vs Finetuned', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=11)
    ax2.legend()
    ax2.set_ylim(0, max(max(bl_rouge), max(ft_rouge)) * 1.2)

    # Helper function to attach text labels above each bar
    def autolabel(ax, rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    # Apply labels to all bars
    autolabel(ax1, rects1)
    autolabel(ax1, rects2)
    autolabel(ax2, rects3)
    autolabel(ax2, rects4)

    # Render the chart cleanly
    plt.tight_layout()
    plt.show()


def plot_percentage_improvements(metrics, model, ft_type, dataset):
    datasets = ['PubMedQA', 'MedQUAD', 'LegalQAEval']

    # 1. Calculate F1 Improvements ((Finetuned - Baseline) / Baseline * 100)
    f1_imp = [
        (metrics['ft_f1_pubmedqa'] - metrics['bl_f1_pubmedqa']) / metrics['bl_f1_pubmedqa'] * 100,
        (metrics['ft_f1_medquad'] - metrics['bl_f1_medquad']) / metrics['bl_f1_medquad'] * 100,
        (metrics['ft_f1_legalqaeval'] - metrics['bl_f1_legalqaeval']) / metrics['bl_f1_legalqaeval'] * 100
    ]

    # 2. Calculate ROUGE-L Improvements
    rouge_imp = [
        (metrics['ft_rouge_pubmedqa']['rougeL'] - metrics['bl_rouge_pubmedqa']['rougeL']) / metrics['bl_rouge_pubmedqa']['rougeL'] * 100,
        (metrics['ft_rouge_medquad']['rougeL'] - metrics['bl_rouge_medquad']['rougeL']) / metrics['bl_rouge_medquad']['rougeL'] * 100,
        (metrics['ft_rouge_legalqaeval']['rougeL'] - metrics['bl_rouge_legalqaeval']['rougeL']) / metrics['bl_rouge_legalqaeval']['rougeL'] * 100
    ]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Relative Improvement After Fine-Tuning\n({model} with {ft_type}) on {dataset}", fontsize=16, fontweight='bold')

    # 3. Plot the bars
    rects1 = ax.bar(x - width/2, f1_imp, width, label='F1 Improvement (%)', color='#2ca02c') # Green
    rects2 = ax.bar(x + width/2, rouge_imp, width, label='ROUGE-L Improvement (%)', color='#ff7f0e') # Orange

    # 4. Formatting
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend()

    # Draw a solid line at 0% to anchor the chart
    ax.axhline(0, color='black', linewidth=1.2)

    # 5. Helper function to attach percentage text labels (+ and - aware)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            # If negative, place the text below the bar
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3

            ax.annotate(f'{height:+.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va=va, fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # Dynamically scale the y-axis so labels don't get cut off
    all_vals = f1_imp + rouge_imp
    y_max = max(max(all_vals) * 1.2, 5) # Minimum 5% ceiling
    y_min = min(min(all_vals) * 1.2, 0) # Floor at 0 or lowest negative
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()
##set up trainer
from trl import SFTConfig, SFTTrainer
def set_trainer(model, tokenizer, dataset):
  return SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        # num_train_epochs = 1, # For longer training runs!
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
)

from unsloth import FastLanguageModel, apply_chat_template
max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

##training and evaluation loop
def train_domain_expert(model, dataset, output_adapter_name, qlora ):
    if not qlora: max_sequence_lenght = 1024  # Reduce memory usage on GPU by reducing sequence length for LoRA
    metrics = {}
    print(f"--- Starting training for {output_adapter_name} ---")

    # Load a fresh instance of the base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = qlora,
        load_in_16bit = not qlora
    )

    # Attach your LoRA configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16 if qlora else 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # Apply chat template
    dataset = apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
    # default_system_message = "You are a helpful assistant", << [OPTIONAL]
)
    # Gather test sets of 50
    test_set_pubmedqa = pubmedqa_dataset.shuffle(seed=42).select(range(50))
    test_set_medquad = medquad_dataset.shuffle(seed=42).select(range(50))
    test_set_legalqaeval = legalqaeval_dataset.shuffle(seed=42).select(range(50))
    # # Get Baseline metrics
    metrics["bl_f1_pubmedqa"] = run_f1_evaluation(model, tokenizer, test_set_pubmedqa, True)
    metrics["bl_f1_medquad"] =run_f1_evaluation(model, tokenizer, test_set_medquad , True)
    metrics["bl_f1_legalqaeval"] = run_f1_evaluation(model, tokenizer, test_set_legalqaeval, True)
    metrics["bl_rouge_pubmedqa"] = run_rouge_evaluation(model, tokenizer, "scientific_papers", 50)
    metrics["bl_rouge_medquad"] =run_rouge_evaluation(model, tokenizer, "scientific_papers", 50)
    metrics["bl_rouge_legalqaeval"] = run_rouge_evaluation(model, tokenizer, "billsum", 50)

    #Train the model
    trainer = set_trainer(model, tokenizer, dataset)
    trainer.train()

    # Get finetuned metrics
    metrics["ft_f1_pubmedqa"] = run_f1_evaluation(model, tokenizer, test_set_pubmedqa, True)
    metrics["ft_f1_medquad"] =run_f1_evaluation(model, tokenizer, test_set_medquad , True)
    metrics["ft_f1_legalqaeval"] = run_f1_evaluation(model, tokenizer, test_set_legalqaeval, True)
    metrics["ft_rouge_pubmedqa"] = run_rouge_evaluation(model, tokenizer, "scientific_papers", 50)
    metrics["ft_rouge_medquad"] =run_rouge_evaluation(model, tokenizer, "scientific_papers", 50)
    metrics["ft_rouge_legalqaeval"] = run_rouge_evaluation(model, tokenizer, "billsum", 50)
    # Save ONLY the tiny LoRA adapter (this is what you push to Hugging Face later)
    model.save_pretrained(output_adapter_name)
    tokenizer.save_pretrained(output_adapter_name)

    # Nuke the model from VRAM and flush the cache!
    del model
    del tokenizer
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"--- Finished {output_adapter_name}. VRAM cleared! ---")
    return metrics

## Function to train and print metrics 
def train_and_evaluate(model, dataset,output_adapter_name, qlora, ft_type, dataset_str ):
  metrics = train_domain_expert(model, dataset, output_adapter_name, qlora)
  print_metrics(metrics, model, ft_type, dataset_str)
  plot_metrics_comparison(metrics, model, ft_type, dataset_str)
  plot_percentage_improvements(metrics, model, ft_type, dataset_str)

## Start training
## QLoRA 
train_and_evaluate("unsloth/Phi-3-mini-4k-instruct-bnb-4bit", pubmedqa_dataset, "phi3_qlora_pubmedqa",True,"QLoRA", "PubMedQA" )
train_and_evaluate("unsloth/Phi-3-mini-4k-instruct-bnb-4bit", medquad_dataset, "phi3_qlora_medquad",True,"QLoRA", "MedQUAD" )
train_and_evaluate("unsloth/Phi-3-mini-4k-instruct-bnb-4bit", legalqaeval_dataset, "phi3_qlora_legalqaeval",True, "QLoRA", "LegalQAEval")
train_and_evaluate("unsloth/llama-3-8b-Instruct-bnb-4bit", pubmedqa_dataset, "llama3_qlora_pubmedqa",True, "QLoRA", "PubMedQA")
train_and_evaluate("unsloth/llama-3-8b-Instruct-bnb-4bit", medquad_dataset, "llama3_qlora_medquad", True, "QLoRA", "MedQUAD")
train_and_evaluate("unsloth/llama-3-8b-Instruct-bnb-4bit", legalqaeval_dataset, "llama3_qlora_legalqaeval", True, "QLoRA", "LegalQAEval")
train_and_evaluate("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", pubmedqa_dataset, "mistral_qlora_pubmedqa",True, "QLoRA", "PubMedQA")
train_and_evaluate("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", medquad_dataset, "mistral_qlora_medquad",True, "QLoRA", "MedQUAD")
train_and_evaluate("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", legalqaeval_dataset, "mistral_qlora_legalqaeval",True, "QLoRA", "LegalQAEval")

##LoRA
train_and_evaluate("unsloth/Phi-3-mini-4k-instruct", pubmedqa_dataset, "phi3_lora_pubmedqa",False, "LoRA", "PubMedQA")
train_and_evaluate("unsloth/Phi-3-mini-4k-instruct", medquad_dataset, "phi3_lora_medquad",False, "LoRA", "MedQUAD")
train_and_evaluate("unsloth/Phi-3-mini-4k-instruct", legalqaeval_dataset, "phi3_lora_legalqaeval",False, "LoRA", "LegalQAEval")
train_and_evaluate("unsloth/llama-3-8b-Instruct", pubmedqa_dataset, "llama3_lora_pubmedqa",False, "LoRA", "PubMedQA")
train_and_evaluate("unsloth/llama-3-8b-Instruct", medquad_dataset, "llama3_lora_medquad", False, "LoRA", "MedQUAD")
train_and_evaluate("unsloth/llama-3-8b-Instruct", legalqaeval_dataset, "llama3_lora_legalqaeval", False, "LoRA", "LegalQAEval")
train_and_evaluate("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", pubmedqa_dataset, "mistral_lora_pubmedqa",False, "LoRA", "PubMedQA")
train_and_evaluate("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", medquad_dataset, "mistral_lora_medquad",False, "LoRA", "MedQUAD")
train_and_evaluate("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", legalqaeval_dataset, "mistral_lora_legalqaeval",False, "LoRA", "LegalQAEval")