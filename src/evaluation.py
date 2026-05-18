"""Evaluation pipeline for fine-tuned QA models.

Contains:
    - postprocess_qa_predictions: convert model logits into text predictions
    - evaluate_model: end-to-end evaluation (F1, EM, inference time)
"""

import time
import collections
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import evaluate

from src.data_processing import prepare_features_valid


def postprocess_qa_predictions(raw_dataset, features, raw_predictions,
                                n_best_size=20, max_answer_length=30):
    """Convert model logits into final text predictions."""
    
    # init
    expl_id_to_idx = {}
    features_per_expl = collections.defaultdict(list)
    predictions = collections.OrderedDict()
    
    # 1st unpack the predictions from the model
    all_start_logits, all_end_logits = raw_predictions
    
    # 2nd build a mapping: example_id → its index in the dataset
    expl_id_to_idx = {k: i for i, k in enumerate(raw_dataset["id"])}
    
    # 3rd group windows by example
    for feat_idx, feat in enumerate(features):
        expl_idx = expl_id_to_idx[feat["example_id"]]
        features_per_expl[expl_idx].append(feat_idx)
    
    # 4th iterate over each example
    for expl_idx, expl in enumerate(tqdm(raw_dataset)):
        # init for each example
        valid_aswrs = []
        feat_indices = features_per_expl[expl_idx]
        context = expl["context"]
        
        # 5th iterate over each window of the current example
        for feat_idx in feat_indices:
            start_logits = all_start_logits[feat_idx]
            end_logits = all_end_logits[feat_idx]
            wrds_map = features[feat_idx]["offset_mapping"]
            
            # 6th get the top-N start/end positions
            top_start_idx = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            top_end_idx = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            
            # 7th try every (start, end) combination
            for start_idx in top_start_idx:
                for end_idx in top_end_idx:
                    
                    # skip if out of bounds or outside context
                    if (start_idx >= len(wrds_map)
                        or end_idx >= len(wrds_map)
                        or wrds_map[start_idx] is None
                        or wrds_map[end_idx] is None):
                        continue
                    
                    # skip if end before start or answer too long
                    if end_idx < start_idx or end_idx - start_idx + 1 > max_answer_length:
                        continue
                    
                    # extract the text using char offsets
                    aswr_charPos_start = wrds_map[start_idx][0]
                    aswr_charPos_end = wrds_map[end_idx][1]
                    
                    valid_aswrs.append({
                        "score": start_logits[start_idx] + end_logits[end_idx],
                        "text": context[aswr_charPos_start:aswr_charPos_end],
                    })
        
        # 8th pick the best answer for this example
        if len(valid_aswrs) > 0:
            best_aswr = sorted(valid_aswrs, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_aswr = {"text": "", "score": 0.0}
        
        # 9th store the prediction keyed by the SQuAD example_id
        predictions[expl["id"]] = best_aswr["text"]
    
    return predictions


def evaluate_model(model_id, raw_validation, batch_size=32):
    """Evaluate a fine-tuned QA model: F1, EM, inference time, parameters."""
    
    # init
    all_start_logits = []
    all_end_logits = []
    
    # Auto-detect device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1st load the fine-tuned model and tokenizer from HF Hub or local path
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id).to(device)
    model.eval()
    
    # 2nd tokenize the validation set with our function
    valid_features = raw_validation.map(
        lambda x: prepare_features_valid(x, tokenizer),
        batched=True,
        remove_columns=raw_validation.column_names,
    )
    
    # 3rd prepare features for inference: drop columns the model doesn't need
    valid_for_inference = valid_features.remove_columns(["example_id", "offset_mapping"])
    valid_for_inference.set_format("torch")
    
    # 4th run inference batch by batch (no gradient computation for speed)
    inference_start = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_for_inference), batch_size)):
            batch = valid_for_inference[i : i + batch_size]
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # remove token_type_ids for DistilBERT (model doesn't support them)
            if "distilbert" in model.config.model_type.lower():
                batch.pop("token_type_ids", None)
            
            # forward pass: model predicts start/end logits for each token
            outputs = model(**batch)
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())
    inference_time = time.time() - inference_start
    
    # 5th concatenate logits from all batches
    start_logits = np.concatenate(all_start_logits, axis=0)
    end_logits = np.concatenate(all_end_logits, axis=0)
    
    # 6th post-process: convert logits into predicted text answers
    final_predictions = postprocess_qa_predictions(
        raw_validation, valid_features, (start_logits, end_logits),
    )
    
    # 7th format predictions and references for the SQuAD metric
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in raw_validation]
    
    # 8th compute F1 and Exact Match scores
    metric = evaluate.load("squad")
    results = metric.compute(predictions=formatted_predictions, references=references)
    
    # 9th return all metrics in a single dict
    return {
        "Model": model_id.split("/")[-1],
        "F1": round(results["f1"], 2),
        "Exact Match": round(results["exact_match"], 2),
        "Total time (s)": round(inference_time, 1),
        "Time/question (ms)": round((inference_time / len(raw_validation)) * 1000, 2),
        "Parameters (M)": round(sum(p.numel() for p in model.parameters()) / 1_000_000, 1),
    }