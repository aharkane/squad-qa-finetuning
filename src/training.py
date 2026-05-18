"""Fine-tuning pipeline for QA models.

Contains:
    - fine_tune_model: end-to-end fine-tuning of one model on SQuAD
"""

import os
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from src.config import (
    MODELS_PATH, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY
)
from src.data_processing import prepare_features_train


def fine_tune_model(model_name, model_checkpoint, raw_datasets,
                    num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                    learning_rate=LEARNING_RATE, warmup_steps=0):
    """Fine-tune a QA model on SQuAD and save it locally.
    
    Args:
        model_name: short name (e.g. "distilbert")
        model_checkpoint: HF identifier (e.g. "distilbert-base-uncased")
        raw_datasets: DatasetDict loaded from HF datasets
        num_epochs: number of training epochs
        batch_size: per-device training batch size
        learning_rate: optimizer learning rate
        warmup_steps: number of warmup steps (useful for hard-to-converge models)
    
    Returns:
        Trained Trainer instance
    """
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning: {model_name} ({model_checkpoint})")
    print(f"Epochs: {num_epochs} | LR: {learning_rate} | Warmup: {warmup_steps}")
    print(f"{'='*60}")
    
    # 1st load the pretrained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    
    # 2nd tokenize the training set with our function
    print("Tokenizing the dataset...")
    train_dataset = raw_datasets["train"].map(
        lambda x: prepare_features_train(x, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    
    # 3rd configure training arguments
    output_dir = os.path.join(MODELS_PATH, f"{model_name}-checkpoints")
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        fp16=True,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        report_to="none",
    )
    
    # 4th create the Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )
    
    # 5th run the training
    trainer.train()
    
    # 6th save the final model and tokenizer
    final_dir = os.path.join(MODELS_PATH, f"{model_name}-final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved at: {final_dir}")
    
    return trainer