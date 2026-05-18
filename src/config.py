"""Project configuration: paths, hyperparameters, model identifiers."""

import os


# === Paths ===
DIR_PATH = "/content/drive/MyDrive/DNF3ED04-Projet-SQuAD_FineTuning"
DATA_PATH = os.path.join(DIR_PATH, "data")
MODELS_PATH = os.path.join(DIR_PATH, "models")
NOTEBOOKS_PATH = os.path.join(DIR_PATH, "notebooks")
REPORTS_PATH = os.path.join(DIR_PATH, "reports")
SRC_PATH = os.path.join(DIR_PATH, "src")


# === Hyperparameters (used by prepare_features_train and prepare_features_valid) ===
MAX_LENGTH = 384       # max length of tokenized sequences
STRIDE = 128           # overlap between sliding windows (~1/3 of MAX_LENGTH)


# === Training arguments ===
NUM_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


# === Models to fine-tune (pretrained checkpoints from HF Hub) ===
MODEL_CHECKPOINTS = {
    "distilbert": "distilbert-base-uncased",
    "mobilebert": "google/mobilebert-uncased",
    "albert": "albert-base-v2",
}


# === Hugging Face Hub identifiers (where fine-tuned models will be pushed) ===
HF_USERNAME = "aharkane"
HF_REPOS = {
    "distilbert": f"{HF_USERNAME}/squad-distilbert-v2",
    "mobilebert": f"{HF_USERNAME}/squad-mobilebert-v2",
    "albert": f"{HF_USERNAME}/squad-albert-v2",
}


# === Local model directories (after fine-tuning, before pushing) ===
LOCAL_MODELS = {
    "distilbert": os.path.join(MODELS_PATH, "distilbert-final"),
    "mobilebert": os.path.join(MODELS_PATH, "mobilebert-final"),
    "albert": os.path.join(MODELS_PATH, "albert-final"),
}


# === Dataset ===
DATASET_NAME = "squad"


# === Display formatting (used in notebooks for bold output) ===
B = '\033[1m'
BE = '\033[0m'