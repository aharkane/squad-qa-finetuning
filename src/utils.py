"""Utility functions: HF Hub interactions, helpers."""

from huggingface_hub import HfApi


def push_model_to_hub(local_path, repo_id, token=None):
    """Push a fine-tuned model to HuggingFace Hub.
    
    Args:
        local_path: path to the local model directory (with config.json, weights, etc.)
        repo_id: target HF repo (e.g. "aharkane/distilbert-squad")
        token: HF write token (if None, uses cached login from notebook_login)
    
    Returns:
        URL of the pushed repo
    """
    api = HfApi(token=token)
    
    # 1st create the repo if it doesn't exist (no error if it exists)
    api.create_repo(repo_id=repo_id, exist_ok=True)
    
    # 2nd upload all files from the local folder
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
    )
    
    url = f"https://huggingface.co/{repo_id}"
    print(f"Model pushed to: {url}")
    return url