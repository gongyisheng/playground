from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os

def build_and_save_model(
    model_name: str ,
    save_path: str,
    config_updates: dict = None
):
    """
    Load a model from HuggingFace, modify its config, and save locally.

    Args:
        model_name: HuggingFace model identifier
        save_path: Local path to save the modified model
        config_updates: Dictionary of config parameters to update
    """
    print(f"Loading model: {model_name}")

    # Load config
    config = AutoConfig.from_pretrained(model_name)
    print(f"Original config: {config}")

    # Update config if modifications are provided
    if config_updates:
        print(f"\nApplying config updates: {config_updates}")
        for key, value in config_updates.items():
            setattr(config, key, value)
        print(f"Modified config: {config}")

    # Load model with modified config
    print(f"\nLoading model with modified config...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config
    )

    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save model, tokenizer, and config
    print(f"\nSaving model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    config.save_pretrained(save_path)

    print(f"Model saved successfully to {save_path}")
    print(f"Config saved to {os.path.join(save_path, 'config.json')}")


def push_to_hub(
    model_path: str,
    repo_id: str,
    token: str = None,
    private: bool = False,
    commit_message: str = "Upload model"
):
    """
    Push a locally saved model to Hugging Face Hub.

    Args:
        model_path: Local path where the model is saved
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        token: HuggingFace API token (if None, uses token from huggingface-cli login)
        private: Whether to create a private repository
        commit_message: Commit message for the upload
    """
    print(f"\nPushing model from {model_path} to {repo_id}...")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Push to hub
    print(f"Uploading model...")
    model.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private,
        commit_message=commit_message
    )

    print(f"Uploading tokenizer...")
    tokenizer.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private,
        commit_message=commit_message
    )

    print(f"Model successfully pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":

    # Update RoPE theta to 300k

    config_modifications = {
        "rope_theta": 300000,
    }

    build_and_save_model(
        model_name="Qwen/Qwen2.5-Math-1.5B",
        save_path="./modified_qwen_model",
        config_updates=config_modifications
    )

    push_to_hub(
        model_path="./modified_qwen_model",
        repo_id="gongyisheng/Qwen2.5-Math-1.5B-RoPE-300k",
        private=False
    )
