from datasets import load_dataset, DatasetDict

from config import Seq2SeqConfig

def get_dataset(config: Seq2SeqConfig) -> DatasetDict:
    dataset = load_dataset(path=config.dataset_path, name=config.dataset_name)
    return dataset

if __name__ == "__main__":
    config = Seq2SeqConfig()
    dataset = get_dataset(config)