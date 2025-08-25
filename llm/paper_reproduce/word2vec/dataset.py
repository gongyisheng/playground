from datasets import load_dataset

from config import Word2VecConfig


class Word2VecDataset:
    def __init__(
        self,
        dataset_path: str = Word2VecConfig.dataset_path,
        dataset_name: str = Word2VecConfig.dataset_name,
    ):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

    def load(self):
        return load_dataset(self.dataset_path, self.dataset_name)


if __name__ == "__main__":
    dataset = Word2VecDataset().load()
    print(dataset)
