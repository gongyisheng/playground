import logging

import datasets
from datasets import DatasetDict, concatenate_datasets

from configs import ScriptArguments


logger = logging.getLogger(__name__)


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    Args:
        args (ScriptArguments): Script arguments containing dataset configuration.

    Returns:
        DatasetDict: The loaded datasets.
    """
    if args.dataset_name and not args.dataset_mixture:
        logger.info(f"Loading dataset: {args.dataset_name}")
        dataset = datasets.load_dataset(args.dataset_name, args.dataset_config)

        # Apply sampling if specified
        if args.dataset_sample_ratio is not None:
            sampled_dataset = {}
            for split_name, split_data in dataset.items():
                num_samples = int(len(split_data) * args.dataset_sample_ratio)
                sampled_dataset[split_name] = split_data.shuffle().select(range(num_samples))
                logger.info(
                    f"Sampled {split_name} split with ratio={args.dataset_sample_ratio}: "
                    f"{len(split_data)} -> {num_samples} examples"
                )
            return DatasetDict(sampled_dataset)

        return dataset
    elif args.dataset_mixture:
        logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")
        seed = args.dataset_mixture.seed
        datasets_list = []

        for dataset_config in args.dataset_mixture.datasets:
            logger.info(f"Loading dataset for mixture: {dataset_config.id} (config: {dataset_config.config})")
            ds = datasets.load_dataset(
                dataset_config.id,
                dataset_config.config,
                split=dataset_config.split,
            )

            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)
            if dataset_config.weight is not None:
                ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
                logger.info(
                    f"Subsampled dataset '{dataset_config.id}' (config: {dataset_config.config}) with weight={dataset_config.weight} to {len(ds)} examples"
                )

            datasets_list.append(ds)

        if datasets_list:
            combined_dataset = concatenate_datasets(datasets_list)
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

            if args.dataset_mixture.test_split_size is not None:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size, seed=seed
                )
                logger.info(
                    f"Split dataset into train and test sets with test size: {args.dataset_mixture.test_split_size}"
                )

                # Apply sampling if specified
                if args.dataset_sample_ratio is not None:
                    sampled_dataset = {}
                    for split_name, split_data in combined_dataset.items():
                        num_samples = int(len(split_data) * args.dataset_sample_ratio)
                        sampled_dataset[split_name] = split_data.shuffle().select(range(num_samples))
                        logger.info(
                            f"Sampled {split_name} split with ratio={args.dataset_sample_ratio}: "
                            f"{len(split_data)} -> {num_samples} examples"
                        )
                    return DatasetDict(sampled_dataset)

                return combined_dataset
            else:
                dataset_dict = DatasetDict({"train": combined_dataset})

                # Apply sampling if specified
                if args.dataset_sample_ratio is not None:
                    num_samples = int(len(combined_dataset) * args.dataset_sample_ratio)
                    sampled_train = combined_dataset.shuffle().select(range(num_samples))
                    logger.info(
                        f"Sampled train split with ratio={args.dataset_sample_ratio}: "
                        f"{len(combined_dataset)} -> {num_samples} examples"
                    )
                    return DatasetDict({"train": sampled_train})

                return dataset_dict
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")