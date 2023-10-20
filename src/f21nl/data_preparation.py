import itertools
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from loguru import logger
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer


LanguageId = Literal["en", "fr"]

LARGE_CORPUS_NAME = "multi30k-en-fr-large"
SMALL_CORPUS_NAME = "multi30k-en-fr-small"

SMALL_CORPUS_NUM_TRAIN_INSTANCES = 750
SMALL_CORPUS_NUM_VALID_INSTANCES = 250

TRAIN_TOKENIZED_EN_URL = (
    "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/train.lc.norm.tok.en"
)
TRAIN_TOKENIZED_FR_URL = (
    "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/train.lc.norm.tok.fr"
)
VALID_TOKENIZED_EN_URL = (
    "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/val.lc.norm.tok.en"
)
VALID_TOKENIZED_FR_URL = (
    "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/val.lc.norm.tok.fr"
)

UNK_TOKEN = "[UNK]"  # noqa: S105
# beginning of sequence special token.
BOS_TOKEN = "[BOS]"  # noqa: S105
# end of sequence special token.
EOS_TOKEN = "[EOS]"  # noqa: S105


def download_file(url: str, output_dir: Path, *, overwrite_existing: bool = False) -> Path:
    """Download a file from the given URL and save it to the given destination."""
    # Figure out the destination
    file_name = Path(urlparse(url).path).name
    destination = output_dir.joinpath(file_name)

    if destination.exists() and not overwrite_existing:
        logger.debug(f"{destination} already exists. Skipping download...")
        return destination

    # Download the file
    with httpx.Client() as client:
        response = client.get(url)
        # If there was any issue, raise an exception
        response.raise_for_status()

        with destination.open("w", encoding="utf-8") as file:
            file.write(response.text)

        logger.debug(f"File downloaded successfully. Destination: {destination}")

    return destination


def create_small_corpus_from_large_corpus(
    large_corpus_dir: Path,
    small_corpus_dir: Path,
    *,
    num_train_instances: int = SMALL_CORPUS_NUM_TRAIN_INSTANCES,
    num_valid_instances: int = SMALL_CORPUS_NUM_VALID_INSTANCES,
    overwrite_existing: bool = False,
) -> None:
    """Create a small corpus from the large corpus."""
    # Create the small corpus dir
    small_corpus_dir.mkdir(parents=True, exist_ok=True)

    corpus_files = itertools.chain.from_iterable(
        large_corpus_dir.glob(f"{file_name}.*") for file_name in ["train", "val"]
    )

    for corpora_file in corpus_files:
        logger.debug(f"Creating small corpus from {corpora_file}...")
        output_file_path = small_corpus_dir.joinpath(corpora_file.name)

        if output_file_path.exists() and not overwrite_existing:
            logger.debug(f"{output_file_path} already exists. Skipping creation...")
            continue

        data_split = corpora_file.name.split(".")[0]
        num_instances = num_train_instances if data_split == "train" else num_valid_instances

        # Read the necessary lines from the file and write them to their relevant file
        with corpora_file.open("r", encoding="utf-8") as corpora_lines_file:
            instance_generator = (corpora_lines_file.readline() for _ in range(num_instances))

            with output_file_path.open("w") as output_file:
                output_file.writelines(instance_generator)


def build_dataset_vocab(
    dataset_path: Path,
    *,
    language_id: LanguageId,
    is_target: bool,
    overwrite_existing: bool = False,
) -> None:
    """Builds a tokenizer from the input dataset path."""
    tokenizer_path = dataset_path.parent.joinpath(f"tokenizer-{language_id}")

    if tokenizer_path.exists() and not overwrite_existing:
        logger.debug(f"Tokenizer {tokenizer_path} already exists. Skipping...")
        return

    # Create the tokenizer without a vocab since we're going to "train" one from the dataset
    tokenizer = Tokenizer(WordLevel(vocab=None, unk_token=UNK_TOKEN))

    # When tokenizing, split inputs on word boundaries before doing anything else
    tokenizer.pre_tokenizer = Whitespace()

    # 'Train' a simple tokenizer that splits on whitespace only from the corpus.
    # This will just create a vocabulary with the mappings between tokens and ids.
    trainer = WordLevelTrainer(special_tokens=[BOS_TOKEN, EOS_TOKEN, UNK_TOKEN], min_frequency=0)

    logger.debug(f"'Train' the tokenizer for {language_id}")
    with Path.open(dataset_path) as in_file:
        tokenizer.train_from_iterator(in_file.readlines(), trainer=trainer)

    if is_target:
        tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            special_tokens=[
                (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
                (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
            ],
        )

    logger.debug(f"Saving tokenizer to {tokenizer_path}")
    tokenizer.save(tokenizer_path.as_posix())


def run_data_preparation(storage_dir: Path, *, overwrite_existing: bool = False) -> None:
    """Run the data preparation."""
    # Create the large corpus dir
    large_corpus_dir = storage_dir.joinpath(LARGE_CORPUS_NAME)
    large_corpus_dir.mkdir(parents=True, exist_ok=True)

    # Create the small corpus dir
    small_corpus_dir = storage_dir.joinpath(SMALL_CORPUS_NAME)
    small_corpus_dir.mkdir(parents=True, exist_ok=True)

    # Download the tokenized corpora into the large corpus dir
    logger.info("Download the tokenized corpora")
    train_tokenized_en = download_file(
        TRAIN_TOKENIZED_EN_URL, large_corpus_dir, overwrite_existing=overwrite_existing
    )
    train_tokenized_fr = download_file(
        TRAIN_TOKENIZED_FR_URL, large_corpus_dir, overwrite_existing=overwrite_existing
    )
    download_file(VALID_TOKENIZED_EN_URL, large_corpus_dir, overwrite_existing=overwrite_existing)
    download_file(VALID_TOKENIZED_FR_URL, large_corpus_dir, overwrite_existing=overwrite_existing)

    # Create the small corpus
    logger.info("Create the small corpus")
    create_small_corpus_from_large_corpus(
        large_corpus_dir, small_corpus_dir, overwrite_existing=overwrite_existing
    )

    # build the vocabulary for the source language
    logger.info("Build the tokenizer for the languages")
    build_dataset_vocab(
        train_tokenized_en,
        language_id="en",
        is_target=False,
        overwrite_existing=overwrite_existing,
    )
    build_dataset_vocab(
        train_tokenized_fr, language_id="fr", is_target=True, overwrite_existing=overwrite_existing
    )


if __name__ == "__main__":
    storage_dir = Path().cwd().joinpath("storage", "data")
    run_data_preparation(storage_dir, overwrite_existing=True)
