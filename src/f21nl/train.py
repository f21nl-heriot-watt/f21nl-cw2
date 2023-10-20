import argparse
from pathlib import Path

from lightning import Trainer, seed_everything

from f21nl.callbacks import NMTCallbacks
from f21nl.datamodule import Multi30kDataModule
from f21nl.interfaces import DatasetSize, DecoderConfig, EncoderConfig, EncoderDecoderConfig
from f21nl.lightning_modules.nmt_seq2seq import NMTModule


def train(args: argparse.Namespace) -> None:
    """Runs the training of the model."""
    seed_everything(42, workers=True)

    config = EncoderDecoderConfig(
        encoder_config=EncoderConfig(
            encoder_cell_type=args.encoder_cell_type,
            encoder_input_dim=args.encoder_input_dim,
            encoder_output_dim=args.encoder_output_dim,
            source_embedding_dim=args.source_embedding_dim,
            bidirectional=args.bidirectional_encoder,
        ),
        decoder_config=DecoderConfig(
            decoder_cell_type=args.decoder_cell_type,
            decoder_input_dim=args.decoder_input_dim,
            decoder_output_dim=args.decoder_output_dim,
            target_embedding_dim=args.target_embedding_dim,
        ),
        use_blue=args.use_bleu,
        attention=args.attention,
    )

    dm = Multi30kDataModule(
        dataset_size=args.dataset_size,
        storage_dir=Path("storage/data/"),
        batch_size=args.batch_size,
    )

    dm.setup(stage="fit")

    model = NMTModule(
        config,
        source_tokenizer=dm.source_tokenizer,
        target_tokenizer=dm.target_tokenizer,
    )

    trainer = Trainer(
        max_epochs=args.train_epochs,
        callbacks=NMTCallbacks(
            checkpoint_dirpath=args.checkpoint_dirpath,
            use_early_stopping=args.use_early_stopping,
            early_stopping_monitor=args.early_stopping_monitor,
            eary_stopping_mode=args.eary_stopping_mode,
        ),
    )

    trainer.fit(model=model, datamodule=dm)


def parse_args() -> argparse.Namespace:
    """Parse any arguments."""
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--dataset_size",
        type=DatasetSize,
        choices=[DatasetSize.large, DatasetSize.small],
        default=DatasetSize.small,
        help="Dataset size",
    )

    arg_parser.add_argument(
        "--train_epochs",
        type=int,
        default=100,
        help="Number of epochs to train",
    )

    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )

    arg_parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers",
    )

    arg_parser.add_argument(
        "--use_bleu",
        action="store_true",
        help="Use BLEU as a metric",
    )

    arg_parser.add_argument(
        "--encoder_cell_type",
        type=str,
        choices=["gru", "lstm"],
        default="gru",
        help="Encoder cell type",
    )

    arg_parser.add_argument(
        "--source_embedding_dim",
        type=int,
        default=50,
        help="Source embedding dimension",
    )

    arg_parser.add_argument(
        "--target_embedding_dim",
        type=int,
        default=50,
        help="Target embedding dimension",
    )

    arg_parser.add_argument(
        "--encoder_input_dim",
        type=int,
        default=50,
        help="Encoder input dimension",
    )

    arg_parser.add_argument(
        "--encoder_output_dim",
        type=int,
        default=200,
        help="Encoder output dimension",
    )

    arg_parser.add_argument(
        "--bidirectional_encoder",
        action="store_true",
        help="Use bidirectional encoder. Make sure to adjust the output of the encoder and the output of the decoder accordingly.",
    )

    arg_parser.add_argument(
        "--decoder_cell_type",
        type=str,
        choices=["gru", "lstm"],
        default="gru",
        help="Decoder cell type",
    )

    arg_parser.add_argument(
        "--decoder_input_dim",
        type=int,
        default=50,
        help="decoder input dimension",
    )

    arg_parser.add_argument(
        "--decoder_output_dim",
        type=int,
        default=200,
        help="Decoder output dimension",
    )

    arg_parser.add_argument(
        "--attention",
        type=str,
        choices=["dot", "bilinear"],
    )

    arg_parser.add_argument(
        "--use_early_stopping",
        action="store_true",
        help="Use early stopping",
    )

    arg_parser.add_argument(
        "--early_stopping_monitor",
        type=str,
        choices=["valid_loss", "valid_bleu_score"],
        default="valid_loss",
        help="Early stopping monitor. Only used if early stopping is enabled.",
    )

    arg_parser.add_argument(
        "--eary_stopping_mode",
        type=str,
        choices=["min", "max"],
        default="min",
        help="Early stopping mode. Only used if early stopping is enabled.",
    )

    arg_parser.add_argument(
        "--checkpoint_dirpath",
        type=Path,
        default=Path("storage/checkpoints/"),
        help="Checkpoint directory path",
    )

    arg_parser.add_argument(
        "--decoding_strategy",
        type=str,
        choices=["greedy", "top_k", "top_p"],
        default="greedy",
        help="Decoding strategy",
    )

    arg_parser.add_argument(
        "--top_k",
        type=int,
        help="Top K words used in top K decoding strategy",
    )

    arg_parser.add_argument(
        "--top_p",
        type=int,
        help="Top P used in top P decoding strategy",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
