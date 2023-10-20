import argparse
from pathlib import Path

from lightning import Trainer

from f21nl.datamodule import Multi30kDataModule
from f21nl.interfaces import DatasetSize, DecoderConfig, EncoderConfig, EncoderDecoderConfig
from f21nl.lightning_modules.nmt_seq2seq import NMTModule


def test(args: argparse.Namespace) -> None:
    """Runs the test of the model."""
    config = EncoderDecoderConfig(
        encoder_config=EncoderConfig(
            encoder_cell_type=args.encoder_cell_type,
            encoder_input_dim=args.encoder_input_dim,
            encoder_output_dim=args.encoder_output_dim,
            bidirectional=args.bidirectional_encoder,
        ),
        decoder_config=DecoderConfig(
            decoder_cell_type=args.decoder_cell_type,
            decoder_input_dim=args.decoder_input_dim,
            decoder_output_dim=args.decoder_output_dim,
        ),
        use_blue=True,
        attention=args.attention,
    )

    dm = Multi30kDataModule(
        dataset_size=args.dataset_size,
        storage_dir=Path("storage/data/"),
        batch_size=args.batch_size,
    )

    dm.setup("test")

    model = NMTModule.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        config=config,
        source_tokenizer=dm.source_tokenizer,
        target_tokenizer=dm.target_tokenizer,
    )

    trainer = Trainer()

    trainer.test(model=model, datamodule=dm)


def parse_args() -> argparse.Namespace:
    """Parse any arguments."""
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint to be loaded.",
    )

    arg_parser.add_argument(
        "--dataset_size",
        type=DatasetSize,
        choices=[DatasetSize.large, DatasetSize.small],
        default=DatasetSize.small,
        help="Dataset size",
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
        "--encoder_cell_type",
        type=str,
        choices=["gru", "lstm"],
        default="gru",
        help="Encoder cell type",
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
        default=400,
        help="Decoder output dimension",
    )

    arg_parser.add_argument(
        "--attention",
        type=str,
        choices=["dot", "bilinear"],
    )

    arg_parser.add_argument(
        "--max_decoder_steps",
        type=int,
        default=15,
        help="Decoder output dimension",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(args)
