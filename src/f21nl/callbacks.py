from pathlib import Path
from typing import Literal

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


class NMTCallbacks:
    def __new__(
        cls,
        checkpoint_dirpath: Path = Path("storage/checkpoints"),
        use_bleu: bool = True,
        use_early_stopping: bool = False,
        early_stopping_monitor: Literal["valid_loss", "valid_bleu_score"] = "valid_loss",
        eary_stopping_mode: Literal["min", "max"] = "min",
    ) -> list[Callback]:
        """Returns a list of callbacks to be used during training.

        Here you can add basically any other callback you want.
        """
        callbacks = []

        if use_early_stopping:
            callbacks.append(
                EarlyStopping(monitor=early_stopping_monitor, mode=eary_stopping_mode)
            )

        model_filename = (
            "{epoch}_{valid_loss:.2f}_{valid_bleu_score:.4f}"
            if use_bleu
            else "{epoch}_{valid_loss:.4f}"
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dirpath,
                filename=model_filename,
                monitor=early_stopping_monitor,
                mode=eary_stopping_mode,
                save_top_k=1,
            )
        )

        return callbacks
