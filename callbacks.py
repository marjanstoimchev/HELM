from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.callbacks.progress import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

class ModelCheckpoint_(ModelCheckpoint):
    def __init__(self, dirpath, metric="val_loss", mode="min", every_n_epochs=1):
        super().__init__(
            save_top_k=1,
            monitor=metric,
            mode=mode,
            dirpath=dirpath,
            save_weights_only=True,
            filename="model-{epoch:2d}-{val_loss:.2f}",
            every_n_epochs=every_n_epochs,
            save_last=False,
            verbose=False,
        )

class EarlyStopping_(EarlyStopping):
    def __init__(self, metric, mode, patience):
        super().__init__(
            monitor=metric,
            min_delta=0.00,
            patience=patience,
            mode=mode
        )

class RichProgressBar_(RichProgressBar):
    def __init__(self):
        super().__init__(
            leave=True,
            theme=RichProgressBarTheme(
                description="green_yellow",
                progress_bar="green1",
                progress_bar_finished="green1",
                progress_bar_pulse="#6206E0",
                batch_progress="green_yellow",
                time="grey82",
                processing_speed="grey82",
                metrics="grey82",
            )
        )