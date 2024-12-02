import os

import itk
import monai
import pytorch_lightning as pl

import toolbox.constants as C
from toolbox.dataset_monai import MonaiDatasetConfig, RSNA_MICCAI_MONAI_Dataset
from toolbox.lightning_monai import Model, ModelConfig

itk.ProcessObject.SetGlobalWarningDisplay(False)

sequences = [C.FLAIR, C.T1GD, C.T1, C.T2]
model_dir = "Monai"
params = "weightdecay=1e-5-dropout_prob=0.25"
architecture = "DenseNet121-Norm"

if __name__ == "__main__":
    pass

    config = MonaiDatasetConfig(
        sequences=sequences,
        dataset_dir="/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
        train_batch_size=4,
        cache_dir="cache"
    )
    dataset = RSNA_MICCAI_MONAI_Dataset(config)
    dataset.prepare_data()

    model_config = ModelConfig()
    net = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=4, out_channels=2, dropout_prob=0.10)
    # net = monai.networks.nets.EfficientNetBN("efficientnet-b1", spatial_dims=3, in_channels=4, num_classes=2)

    model = Model("ALL_SEQ", net, model_config)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
    )

    checkpoint_dir = os.path.join(model_dir, f"{architecture}-{params}", "All_SEQ")
    filename = f"All_SEQ-{architecture}-" + "{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-{auc:.2f}"

    model_checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename=filename,
        save_top_k=5,
        mode="min",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    logger = pl.loggers.TensorBoardLogger("tb_logs", name=f"{architecture}-{params}-ALL_SEQ")

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        auto_lr_find=True,
        min_epochs=20,
        max_epochs=100,
        log_every_n_steps=10,
        num_sanity_val_steps=20,
        # track_grad_norm=2, #L2 Norm
        callbacks=[early_stopping, model_checkpoint, lr_monitor],
        logger = logger,
    )

    trainer.fit(model=model, datamodule=dataset)