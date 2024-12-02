import os, argparse

import SimpleITK
import monai
from datetime import datetime
import torch
import torchio as tio
import torchmetrics
from monai.metrics import ROCAUCMetric
import pytorch_lightning as pl
from toolbox.dataset import RSNA_MICCAIBrainTumorDataset
from toolbox.landmarks import landmarks_dict
from toolbox.constants import *


SimpleITK.ProcessObject_SetGlobalWarningDisplay(False)


class Model(pl.LightningModule):
    def __init__(self, sequence, net, criterion, learning_rate, optimizer_class, lr_scheduler_class):
        """
        :param sequence - sequence type on which model will be trained on
        :param net - model architecture to train
        :learning_rate - initial learning rate
        :optimizer_class - torch optimizer
        """
        super().__init__()
        self.sequence = sequence
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.lr_scheduler_class = lr_scheduler_class
        self.accuracy = torchmetrics.Accuracy()
        self.auc = ROCAUCMetric()

    def configure_optimizers(self):
        """
        Even though model is initialized when passed to .fit() function, Calls to configure_optimizers function are
        lazy, which means that it won't actually be called untill it is needed for training.
        """
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss', }

    def prepare_batch(self, batch):
        """
        Prepares batches for training (i.e. batch['label'] and for testing.
        When testing we will encounter key error which will return batch
        without label since they are not provided.
        """
        try:
            return batch[self.sequence][tio.DATA], batch['label']
        except KeyError:
            return batch[self.sequence][tio.DATA]

    def infer_batch(self, batch):
        """
        Perform inference over a batch and return predicted value along with
        the ground truth.
        """
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        """
        Perform one training step. If learning rate schedulers are defined, one must manually call
        self.lr_schedulers().step(); this is required by PyTorch Lightning 1.3+ version.
        """
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform one validation step.
        """
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds, y)
        self.auc(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        auc_result = self.auc.aggregate()
        self.log("auc", auc_result, prog_bar=True, on_epoch=True)

    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Perform inference over the batch and return model prediction.
        """
        y, _ = self.infer_batch(batch)
        return y.argmax(dim=1)


def create_model(net,
                 architecture,
                 sequence_type,
                 criterion=torch.nn.CrossEntropyLoss(),
                 optimizer_class=torch.optim.AdamW,
                 lr_scheduler_class= None,
                 learning_rate=1e-2):
    """
    Instantiates PyTorch lightning custom Model class.

    :param net: neural network architecture which will be trained; instantiated using Monai framework
    :param architecture: name of the architecture which will be used for saving checkpoints
    :param sequence_type: type of data on which model is trained on
    :param criterion: loss function
    :param optimizer_class: class from torch.optim package which will be instantiated inside PyTorch Lightning class
    :param: lr_scheduler_class: class from torch.optim package which will be instantiated inside PyTorch Lightning class
    :param: learning_rate: initial learning rate
    :return: instance of PyTorch Lightning custom Model class
    """
    print(f"[INFO]: Creating new {architecture} model ...")
    # Change to desired learning rate scheduler, or leave as default (i.e. None)
    lr_scheduler_class = torch.optim.lr_scheduler.CyclicLR
    return Model(net=net,
                criterion=criterion,
                optimizer_class=optimizer_class,
                lr_scheduler_class=lr_scheduler_class,
                learning_rate=learning_rate,
                sequence=sequence_type,
    )


def create_trainer(checkpoint_dir,
                   filename,
                   gpu_count=1,
                   auto_lr_find=True):
    """
    Instantiates PyTorch Lightnign Trainer class.

    :param checkpoint_dir: where model checkpoints will be sasved
    :param filename: name of the checkpoints
    :param gpu_count: number of GPUs used for training
    :param auto_lr_find: flag whether or not to automatically find optimal learning rate
    :returns instance of PytorchLightning Trainer class
    """
    print("[INFO]: Initializing PytorchLightning Trainer ...")
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
    )
    model_checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename=filename,
        save_top_k=5,
        mode='min',
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        gpus=-1, auto_select_gpus=True,
        precision=16,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size="binsearch",
        min_epochs=10,
        max_epochs=100,
        #track_grad_norm=2, #L2 Norm
        callbacks=[early_stopping,
                   model_checkpoint,
                   lr_monitor]
    )
    return trainer


def train(net, sequence_type, dataset, architecture, checkpoint_dir):
    """
    Executes training procedure.

    :param net: instance of Monai framework neural network architecture which will be trained.
    :param sequence_type: type of data on which model wil be trained on
    :param dataset: instance of custom Kaggle dataset class
    :param architecture: neural network architecture name
    :param checkpoint_dir: absolute path to directory where model checkpoitns will be stored
    """
    model = create_model(net, architecture, sequence_type)
    trainer = create_trainer(checkpoint_dir=checkpoint_dir,
                             filename=sequence_type + f'{architecture}' + '-{epoch:02d}-{val_loss:.2f}')

    print("[INFO]: Looking for OPTIMAL Learning Rate ...")
    lr_finder = trainer.tuner.lr_find(model, dataset)
    model.hparams.learning_rate = lr_finder.suggestion()
    trainer.logger._default_hp_metric = False

    start = datetime.now()
    print(f"[INFO]: Training {sequence_type} model started at: ", start)
    trainer.fit(model=model, datamodule=dataset)
    print(f"[INFO]: Training {sequence_type} duration was: ", datetime.now() - start)


def init_argparse():
    """
    Initializes argument parser object with input data directory, where model checkpoints will be saved
    and which neural network architecture will be trained.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path where input data is"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Path where model checkpoints will be saved"
    )
    parser.add_argument(
        "--architecture", type=str, required=True, help="Directory name of the model architecture being trained"
    )
    return parser


def main():
    arguments = init_argparse().parse_args()

    if not os.path.exists(arguments.input_dir):
        raise Exception(f"{arguments.input_dir} does not exist")
    if not os.path.exists(arguments.model_dir):
        raise Exception(f"{arguments.model_dir} does not exist")

    dataset_dict  = {}
    for sequence in ALL_SEQUENCES:
        preprocess = tio.Compose([
            tio.ToCanonical(),
            tio.HistogramStandardization(landmarks_dict),
            tio.RescaleIntensity((-1, 1)),
            tio.Resample((1.0, 1.0, 1.0)),
            tio.Resize((128, 128, 64)),
        ])

        dataset = RSNA_MICCAIBrainTumorDataset(
            dataset_dir=arguments.input_dir,
            batch_size=2,
            train_label_csv=os.path.join(arguments.input_dir, "train_labels.csv"),
            train_val_ratio=0.8,
            preprocess=preprocess,
            sequence=[sequence],
            num_workers=2
        )

        dataset.setup()
        print(f"[INFO]: Number of Subjects in TRAINING Set: {len(dataset.train_set)}")
        print(f"[INFO]: Number of Subjects in VALIDATION Set: {len(dataset.val_set)}")
        print(f"[INFO]: Number of Subjects in TEST Set: {len(dataset.test_set)}\n")

        dataset_dict[sequence] = dataset

        print("[INFO]: Training ...")
        checkpoint_dir = os.path.join(arguments.model_dir, arguments.architecture, sequence)
        train(monai.networks.nets.DenseNet(spatial_dims=3, in_channels=1, out_channels=2),
                           sequence,
                           dataset,
                           arguments.architecture,
                           checkpoint_dir)


if __name__ == "__main__":
    main()