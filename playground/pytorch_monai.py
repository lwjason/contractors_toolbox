import os
import pathlib

import monai
import pandas as pd
import torch
from monai.data import DataLoader, Dataset, decollate_batch, PersistentDataset
from monai.metrics import ROCAUCMetric
from monai.transforms import LoadImaged, AddChanneld, Spacingd, Orientationd, EnsureTyped, Compose, Resized, EnsureType, \
    Activations, AsDiscrete
from torch.utils.tensorboard import SummaryWriter

from toolbox.monai import DicomSeries3DReader

INPUT_DIR = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
T1 = "T1w"
T2 = "T2w"
T1GD = "T1wCE"
FLAIR = "FLAIR"
ALL_SEQUENCES = [T1, T2, T1GD, FLAIR]
LABEL = "label"


def gen_data_dicts(df, mode):
    data_list = []
    for _, row in df.iterrows():
        subject_id = row["BraTS21ID"]
        label = row["MGMT_value"]
        data_list.append({
            T1: os.path.join(INPUT_DIR, mode, subject_id, T1),
            T2: os.path.join(INPUT_DIR, mode, subject_id, T2),
            T1GD: os.path.join(INPUT_DIR, mode, subject_id, T1GD),
            FLAIR: os.path.join(INPUT_DIR, mode, subject_id, FLAIR),
            LABEL: label
        })
    return data_list


def get_transforms(sequence):
    train_transforms = [
        LoadImaged(keys=[sequence], reader=DicomSeries3DReader()),
        AddChanneld(keys=[sequence]),
        Spacingd(
            keys=[sequence],
            pixdim=1.0,
            mode=("bilinear"),
        ),
        Orientationd(keys=[sequence], axcodes="RAS"),
        Resized(keys=[sequence], spatial_size=(96, 96, 96)),
        EnsureTyped(keys=[sequence]),
    ]
    val_transforms = [
        LoadImaged(keys=[sequence], reader=DicomSeries3DReader()),
        AddChanneld(keys=[sequence]),
        Spacingd(
            keys=[sequence],
            pixdim=1.0,
            mode=("bilinear"),
        ),
        Orientationd(keys=[sequence], axcodes="RAS"),
        Resized(keys=[sequence], spatial_size=(96, 96, 96)),
        EnsureTyped(keys=[sequence]),
    ]
    return Compose(train_transforms), Compose(val_transforms)


def run(train_files, val_files, sequence=FLAIR, root_dir="."):
    train_trans, val_trans = get_transforms(sequence=sequence)
    post_pred = Compose([EnsureType(), Activations(softmax=True)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=2)])

    persistent_cache = pathlib.Path(root_dir, "persistent_cache")
    persistent_cache.mkdir(parents=True, exist_ok=True)

    # Define dataset, data loader
    train_ds = PersistentDataset(
        data=train_files,
        transform=train_trans,
        cache_dir=persistent_cache
    )
    val_ds = PersistentDataset(
        data=val_files,
        transform=val_trans,
        cache_dir=persistent_cache
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    check_ds = Dataset(data=train_files, transform=train_trans)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=4)
    check_data = monai.utils.misc.first(check_loader)
    print(check_data[sequence].shape, check_data["label"])

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    auc_metric = ROCAUCMetric()

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter()
    for epoch in range(5):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[sequence].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data[sequence].to(device), val_data["label"].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                y_onehot = [post_label(i) for i in decollate_batch(y)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_classification3d_dict.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == '__main__':
    df_train_labels = pd.read_csv(os.path.join(INPUT_DIR, "train_labels.csv"),
                                  dtype={"BraTS21ID": str, "MGMT_value": int})
    data_dicts = gen_data_dicts(df_train_labels, "train")
    # testing
    train_files, val_files = data_dicts[:10], data_dicts[10:15]

    run(train_files, val_files, sequence=FLAIR)
