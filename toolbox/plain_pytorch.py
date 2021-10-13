import os
import argparse
import SimpleITK
import time
import monai
import torch
import torchio as tio
import matplotlib.pyplot as plt
from tqdm import tqdm
from toolbox.dataset import RSNA_MICCAIBrainTumorDataset
from toolbox.landmarks import landmarks_dict
from toolbox.constants import *

SimpleITK.ProcessObject_SetGlobalWarningDisplay(False)


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

    dataset_dict  = {}
    for sequence in ALL_SEQUENCES:
        preprocess = tio.Compose([
            tio.ToCanonical(),
            tio.HistogramStandardization(landmarks_dict),
            tio.RescaleIntensity((-1, 1)),
            #tio.Resample(4.0),
            tio.CropOrPad((128, 128, 128)),
        ])
        import pathlib
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

        # specify our batch size, number of epochs, and learning rate
        BATCH_SIZE = 2
        EPOCHS = 50
        # determine the device we will be using for training
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print("[INFO] training using {}...".format(DEVICE))

        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2, dropout_prob=0.25).to(DEVICE)
        # initialize optimizer and loss function
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        lossFn = torch.nn.CrossEntropyLoss()

        trainDataLoader = dataset.train_dataloader()
        valDataLoader = dataset.val_dataloader()
        # calculate steps per epoch for training and validation set
        trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
        valSteps = len(valDataLoader.dataset) // BATCH_SIZE

        # initialize a dictionary to store training history
        H = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        # measure how long training is going to take
        print("[INFO] training the network...")
        startTime = time.time()

        # set the device we will be using to train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir = os.path.join(arguments.model_dir, sequence)
        os.makedirs(model_dir, exist_ok=True)
        # loop over our epochs
        for e in range(0, EPOCHS):
            # set the model in training mode
            model.train()
            # initialize the total training and validation loss
            totalTrainLoss = 0
            totalValLoss = 0
            # initialize the number of correct predictions in the training
            # and validation step
            trainCorrect = 0
            valCorrect = 0
            # loop over the training set
            import numpy as np
            for idx, subject in enumerate(tqdm(trainDataLoader)):
                x = subject[sequence]['data']
                y = subject['label']
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # perform a forward pass and calculate the training loss
                pred = model(x)
                loss = lossFn(pred, y)
                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far and
                # calculate the number of correct predictions
                totalTrainLoss += loss
                trainCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()
            # switch off autograd for evaluation
            with torch.no_grad():
                # set the model in evaluation mode
                model.eval()
                # loop over the validation set
                for idx, subject in enumerate(tqdm(valDataLoader)):
                    x = subject[sequence]['data']
                    y = subject['label']
                    # send the input to the device
                    (x, y) = (x.to(device), y.to(device))
                    # make the predictions and calculate the validation loss
                    pred = model(x)
                    totalValLoss += lossFn(pred, y)
                    # calculate the number of correct predictions
                    valCorrect += (pred.argmax(1) == y).type(
                        torch.float).sum().item()
            # calculate the average training and validation loss
            avgTrainLoss = totalTrainLoss / trainSteps
            avgValLoss = totalValLoss / valSteps
            # calculate the training and validation accuracy
            trainCorrect = trainCorrect / len(trainDataLoader.dataset)
            valCorrect = valCorrect / len(valDataLoader.dataset)
            # update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["train_acc"].append(trainCorrect)
            H["val_loss"].append(avgValLoss.cpu().detach().numpy())
            H["val_acc"].append(valCorrect)
            # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
            print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                avgTrainLoss, trainCorrect))
            print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
                    avgValLoss, valCorrect))

            # plot the training loss and accuracy
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(H["train_loss"], label="train_loss")
            plt.plot(H["val_loss"], label="val_loss")
            plt.plot(H["train_acc"], label="train_acc")
            plt.plot(H["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy on Dataset")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            plt.savefig(os.path.join(model_dir, "training_plot.png"))
            #print('[INFO]: Learning Rate History: ', lrs)
            # serialize the model to disk
            torch.save(model, os.path.join(model_dir, arguments.architecture + ".pth"))
if __name__ == "__main__":
    main()