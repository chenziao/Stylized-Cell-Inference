import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.linear_model import LinearRegression
from typing import Optional, Tuple
import os
from joblib import dump, load

from cell_inference.config import paths


def train_regression(model: nn.Module, training_loader: DataLoader,
                     validation_loader: DataLoader,
                     epochs: int = 50,
                     learning_rate: float = 0.005,
                     decay_rate: float = 1.0,
                     device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
    epochs_list = []
    train_loss_list = []
    val_loss_list = []

    # splitting the dataloaders to generalize code
    data_loaders = {"train": training_loader, "val": validation_loader}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    decay_rate = decay_rate
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

    for epoch in tqdm(range(epochs), position=0, leave=True):
        train_loss = 0.0
        val_loss = 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            for i, (x, y) in enumerate(data_loaders[phase]):
                x = x.to(device)
                output = model(x)
                y = y.to(device)
                loss = loss_func(torch.squeeze(output), torch.squeeze(y))
                # backprop
                optimizer.zero_grad()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # calculating total loss
                running_loss += loss.item()

            if phase == 'train':
                train_loss = running_loss
                lr_sch.step()
            else:
                val_loss = running_loss

        epochs_list.append(epoch)
        if epoch % 10 == 0:
            tqdm.write("Training Loss: {} Validation Loss: {}".format(train_loss, val_loss))
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    df = pd.DataFrame({"Training_Loss": np.array(train_loss_list), "Validation_Loss": np.array(val_loss_list)})
    now = datetime.now()
    date_time = now.strftime("%H_%M_%S__%m_%d_%Y")
    torch.save(model, paths.MODELS_ROOT + date_time + ".pt")
    df.to_csv(paths.LOSSES_ROOT + date_time + ".csv", index=False)


def predict_gmax(input_arr: np.ndarray, clf: str = paths.RESOURCES_ROOT + "gmax_classifier.joblib") -> np.ndarray:
    classifier = load(clf)
    return classifier.predict(input_arr)


def build_dataloader_from_numpy(input_arr: np.ndarray,
                                labels_arr: np.ndarray,
                                batch_size: int = 1,
                                shuffle: bool = False) -> Tuple[DataLoader, DataLoader]:

    if shuffle:
        shuffler = np.random.permutation(input_arr.shape[0])
        input_arr = input_arr[shuffler]
        labels_arr = labels_arr[shuffler]

    idx = int(input_arr.shape[0] * .75)

    training_dataset = TensorDataset(torch.Tensor(input_arr[:idx, :]), torch.Tensor(labels_arr[:idx, :]))
    testing_dataset = TensorDataset(torch.Tensor(input_arr[idx:, :]), torch.Tensor(labels_arr[idx:, :]))
    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=testing_dataset, batch_size=batch_size)
    return train_loader, test_loader


def build_dataloader_from_file(input_file: str,
                               labels_file: Optional[str] = None,
                               batch_size: int = 1,
                               shuffle: bool = False) -> Tuple[DataLoader, DataLoader]:
    if not os.path.isfile(input_file):
        raise FileNotFoundError("File {} does not exist".format(str))
    if not os.path.isfile(labels_file):
        print("No labels file found, using last index of input file as labels")
        inputs = pd.read_csv(input_file).to_numpy()[:, :-1]
        labels = pd.read_csv(input_file).to_numpy()[:, -1]
    else:
        inputs = pd.read_csv(input_file).to_numpy()
        labels = pd.read_csv(labels_file).to_numpy()

    if shuffle:
        shuffler = np.random.permutation(inputs.shape[0])
        inputs = inputs[shuffler]
        labels = labels[shuffler]

    idx = int(inputs.shape[0] * .75)

    training_dataset = TensorDataset(torch.Tensor(inputs[:idx, :]), torch.Tensor(labels[:idx]))
    testing_dataset = TensorDataset(torch.Tensor(inputs[idx:, :]), torch.Tensor(labels[idx:]))
    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=testing_dataset, batch_size=batch_size)
    return train_loader, test_loader
