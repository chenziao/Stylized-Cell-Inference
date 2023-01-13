import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from contextlib import nullcontext
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Union, Tuple
import os
from joblib import dump, load

from cell_inference.config import paths


def train_model(model: nn.Module, training_loader: DataLoader, validation_loader: DataLoader,
                     epochs: int = 50, learning_rate: Optional[float] = None, decay_rate: float = 1.0,
                     optimizer: Union[torch.optim.Optimizer, str] = 'Adam',
                     loss_func: nn.modules.loss._Loss = nn.MSELoss(),
                     device: torch.device = None, save_model: bool = True, save_history: bool = True) -> None:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_loaders = {"train": training_loader, "val": validation_loader}
    phases = ['train', 'val'] if len(validation_loader) > 0 else ['train']

    if isinstance(optimizer, str):
        if learning_rate is None:
            learning_rate = 0.001
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        if learning_rate is not None:
            optimizer.param_groups[0]['lr'] = learning_rate
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

    epochs_list = []
    train_loss_list = []
    val_loss_list = []
    for epoch in tqdm(range(epochs), position=0, leave=True):
        val_loss = 0.0
        for phase in phases:
            training = phase == 'train'
            model.train(training)
            running_loss = 0.0
            with nullcontext() if training else torch.no_grad():
                for i, (x, y) in enumerate(data_loaders[phase]):
                    output = model(x.to(device))
                    loss = loss_func(output, y.to(device))
                    # backprop
                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    # calculating total loss
                    running_loss += loss.item()

            running_loss /= len(data_loaders[phase].dataset)
            if training:
                train_loss = running_loss
                lr_sch.step()
            else:
                val_loss = running_loss

        if epoch % 10 == 0:
            tqdm.write("Training Loss: {} Validation Loss: {}".format(train_loss, val_loss))
        if save_history:
            epochs_list.append(epoch)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

    history = pd.DataFrame({"Epochs": epochs_list, "Training_Loss": train_loss_list,
                            "Validation_Loss": val_loss_list}) if save_history else None
    files = save_model_history(model, history) if save_model else ['', '']
    return history, files


def save_model_history(model, history=None):
    date_time = datetime.now().strftime("%H_%M_%S__%m_%d_%Y")
    model_file = date_time + ".pt"
    torch.save(model, os.path.join(paths.MODELS_ROOT, model_file))
    files = [model_file]
    if history is None:
        files.append('')
    else:
        history_file = date_time + ".csv"
        history.to_csv(os.path.join(paths.LOSSES_ROOT, history_file), index=False)
        files.append(history_file)
    return files


def predict_gmax(input_arr: np.ndarray, clf: str = paths.RESOURCES_ROOT + "gmax_classifier.joblib") -> np.ndarray:
    classifier = load(clf)
    return classifier.predict(input_arr)


def build_dataloader_from_numpy(input_arr: np.ndarray, label_arr: np.ndarray, keep_dtype: bool = False,
                                batch_size: int = 1, train_size: Union[float, int] = 0.8,
                                shuffle: bool = False, seed: int = 0) -> Tuple[DataLoader, DataLoader]:
    input_arr - np.asarray(input_arr)
    label_arr - np.asarray(label_arr)
    input_dtype = getattr(torch, str(input_arr.dtype)) if keep_dtype else torch.float32
    label_dtype = getattr(torch, str(label_arr.dtype)) if keep_dtype else torch.float32

    data_size = input_arr.shape[0]
    if train_size <= 1:
        idx = max(int(np.floor(data_size * train_size)), 1)
    else:
        idx = min(int(train_size), data_size)

    idx_train = slice(idx)
    idx_test = slice(idx, None)
    if shuffle:
        np.random.seed(seed)
        shuffler = np.random.permutation(data_size)
        idx_train = shuffler[idx_train]
        idx_test = shuffler[idx_test]

    training_dataset = TensorDataset(torch.as_tensor(input_arr[idx_train], dtype=input_dtype),
                                     torch.as_tensor(label_arr[idx_train], dtype=label_dtype))
    testing_dataset = TensorDataset(torch.as_tensor(input_arr[idx_test], dtype=input_dtype),
                                    torch.as_tensor(label_arr[idx_test], dtype=label_dtype))
    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=testing_dataset, batch_size=batch_size)
    return train_loader, test_loader


def build_dataloader_from_file(input_file: str, labels_file: Optional[str] = None,
                               batch_size: int = 1, train_size: Union[float, int] = 0.8,
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

    data_size = inputs.shape[0]
    if train_size <= 1:
        idx = max(int(np.floor(data_size * train_size)), 1)
    else:
        idx = min(int(train_size), data_size)

    idx_train = slice(idx)
    idx_test = slice(idx, None)
    if shuffle:
        shuffler = np.random.permutation(data_size)
        idx_train = shuffler[idx_train]
        idx_test = shuffler[idx_test]

    training_dataset = TensorDataset(torch.Tensor(inputs[idx_train]), torch.Tensor(labels[idx_train]))
    testing_dataset = TensorDataset(torch.Tensor(inputs[idx_test]), torch.Tensor(labels[idx_test]))
    train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=testing_dataset, batch_size=batch_size)
    return train_loader, test_loader
