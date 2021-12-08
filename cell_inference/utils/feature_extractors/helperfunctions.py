import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

from cell_inference.config import paths


def train_regression(model: nn.Module, training_loader: torch.utils.data.DataLoader,
                     validation_loader: torch.utils.data.DataLoader,
                     epochs: int = 50,
                     learning_rate: float = 0.005,
                     decay_rate: float = 1.0,
                     device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

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
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    df = pd.DataFrame({"Training_Loss": np.array(train_loss_list), "Validation_Loss": np.array(val_loss_list)})
    now = datetime.now()
    date_time = now.strftime("%H_%M_%S__%m_%d_%Y")
    torch.save(model, paths.MODELS_ROOT + date_time + ".pt")
    df.to_csv(paths.LOSSES_ROOT + date_time + ".csv", index=False)
