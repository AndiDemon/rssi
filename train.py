import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.data import Data
from models.CNN_former import CNNformer


def main():
    BATCH_SIZE = 16
    EPOCH = 1000

    data_file = "./dataset/data_n_1.csv"

    data = pd.read_csv(data_file)
    feature = []  # data[["1", "2", "3"]]
    label = []

    # (100, 3, 3)

    for i in range(len(data)):
        # string to int
        feature.append([[(int(data["1"][i]) - (-110)) / (max(data["1"]) - (-110)), 7/13, 1/29],
                        [(int(data["2"][i]) - (-110)) / (max(data["2"]) - (-110)), 7/13, 29/29],
                        [(int(data["3"][i]) - (-110)) / (max(data["3"]) - (-110)), 1/13, 15/29]])
        label.append([(int(data["(x,y)"][i].split(",", 2)[0]) - 1)/(13-1), (int(data["(x,y)"][i].split(",", 2)[1]) - 1)/(29-1)])

    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)

    print("shape X_Train = ", np.array(X_train).shape)
    print("shape y_Train = ", np.array(y_train).shape)
    print("shape X_test = ", np.array(X_test).shape)
    print("shape y_test = ", np.array(y_test).shape)

    train_data, test_data = [], []
    for i in range(len(X_train)):
        train_data.append([X_train[i], y_train[i]])
    for i in range(len(X_test)):
        train_data.append([X_test[i], y_test[i]])

    train_loader = DataLoader(Data(data=train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Data(data=test_data), batch_size=BATCH_SIZE, shuffle=False)

    model = CNNformer(input_dim=3, conv_hidden=32, lin_hidden=32, num_heads=4, drop=0.2)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    loss_by_epoch = []
    epoch_now = []
    for epoch in range(EPOCH):
        loss_all = 0
        for batch, (src, trg) in enumerate(train_loader):
            pred = model(src)  # .to("cuda:0")

            loss = torch.sqrt(criterion(pred, trg))

            loss_all += loss.detach().cpu().numpy()

            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
        loss_by_epoch.append(loss_all / len(train_loader))
        epoch_now.append(epoch + 1)
        print("EPOCH = ", epoch, ", LOSS = ", loss_all / len(train_loader))

        plt.plot(epoch_now, loss_by_epoch)
        plt.savefig("loss.png")


if __name__ == "__main__":
    main()
