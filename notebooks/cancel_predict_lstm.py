import numpy as np
import pandas as pd

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from sam.sam import SAM

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def data_loader(path):
    df = pd.read_csv(path)

    '''Data Prep'''

    df['arrival_date_month'] = pd.to_datetime(df.arrival_date_month, format='%B').dt.month
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].map(str) + '/' + df['arrival_date_month'].map(str) + '/' + df[
            'arrival_date_day_of_month'].map(str))
    df['arrival_day'] = df['arrival_date'].dt.day_name()
    df['length_of_stay'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
    df['departure_date'] = df['arrival_date'] + pd.TimedeltaIndex(df['length_of_stay'], unit='D')
    df['departure_day'] = df['departure_date'].dt.day_name()
    df['booking_date'] = df['arrival_date'] - pd.TimedeltaIndex(df['lead_time'], unit='D')
    df['booking_day'] = df['booking_date'].dt.day_name()
    df['revenue'] = df['adr'] * df['length_of_stay']
    df['occupants'] = df['adults'] + df['children'].fillna(value=0) + df['babies']

    cancel_city = df.loc[
        df['hotel'] == 'City Hotel', ['is_canceled', 'lead_time', 'length_of_stay', 'occupants', 'distribution_channel',
                                      'is_repeated_guest',
                                      'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes',
                                      'deposit_type',
                                      'days_in_waiting_list', 'customer_type', 'adr', 'required_car_parking_spaces',
                                      'total_of_special_requests']]

    # Create Dummy Variables

    cat_vars = ['distribution_channel', 'deposit_type', 'customer_type']
    for var in cat_vars:
        cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(cancel_city[var], prefix=var)
        data = cancel_city.join(cat_list)
        cancel_city = data
    data_vars = cancel_city.columns.values.tolist()
    to_keep = [i for i in data_vars if i not in cat_vars]
    city_cancellation = cancel_city[to_keep]

    X = city_cancellation.iloc[:, 1:]
    y = city_cancellation.iloc[:, 0]

    return X, y


class LstmModel(nn.Module):
    def __init__(self):
        super(LstmModel, self).__init__()

        self.lstm = nn.LSTM(input_size=X_train.shape[1], hidden_size=128, batch_first=True)
        self.linear = nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(1, len(input_seq), -1))
        # print(lstm_out.shape)
        outputs = self.linear(lstm_out.squeeze(0))
        # print(outputs)
        outputs = torch.sigmoid(outputs)
        # print(outputs)

        return outputs[-1]


class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TestData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


def binary_acc(y_pred, y_label):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_label).sum().float()
    acc = correct_results_sum / y_label.shape[0]
    acc = torch.round(acc * 100)

    return acc


if __name__ == "__main__":
    data_path = 'D:/Project/datasets/hotel_bookings.csv'
    model_path = 'D:/Project/cancellation_prediction/ckpt/cancellation_lstm.pt'

    X_data, y_data = data_loader(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=13)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_data = TrainData(torch.tensor(X_train, dtype=torch.float32),
                           torch.tensor(np.array(y_train), dtype=torch.float32))

    test_data = TestData(torch.tensor(X_test, dtype=torch.float32))

    train_loader = DataLoader(dataset=train_data, batch_size=1)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    do_train = False

    # Model, loss function and optimizer
    model = LstmModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # base_optimizer = torch.optim.SGD
    # optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    if do_train:
        epochs = 20
        for i in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                y_pred = model(X_batch)

                loss = loss_function(y_pred, y_batch)
                acc = binary_acc(y_pred, y_batch)

                loss.backward()
                optimizer.step()
                # optimizer.first_step(zero_grad=True)

                # loss_function(model(X_batch), y_batch).backward()
                # optimizer.second_step(zero_grad=True)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            print(f'Epoch{i+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

        torch.save(model.state_dict(), model_path)

    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()

        y_pred_list = []

        with torch.no_grad():
            for X_batch in test_loader:
                y_test_pred = model(X_batch)
                y_pred_tag = torch.round(y_test_pred)
                y_pred_list.append(np.array(y_pred_tag))

        y_pred_list = [a.tolist() for a in y_pred_list]

        print(confusion_matrix(y_test, y_pred_list))
        print(classification_report(y_test, y_pred_list))
        print(len(y_pred_list))