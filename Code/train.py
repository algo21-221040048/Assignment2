# This part of code is the main body, which used to train the data and make validation
import torch.utils.data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch import optim
from pytorchtools import EarlyStopping
from model import *
import numpy as np
import joblib


# Hyper parameters
EPOCHS = 20  # repeat times
LR = 0.0001  # learning rate
PATIENCE = 5  # early-stopping
BATCH_SIZE = 1000  # batch size
TRAIN_SIZE = 0.5  # train test ratio
DATA_NUM = 7  # there are seven period of data from 2011-01-31 to 2020-05-29


__all__ = ["WrappedDataLoader",
           "read_data",
           "get_data",
           "get_model",
           "preprocess",
           "loss_batch",
           "fit",
           "main"]


class WrappedDataLoader:
    """
    This class is used to preprocess each batch
    """
    def __init__(self, dl, func, dev):
        self.dl = dl
        self.func = func
        self.dev = dev

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.func(*b, self.dev)


def read_data(part_num: int) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """
    This function is used to read and split the particular data
    :param part_num: the ith part of the training code
    """
    x_name = '../Data_preprocessing/data_train_x_part_{}.pkl'.format(part_num)
    x_date_name = '../Data_preprocessing/data_train_x_date_part_{}.pkl'.format(part_num)
    y_name = '../Data_preprocessing/data_train_y_part_{}.pkl'.format(part_num)
    x = joblib.load(x_name)
    x_date = joblib.load(x_date_name)
    y = joblib.load(y_name)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=TRAIN_SIZE, shuffle=False)
    x_train, x_valid, y_train, y_valid = map(torch.tensor, (x_train, x_valid, y_train, y_valid))
    x_date_train = x_date[0:x_train.shape[0]]
    x_date_valid = x_date[x_train.shape[0]:]
    # this step might not be necessary, need test! key thought is like 'ResNet'
    # y_train = y_train.apply_(lambda u: None if u is None else 0.2*10 if u > 0.2 else 10*u if u > -0.2 else -0.2*10)
    # y_valid = y_valid.apply_(lambda u: None if u is None else 0.2*10 if u > 0.2 else 10*u if u > -0.2 else -0.2*10)
    return x_train, x_valid, y_train, y_valid, x_date_train, x_date_valid


def get_data(train_ds: torch.utils.data.TensorDataset, valid_ds: torch.utils.data.TensorDataset, bs: int)\
        -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """
    This function is used to reconstruct batches through `Dataloader` method
    :param train_ds: train dataset
    :param valid_ds: valid dataset
    :param bs: batch size
    """
    return DataLoader(train_ds, batch_size=bs, shuffle=False), DataLoader(valid_ds, batch_size=bs * 2)


def get_model() -> (AlphaNet_v1, optim, torch.device):
    """
    This function is used to build the model and optimizer
    """
    model_init = AlphaNet_v1()
    print(torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_init.to(dev)
    return model_init, optim.RMSprop(model_init.parameters(), lr=LR), dev


def preprocess(x: torch.tensor, y: torch.tensor, dev: torch.device) -> (torch.tensor, torch.tensor):
    """
    This function is used to reshape the training data
    :param x: data picture
    :param y: return5
    :param dev: device `cpu` or `cuda`
    """
    return x.view(-1, 1, 9, 30).to(dev), y.to(dev)


def loss_batch(model: AlphaNet_v1, loss_func: torch.nn.MSELoss, xb: torch.tensor, yb: torch.tensor, opt=None) -> (torch.tensor, int):
    """
    This function is used to calculate the loss of each batch
    :param model: training model
    :param loss_func: MSE
    :param xb: data picture in a batch
    :param yb: return5 in a batch
    :param opt: RMSprop
    """
    loss = loss_func(model(xb).float(), yb.float())

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()  # zero the gradient buffers! then it will not related with the above mini-batch

    return loss.item(), len(xb)


def fit(part_num: int, epochs: int, model: AlphaNet_v1, loss_func: torch.nn.MSELoss, opt: torch.optim.RMSprop, train_dl: WrappedDataLoader, valid_dl: WrappedDataLoader, train_ds: torch.utils.data.Dataset, patience: int):
    """
    This function is used to train the model and make the validation
    :param part_num: 滚动训练的第几部分数
    :param epochs: EPOCHS
    :param model: model
    :param loss_func: MSE
    :param opt: RMSprop
    :param train_dl: train dataloader
    :param valid_dl: valid dataloader
    :param train_ds: train dataset
    :param patience: PATIENCE
    """
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    path = '../IC_test_and_plot_data_trade_order/model_checkpoint_in_part_{}.pt'.format(part_num)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (xb, yb) in enumerate(train_dl):
            loss, xb_len = loss_batch(model, loss_func, xb, yb, opt)
            batch_idx = batch_idx + 1
            if batch_idx % len(train_dl) != 0:
                print('Train Epoch: {} [{}/{} {:.2f}%]\tLoss: {:.6f}'.format(epoch + 1, batch_idx * xb_len, len(train_ds), (100 * batch_idx * xb_len) / len(train_ds), loss))
            else:
                print('Train Epoch: {} [{}/{} {:.0f}%]\tLoss: {:.6f}'.format(epoch + 1, (batch_idx - 1) * BATCH_SIZE + xb_len, len(train_ds), 100, loss))
            train_losses.append(loss / xb_len)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
            for i in range(len(losses)):
                valid_losses.append(losses[i] / nums[i])

        # valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums), loss_func = F.cross_entropy
        valid_loss = np.sum(losses) / np.sum(nums)  # namely, valid all data, for period = 1, LOSS / 599719
        avg_valid_losses.append((epoch + 1, valid_loss))

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(path))
    print("滚动训练第{}部分训练结束的模型参数为：".format(part_num))
    for name, each in model.named_parameters():
        print(name, each, each.shape, each.requires_grad)
    return train_losses, valid_losses, avg_valid_losses


def main(part_num: int):
    """
    This function is the main function
    :param part_num: 滚动训练的第几部分数
    """
    PATH_train_losses = '../IC_test_and_plot_data_trade_order/train_losses_{}.pkl'.format(part_num)
    PATH_valid_losses = '../IC_test_and_plot_data_trade_order/valid_losses_{}.pkl'.format(part_num)
    PATH_avg_valid_losses = '../IC_test_and_plot_data_trade_order/avg_valid_losses_{}.pkl'.format(part_num)
    x_train, x_valid, y_train, y_valid, x_date_train, x_date_valid = read_data(part_num)
    train_data_dataset = TensorDataset(x_train, y_train)
    valid_data_dataset = TensorDataset(x_valid, y_valid)
    train_data_dataloader, valid_data_dataloader = get_data(train_data_dataset, valid_data_dataset, BATCH_SIZE)
    train_data_dataloader = WrappedDataLoader(train_data_dataloader, preprocess, device)
    valid_data_dataloader = WrappedDataLoader(valid_data_dataloader, preprocess, device)
    print("\n第{}部分train数据集有{}个数据，每个数据的大小为{}和{};valid数据集有{}个数据，每个数据的大小为{}和{}".format(
        part_num,
        len(train_data_dataset),
        train_data_dataset.__getitem__(0)[0].unsqueeze(0).shape,
        train_data_dataset.__getitem__(0)[1].unsqueeze(0).shape,
        len(valid_data_dataset),
        valid_data_dataset.__getitem__(0)[0].unsqueeze(0).shape,
        valid_data_dataset.__getitem__(0)[1].unsqueeze(0).shape))
    print("第{}部分训练集分组数为{}, 验证级分组数为{}".format(part_num, len(train_data_dataloader), len(valid_data_dataloader)))

    train_losses, valid_losses, avg_valid_losses = fit(part_num, EPOCHS, model, loss_func, opt, train_data_dataloader,
                                                       valid_data_dataloader, train_data_dataset, PATIENCE)
    joblib.dump(train_losses, PATH_train_losses)
    joblib.dump(valid_losses, PATH_valid_losses)
    joblib.dump(avg_valid_losses, PATH_avg_valid_losses)


if __name__ == '__main__':
    model, opt, device = get_model()
    model.to(device)
    loss_func = torch.nn.MSELoss(reduction='sum')  # only sum operation, not mean; if reduction='mean', then it will get average on both batch and features
    print("未训练的模型参数为：")
    # model.load_state_dict(torch.load('../IC_test_and_plot_data_trade_order/model_checkpoint_in_part_1.pt'))
    for name, each in model.named_parameters():
        print(name, each, each.shape, each.requires_grad)
    for data_num in range(1, DATA_NUM + 1):
        print("Viewing data_x_part_{}.pkl and data_y_part_{}.pkl".format(data_num, data_num))
        print("滚动训练第{}部分训练开始的模型参数为：".format(data_num))
        for name, each in model.named_parameters():
            print(name, each, each.shape, each.requires_grad)
        main(data_num)




