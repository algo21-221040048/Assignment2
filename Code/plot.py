# This part of code is used to visualize the training result
# Input: the train losses, valid losses, avg valid losses in each part
# Output: two plots
import matplotlib.pyplot as plt
import joblib

__all__ = ["read_data",
           "plot_loss",
           "plot_early_stop"]


def read_data(part_num: int) -> (list, list, list):
    """
    This function is used to read the particular data
    :param part_num: test data part
    """
    x_name = '../IC_test_and_plot_data_trade_order/train_losses_{}.pkl.gz'.format(part_num)
    y_name = '../IC_test_and_plot_data_trade_order/valid_losses_{}.pkl.gz'.format(part_num)
    z_name = '../IC_test_and_plot_data_trade_order/avg_valid_losses_{}.pkl.gz'.format(part_num)
    x = joblib.load(x_name)
    y = joblib.load(y_name)
    z = joblib.load(z_name)
    return x, y, z


def plot_loss(t_l: list, v_l: list) -> ():
    """
    This function is used to visualize the Loss and the Early Stopping Checkpoint and the loss as the network trained
    :param t_l: train_losses
    :param v_l: valid_losses
    """
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(t_l) + 1), t_l, label='Training Loss, BS={}'.format(1000))
    plt.plot(range(1, len(v_l) + 1), v_l, label='Validation Loss, BS={}'.format(2000))
    plt.xlabel('batches')
    plt.ylabel('average loss in each batch')
    plt.xlim(0, len(t_l) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')


def plot_early_stop(avg_v_l: list) -> ():
    """
    This function is used to visualize the Loss and the Early Stopping Checkpoint and the loss as the network trained
    :param avg_v_l: avg_valid_losses
    """
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(avg_v_l) + 1), avg_v_l, label='Average Validation Loss Per Epoch')

    # find position of lowest validation loss
    min_poss = avg_v_l.index(min(avg_v_l)) + 1
    plt.axvline(min_poss, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.xlabel('epochs')
    plt.ylabel('average loss in each epoch')
    plt.xlim(0, len(avg_v_l) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot_epochs.png', bbox_inches='tight')


if __name__ == '__main__':
    train_losses, valid_losses, avg_valid_losses = read_data(1)
    plot_loss(train_losses, valid_losses)
    num, avg_valid_losses = zip(*avg_valid_losses)
    plot_early_stop(avg_valid_losses)

