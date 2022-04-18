# This part of code is used to calculate the IC, neutralized IC and do the stratification test， finally get the return, sharp and max_draw_down
# Input: {each part: test data(x, x_date, y), model_checkpoint.pt}, ev_data, industry_data
# Output: IC, cumulative_RankIC, neutralize IC, stratification result
from model import *
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.linear_model import LinearRegression


# Hyper parameters
TEST_DATA_PART = 1  # backtest data part
MODEL_CHECK_POINT = 2  # checking-model parameters part
GROUP_NUM = 5  # stratification nums
WINDOW_SIZE = 5  # rolling_window_size in stratification


__all__ = ["read_ev_and_industry_data",
           "read_test_data",
           "read_original_data",
           "read_trading_date_data",
           "data_fillingNa",
           "get_model",
           "preprocess",
           "mark_data_set",
           "return5_cal",
           "_window_data_handle_two_neutralize",
           "_window_data_handle_five_neutralize",
           "_preprocess_factor_two_neutralize",
           "_preprocess_factor_five_neutralize",
           "_calculate_ic",
           "_plot_cumulative_RankIC",
           "_get_stock_daily_date",
           "_stratification_test_one_date",
           "_stratification_test_one_window",
           "_cal_max_draw_down",
           "_cal_net_indexes"]


def read_ev_and_industry_data() -> (pd.DataFrame, pd.DataFrame):
    """
    This function is used to read the data which is necessary to do the neutralize
    """
    ev_data = pd.read_csv('../OriginalData_csv/ev_data.csv', header=0, low_memory=False)
    if 'Unnamed: 0' in ev_data.columns:
        ev_data = ev_data.drop(['Unnamed: 0'], axis='columns')

    industry_data = pd.read_csv('../OriginalData_csv/industry_data.csv', header=0, low_memory=False)
    if 'Unnamed: 0' in industry_data.columns:
        industry_data = industry_data.drop(['Unnamed: 0'], axis='columns')

    return ev_data, industry_data


def read_test_data(part_num: int) -> (np.array, np.array, np.array):
    """
    This function is used to read the backtest data
    :param part_num: backtest data part
    """
    x_name = '../Data_preprocessing/data_test_x_part_{}.pkl'.format(part_num)
    x_date_name = '../Data_preprocessing/data_test_x_date_part_{}.pkl'.format(part_num)
    y_name = '../Data_preprocessing/data_test_y_part_{}.pkl'.format(part_num)
    x = joblib.load(x_name)
    z = joblib.load(x_date_name)
    y = joblib.load(y_name)
    return x, z, y


def read_original_data(part_num: int, to_filter: bool = True) -> pd.DataFrame:
    """
    This function is used to get the particular period original data of the whole stock and filter the data if it is necessary
    :param part_num: backtest data part
    :param to_filter: whether to do the filter operation
    """
    x_data_without_return1 = pd.read_csv('../OriginalData_csv/x_data_without_return1.csv', header=0)
    if 'Unnamed: 0' in x_data_without_return1.columns:
        x_data_without_return1 = x_data_without_return1.drop(['Unnamed: 0'], axis='columns')
    filter_data = pd.read_csv('../OriginalData_csv/filter_data.csv', header=0, low_memory=False)
    if 'Unnamed: 0' in filter_data.columns:
        filter_data = filter_data.drop(['Unnamed: 0'], axis='columns')
    train_end_list = joblib.load('../Data_preprocessing/train_end_list.pkl')
    data = pd.merge(x_data_without_return1, filter_data, how='left', on=['trading_date', 'wind_code'])
    medium = data[(data['trading_date'] >= train_end_list[part_num - 1]) & (data['trading_date'] < train_end_list[part_num])]
    medium = medium.reset_index(drop=True)
    medium = medium[~medium['adj_factor'].isna()].reset_index(drop=True)  # filter several stocks
    medium = medium[['trading_date', 'wind_code', 'open_price', 'close_price', 'adj_factor', 'vwap', 'is_st', 'trade_status']]
    medium['open_price'] = medium['open_price'] * medium['adj_factor']
    medium['close_price'] = medium['close_price'] * medium['adj_factor']
    if to_filter:
        medium = medium[(medium['is_st'] == 0) & (medium['trade_status'] == '交易')]  # 过滤掉停牌和非交易的股票
        medium = medium[['trading_date', 'wind_code', 'open_price', 'close_price']]
        assert medium[medium['open_price'].isna()].shape[0] == 0
        return medium.reset_index(drop=True)
    else:
        medium = medium[['trading_date', 'wind_code', 'open_price', 'close_price']]
        new_medium = medium.groupby('wind_code').apply(lambda x: x.fillna(method='bfill', axis=0))  # 后填充
        new_medium = new_medium.dropna(how='any', axis=0)  # 这段时间后期全部停牌的股票，无法填充，直接过滤
        assert new_medium[new_medium['open_price'].isna()].shape[0] == 0
        return new_medium.reset_index(drop=True)


def read_trading_date_data() -> list:
    """
    This function is used to read the particular trading_date data
    """
    trade_list = pd.read_csv('../OriginalData_csv/Trading_date.csv', header=0, index_col=False)
    trade_list = trade_list['trading_date'].tolist()
    return trade_list


def data_fillingNa(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to back fill the data
    :param df: df
    """
    medium = df[['wind_code', 'open_price', 'close_price']]
    medium = medium.groupby('wind_code').fillna(method='bfill', axis=0)
    return df.fillna(medium)


def get_model() -> (AlphaNet_v1, torch.device):
    """
    This function is used to initialize the model
    """
    model_init = AlphaNet_v1()
    print(torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_init.to(dev)
    return model_init, dev


def preprocess(x: torch.tensor, y: torch.tensor, dev: torch.device) -> (torch.tensor, torch.tensor):
    """
    This function is used to reshape the training data
    :param x: data_picture
    :param y: return5
    :param dev: `cpu` or `cuda`
    """
    return x.view(-1, 1, 9, 30).to(dev), y.to(dev)


def mark_data_set(df, industry_list):
    """
    This function is used to transform the industry variables to {0, 1} distribution
    :param df: df
    :param industry_list: 28 industries
    """
    tmp_df = df
    for k in industry_list:
        tmp_df[k] = (tmp_df['industry_sw'] == k).astype(int)
    return tmp_df


def return5_cal(row):
    """
    This function is used to calculate the return from t_{-5} to t_0
    :param row: row
    """
    return5 = (row[-1] - row[0]) / row[0]
    return return5


def _window_data_handle_two_neutralize(input_data: pd.DataFrame) -> (pd.DataFrame, list):
    """
    This function is used to process the period data in order to get the two factors
    :param input_data: window data, `trading_date`, `wind_code`, `factor=alphaNet`, `true`
    """
    ev_data, industry_data = read_ev_and_industry_data()
    industry_list = industry_data['industry_sw'].unique().tolist()
    input_data = pd.merge(input_data, ev_data, how='left', on=['trading_date', 'wind_code'])
    input_data = pd.merge(input_data, industry_data, how='left', on=['trading_date', 'wind_code'])
    input_data = input_data.groupby('trading_date').apply(mark_data_set, industry_list)
    for i in industry_list:
        assert input_data[i].sum() == input_data[input_data['industry_sw'] == i].shape[0]

    # handle ev
    input_data['log_ev'] = np.log(input_data['ev'])

    return input_data[~input_data['ev'].isna()].reset_index(drop=True), industry_list  # there are several records have no ev data


def _window_data_handle_five_neutralize(input_data: pd.DataFrame, x: torch.Tensor) -> (pd.DataFrame, list):
    """
    This function is used to process the period data in order to get the five factors
    :param input_data: window data, `trading_date`, `wind_code`, `factor=alphaNet`, `true`
    :param x: data_picture
    """
    ev_data, industry_data = read_ev_and_industry_data()
    industry_list = industry_data['industry_sw'].unique().tolist()
    input_data = pd.merge(input_data, ev_data, how='left', on=['trading_date', 'wind_code'])
    input_data = pd.merge(input_data, industry_data, how='left', on=['trading_date', 'wind_code'])
    input_data = input_data.groupby('trading_date').apply(mark_data_set, industry_list)
    for i in industry_list:
        assert input_data[i].sum() == input_data[input_data['industry_sw'] == i].shape[0]

    # handle ev
    input_data['log_ev'] = np.log(input_data['ev'])

    # handle volatility5
    input_data['volatility5'] = x[:, 205:210].numpy().std(axis=1, ddof=0)

    # handle turn5
    input_data['turn5'] = x[:, 235:240].mean(axis=1)

    # handle return5
    input_data['return5'] = np.apply_along_axis(return5_cal, 1, x[:, 114:120])

    return input_data[~input_data['ev'].isna()].reset_index(drop=True), industry_list  # there are several records have no ev data


def _preprocess_factor_two_neutralize(data: pd.DataFrame, factor: str, idu_list: list):
    """
    This function is used to do the two-factor-neutralize: industry and ev
    :param data: _window_data_handle_two_neutralize result, log_ev, industry is ready
    :param factor: factor
    :param idu_list: the list of 28 industries
    """
    if data[factor].isna().sum() == len(data):
        data[factor] = [None] * len(data)
    else:
        # get rid of the extremum by using $D_M \pm 5D_{M1}$
        median = data[factor].median()
        diff = abs(data[factor] - median).median()
        if diff > 0:
            up = median + 5 * diff
            bot = median - 5 * diff
            data[factor] = data[factor].apply(lambda x: None if x is None else up if x > up else x if x > bot else bot)

        # na factor handle
        naDict = data[factor].groupby(data['industry_sw']).median()
        data[factor] = data[[factor, 'industry_sw']].apply(lambda row: row[factor] if row[factor] is not None and not np.isnan(row[factor]) else naDict[row['industry_sw']], axis=1)

        # industry and log(ev) neutralize
        linearRegression = LinearRegression()
        if factor != 'log_ev':
            colList = idu_list + ['log_ev']
            linearRegression.fit(data[colList], data[factor])
            pred = linearRegression.predict(data[colList])
            data[factor] = np.array(data[factor]) - pred

        # standard
        data[factor] = round((data[factor] - data[factor].mean()) / data[factor].std(), 4)


def _preprocess_factor_five_neutralize(data: pd.DataFrame, factor: str, idu_list: list):
    """
    This function is used to do the five-factor-neutralize: industry, ev, volatility5, turn5, return5
    :param data: _window_data_handle_five_neutralize result, log_ev, industry, volatility5, turn5, return5 is ready
    :param factor: factor
    :param idu_list: the list of 28 industries
    """
    if data[factor].isna().sum() == len(data):
        data[factor] = [None] * len(data)
    else:
        # get rid of the extremum by using $D_M \pm 5D_{M1}$
        median = data[factor].median()
        diff = abs(data[factor] - median).median()
        if diff > 0:
            up = median + 5 * diff
            bot = median - 5 * diff
            data[factor] = data[factor].apply(lambda x: None if x is None else up if x > up else x if x > bot else bot)

        # na factor handle
        naDict = data[factor].groupby(data['industry_sw']).median()
        data[factor] = data[[factor, 'industry_sw']].apply(lambda row: row[factor] if row[factor] is not None and not np.isnan(row[factor]) else naDict[row['industry_sw']], axis=1)

        # industry and log(ev) neutralize
        linearRegression = LinearRegression()
        if factor != 'log_ev':
            colList = idu_list + ['log_ev', 'volatility5', 'turn5', 'return5']
            linearRegression.fit(data[colList], data[factor])
            pred = linearRegression.predict(data[colList])
            data[factor] = np.array(data[factor]) - pred

        # standard
        data[factor] = round((data[factor] - data[factor].mean()) / data[factor].std(), 4)


def _calculate_ic(medium: pd.DataFrame, factor: str) -> (pd.Series, float):
    """
    This function is used to calculate the ic
    :param medium: window_data
    :param factor: factor=alphaNet
    Note: pd.Dataframe.corr(method=‘spearman’) = pd.Dataframe.rank().corr(method=‘pearson’)
    """
    print("Now calculating the IC in test part {} by using model parameters saved in part {}".format(TEST_DATA_PART, MODEL_CHECK_POINT))
    ic_medium = medium.groupby('trading_date').apply(lambda x: x[[factor, 'true']].corr(method='spearman').iloc[0]['true'])
    ic_avg_medium = ic_medium.mean()
    ic_std_medium = ic_medium.std(ddof=0)
    IC_IR_medium = ic_avg_medium / ic_std_medium
    IC_over_zero = ic_medium[ic_medium > 0].shape[0] / ic_medium.shape[0]
    print("The avg_single_stock_loss in test part {} is {}, the avg_RankIC = {:.2f}%, std_RankIC = {:.2f}%, IC_IR = {:.2f}, IC > 0占比 = {:.2f}%".format(TEST_DATA_PART, loss / len(data_picture), ic_avg_medium * 100, ic_std_medium * 100, IC_IR_medium, IC_over_zero * 100))
    return ic_medium, ic_avg_medium


def _plot_cumulative_RankIC(x: pd.Series, factor: str) -> ():
    """
    This function is used to calculate the cumulative ic and plot it
    :param x: index `trading_date`, value `ic`
    :param factor: factor=alphaNet
    """
    cumulative_data = np.cumsum(x)
    num_data_points = len(cumulative_data.index)
    fig = figure(figsize=(30, 10), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(cumulative_data.index.tolist(), cumulative_data.values.tolist(), color="#001f3f")
    plt.title('{}_IC From {} to {}'.format(factor, cumulative_data.index[0], cumulative_data.index[-1]))
    xticks = [cumulative_data.index[i] if ((i % 3 == 0 and (num_data_points - i) > 3) or i == num_data_points - 1)
              else None for i in range(num_data_points)]
    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.show()
    fig.savefig('{}_IC From {} to {}'.format(factor, cumulative_data.index[0], cumulative_data.index[-1]), bbox_inches='tight')


def _get_stock_daily_date(date: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to get the stocks' open and close price on specific day
    :param date: trading_date
    :param data: read_original_data result，two situations: filter=True and filter=False
    """
    medium = data[data['trading_date'] == date].reset_index(drop=True)
    return medium


def _stratification_test_one_date(date: str, factor_list: list, trade_date_list: list, x: pd.DataFrame) -> (str, pd.DataFrame):
    """
    This function is used to do the stratification test of two portfolios on specific day
    :param date: 换仓日
    :param factor_list: 待测试的因子(列表)
    :param trade_date_list: 交易日(列表)
    :param x: 将因子暴露度向量进行预处理之后的内容dataframe, index `trading_date`, `wind_code`, `factor_list`
    """
    # 1st day: get two portfolios according to factor values
    data = x[x['trading_date'] == date]
    data = data[['trading_date', 'wind_code'] + factor_list]
    hold_dict = {}
    for factor in factor_list:
        factor_dict = {}
        if len(data[factor].dropna()) == 0:
            factor_dict = {'top': [], 'bot': []}
        else:
            # get two portfolios
            top_pct = np.percentile(data[factor], 100 - 100 / GROUP_NUM)  # 80% < x, interpolation='linear'
            bot_pct = np.percentile(data[factor], 100 / GROUP_NUM)
            factor_dict['top'] = list(data['wind_code'].where(data[factor] >= top_pct).dropna())
            factor_dict['bot'] = list(data['wind_code'].where(data[factor] <= bot_pct).dropna())
            hold_dict[factor] = factor_dict
    net_list = []
    start_index = trade_date_list.index(date)
    # 2 st day: buy the stocks in each portfolio on their open price
    open_trade_data = _get_stock_daily_date(trade_date_list[start_index + 1], original_data_to_filter_true)
    open_trade_data['open_trade_price'] = open_trade_data['open_price']
    for window in range(1, WINDOW_SIZE + 1):
        close_trade_data = _get_stock_daily_date(trade_date_list[start_index + window + 1], original_data_to_filter_false)
        if close_trade_data.shape[0] == 0:
            break
        # 3 st day: cal the return of stocks in each portfolio on their open price
        close_trade_data['close_trade_price'] = close_trade_data['open_price']
        # ensure the stock is both not nan in two days
        price_data = pd.merge(open_trade_data[['open_trade_price', 'wind_code']],
                              close_trade_data[['close_trade_price', 'wind_code']])  # to do inner
        price_data['ret'] = price_data['close_trade_price'] / price_data['open_trade_price'] - 1
        price_data.index = price_data['wind_code']
        net_dict = {'trading_date': trade_date_list[start_index + window + 1], 'avg_return': price_data['ret'].mean()}
        for factor, factor_dict in hold_dict.items():
            if len(factor_dict['top']) == 0:
                net_dict = {factor + '_top': None, factor + '_bot': None, factor + '_diff_net': 1,
                            factor + '_top_ex_net': 1, factor + '_bot_ex_net': 1}
            else:
                # if one stock trade on 2st day but not trade on 3st day, then it have no effect on the portfolio return and net return since it is zero
                top_stock_list = [x for x in factor_dict['top'] if x in list(price_data['wind_code'])]
                net_dict[factor + '_top'] = price_data['ret'][top_stock_list].mean()
                bot_stock_list = [x for x in factor_dict['bot'] if x in list(price_data['wind_code'])]
                net_dict[factor + '_bot'] = price_data['ret'][bot_stock_list].mean()
                net_dict[factor + '_diff_net'] = net_dict[factor + '_top'] - net_dict[factor + '_bot'] + 1
                net_dict[factor + '_top_ex_net'] = net_dict[factor + '_top'] - net_dict['avg_return'] + 1
                net_dict[factor + '_bot_ex_net'] = net_dict[factor + '_bot'] - net_dict['avg_return'] + 1
        net_list.append(net_dict)
    net_df = pd.DataFrame(net_list)
    for factor in factor_list:
        if len(hold_dict[factor]['top']) == 0:
            net_df[factor + '_diff'] = [None] * len(net_df)
            net_df[factor + '_top_ex'] = [None] * len(net_df)
            net_df[factor + '_bot_ex'] = [None] * len(net_df)
        else:
            net_df[factor + '_diff'] = (net_df[factor + '_diff_net'] / net_df[factor + '_diff_net'].shift()).fillna(
                net_df[factor + '_diff_net'][0]) - 1
            net_df[factor + '_top_ex'] = (net_df[factor + '_top_ex_net'] / net_df[factor + '_top_ex_net'].shift()).fillna(
                net_df[factor + '_top_ex_net'][0]) - 1
            net_df[factor + '_bot_ex'] = (net_df[factor + '_bot_ex_net'] / net_df[factor + '_bot_ex_net'].shift()).fillna(
                net_df[factor + '_bot_ex_net'][0]) - 1
    return trade_date_list[start_index + window], net_df


def _stratification_test_one_window(date_list: list, factor_list: list, trade_date_list: list, x: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    This function is used to do the stratification test of two portfolios during a period
    :param date_list: 调仓日，即ic测试计算的列表的索引(回测开始日期与结束日期)
    :param factor_list: 待测试的因子(列表)
    :param trade_date_list: 交易日(列表)
    :param x: 将因子暴露度向量进行预处理之后的内容dataframe, index `trading_date`, `wind_code`, `factor_list`
    """
    start_date = date_list[0]
    last_date = date_list[-1]
    net_df = pd.DataFrame()
    last_net_dict = {}
    while start_date <= last_date:
        start_date, date_net_df = _stratification_test_one_date(start_date, factor_list, trade_date_list, x)
        for factor in factor_list:
            for col in ['_diff_net', '_top_ex_net', '_bot_ex_net']:
                date_net_df[factor + col] = date_net_df[factor + col] * last_net_dict.get(factor + col, 1)
                last_net_dict[factor + col] = list(date_net_df[factor + col])[-1]
        net_df = net_df.append(date_net_df)
    net_df.to_csv(f'stratification_test_5.csv', index=None)
    return net_df, last_net_dict


def _cal_max_draw_down(net_list: list) -> float:
    """
    This function is used to calculate the max_draw_down of the portfolio
    :param net_list: 分层测试结果的一列(代表一个投资组合)
    """
    high = 1
    max_diff = 0
    for net in net_list:
        diff = net / high - 1
        if diff > 0:
            high = net
        if diff < max_diff:
            max_diff = diff
    return max_diff


def _cal_net_indexes(net_daily_df: pd.DataFrame, factor_list) -> list:
    """
    This function is used to calculate the return, sharp and max_draw_down of the portfolio
    :param net_daily_df: 分层测试的结果
    :param factor_list: 待测试的因子(列表)
    """
    ind_list = []
    num = len(net_daily_df)
    for factor in factor_list:
        ind_dict = {}
        ind_dict['factor'] = factor
        ind_dict['long_short_Return'] = list(net_daily_df[factor + '_diff_net'])[-1] ** (252/num)
        ind_dict['long_short_Sharpe'] = net_daily_df[factor + '_diff'].mean() / net_daily_df[factor + '_diff'].std() * np.sqrt(252)
        ind_dict['long_short_Max_DrawDown'] = _cal_max_draw_down(list(net_daily_df[factor + '_diff_net']))
        ind_dict['top_ex_Return'] = list(net_daily_df[factor + '_top_ex_net'])[-1] ** (252/num)
        ind_dict['top_ex_Sharpe'] = net_daily_df[factor + '_top_ex'].mean() / net_daily_df[factor + '_top_ex'].std() * np.sqrt(252)
        ind_dict['top_ex_Max_DrawDown'] = _cal_max_draw_down(list(net_daily_df[factor + '_top_ex_net']))
        ind_dict['bot_ex_Return'] = list(net_daily_df[factor + '_bot_ex_net'])[-1] ** (252/num)
        ind_dict['bot_ex_Sharpe'] = net_daily_df[factor + '_bot_ex'].mean() / net_daily_df[factor + '_bot_ex'].std() * np.sqrt(252)
        ind_dict['bot_ex_Max_DrawDown'] = _cal_max_draw_down(list(net_daily_df[factor + '_bot_ex_net']))
        ind_list.append(ind_dict)
    return ind_list


if __name__ == '__main__':
    trading_list = read_trading_date_data()
    original_data_to_filter_true = read_original_data(TEST_DATA_PART, True)
    original_data_to_filter_false = read_original_data(TEST_DATA_PART, False)

    loss_func = torch.nn.MSELoss(reduction='sum')  # only sum operation, not mean; if reduction='mean', then it will get average on both batch and features
    path = '../IC_test_and_plot_data_trade_order/model_checkpoint_in_part_{}.pt'.format(MODEL_CHECK_POINT)
    model, device = get_model()
    model.to(device)
    model.load_state_dict(torch.load(path))
    for name, each in model.named_parameters():
        each.requires_grad = False
        print(name, each, each.shape, each.requires_grad)
    data_picture, data_date, true_value = read_test_data(TEST_DATA_PART)
    data_picture, true_value = map(torch.tensor, (data_picture, true_value))
    y_new = torch.tensor([])
    for t in range(len(data_picture)):
        if t % 1000 == 0 and t != 0:
            print("Have already finished the prediction of {} items".format(t))
        torch.manual_seed(3407)  # since there are batch normalization, different batch shape will course different result
        xb, yb = preprocess(data_picture[t], true_value[t], device)
        predict = model(xb)
        y_new = torch.cat((y_new.double(), predict.cpu().detach().double()), 0)
    loss = loss_func(y_new.float(), true_value.float())

    # ic_test
    window_data = pd.DataFrame(data={'trading_date': data_date[:, 0], 'wind_code': data_date[:, 1], 'alphaNet': y_new, 'true': np.array(true_value)})
    ic, ic_avg = _calculate_ic(window_data, 'alphaNet')
    _plot_cumulative_RankIC(ic, 'alphaNet')

    # ic_neutralize_test
    window_data_two_neutralize, industry = _window_data_handle_two_neutralize(window_data)
    window_data_five_neutralize, industry = _window_data_handle_five_neutralize(window_data, data_picture)
    _preprocess_factor_two_neutralize(window_data_two_neutralize, 'alphaNet', industry)
    _preprocess_factor_five_neutralize(window_data_five_neutralize, 'alphaNet', industry)
    ic_two_factor_neutralize, ic_avg_two_factor_neutralize = _calculate_ic(window_data_two_neutralize, 'alphaNet')
    ic_five_factor_neutralize, ic_avg_five_factor_neutralize = _calculate_ic(window_data_five_neutralize, 'alphaNet')

    # stratification_test
    result_df, last_net = _stratification_test_one_window(ic.index.tolist(), ['alphaNet'], trading_list, window_data_two_neutralize)

    # return, sharp, max_draw_down
    result_index = _cal_net_indexes(result_df, ['alphaNet'])




