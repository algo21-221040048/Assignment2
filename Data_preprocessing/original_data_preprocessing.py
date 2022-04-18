# This part of code is used to cleaning and handling the original data
# Input: original data
# Output: data picture and y
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as f
import joblib


# Hyper parameters
TRADING_DAY_EVERY_YEAR = 252  # 全年交易日
TRAIN_UPDATE_DAYS = 122  # 每半年滚动一次
FREQUENCY = 2  # 训练采样频率
FREQUENCY_BACKTEST = 5  # 回测采样频率
PERIOD = 5  # 5日收益率


def read():
    """
    This function is used to read the original data from directory `OriginalData_csv`
    """
    trading_list = pd.read_csv('../OriginalData_csv/Trading_date.csv', header=0, index_col=False)
    trading_list = trading_list['trading_date'].tolist()

    x_data_without_return1 = pd.read_csv('../OriginalData_csv/x_data_without_return1.csv', header=0,
                                         index_col='trading_date')
    if 'Unnamed: 0' in x_data_without_return1.columns:
        x_data_without_return1 = x_data_without_return1.drop(['Unnamed: 0'], axis='columns')

    filter_data = pd.read_csv('../OriginalData_csv/filter_data.csv', header=0, index_col='trading_date',
                              low_memory=False)
    if 'Unnamed: 0' in filter_data.columns:
        filter_data = filter_data.drop(['Unnamed: 0'], axis='columns')

    delist_data = pd.read_csv('../OriginalData_csv/delist_data.csv', header=0, low_memory=False)
    if 'Unnamed: 0' in delist_data.columns:
        delist_data = delist_data.drop(['Unnamed: 0'], axis='columns')
    # delist_data.dropna(how='any', inplace=True)

    ev_data = pd.read_csv('../OriginalData_csv/ev_data.csv', header=0, low_memory=False)
    if 'Unnamed: 0' in ev_data.columns:
        ev_data = ev_data.drop(['Unnamed: 0'], axis='columns')

    industry_data = pd.read_csv('../OriginalData_csv/industry_data.csv', header=0, low_memory=False)
    if 'Unnamed: 0' in industry_data.columns:
        industry_data = industry_data.drop(['Unnamed: 0'], axis='columns')

    return trading_list, x_data_without_return1, filter_data, delist_data, ev_data, industry_data


def calculate_return1(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to calculate the column return1
    :param df: df
    """
    df_new = df.copy()  # returning a view versus a copy
    df_new['close_adj'] = df_new.groupby('trading_date').apply(lambda x: x['close_price'] * x['adj_factor']).reset_index(drop=True)
    df_new['return1'] = (df_new.groupby('wind_code'))['close_adj'].apply(lambda x: x.pct_change(fill_method=None)).reset_index(drop=True)
    df_new = df_new.drop(['adj_factor'], axis='columns')
    return df_new


def calculate_return_bn(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function must be used after `calculate_return1` function in order to calculate the return5 and return5_bn
    :param df: df
    """
    df_new = df.copy()
    df_new['return5'] = (df_new.groupby('wind_code'))['close_adj'].apply(lambda x: x.pct_change(periods=PERIOD, fill_method=None)).reset_index(drop=True)
    df_new['return5'] = (df_new.groupby('wind_code'))['return5'].shift(periods=-PERIOD)
    df_new['return_bn'] = (df_new.groupby('trading_date'))['return5'].transform(lambda x: (x - x.mean())/x.std(ddof=0))
    return df_new


def data_cleaning(df: pd.DataFrame, delist: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to filter the ST and PT stocks
    :param df: df
    :param delist: 摘牌日期, 公司股票终止上市并摘牌
    """
    df_new = df.copy()
    # find st
    st = df_new[df_new['is_st'] == 1.0]['wind_code'].unique().tolist()

    # find pt
    stock_delist_date = delist[delist['delist_date'] != '1899-12-30']
    stock_delist_date = stock_delist_date[
        (stock_delist_date['delist_date'] <= '2020-05-29') & (stock_delist_date['delist_date'] >= '2011-01-31')]
    pt = stock_delist_date['wind_code'].unique().tolist()

    # delete st, pt stocks
    df_new = df_new[~df_new['wind_code'].isin(set(st + pt))]
    return df_new.reset_index(drop=True)


def sp_handle(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to handle special case: '停牌一天', 把它们的开盘和收盘价都设置成nan，然后进行填充
    :param df: df
    """
    df.loc[((df['trade_status'] == '停牌一天') & (~df['close_price'].isna())), "open_price":"close_price"] = np.nan
    return df


def processing(x: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to handle with the NaN in return1 and return5 group by wind_code
    :param x: df
    """
    x = x.drop(['susp_days', 'susp_reason'], axis='columns')
    x = x.fillna(-1000)
    # x = x[~((x['return1'].isna()) | (x['return_bn'].isna()))].reset_index(drop=True)
    x = x.reset_index(drop=True)
    return x


def get_new_data(x_date: np.array, x: torch.tensor, y: torch.tensor) -> (torch.tensor, np.array, torch.tensor):
    """
    此函数的目的为让杂乱的数据，重新排序成按照交易日顺序的股票排列，并输出对应的日期和股票代码作为索引，方便后续查找
    This function is used to rearrange the data according with date
    :param x_date: 数据图片对应的索引(trading_date 和 wind_code)
    :param x: 数据图片
    :param y: 数据图片对应的输出(5日收益率)
    """
    time_step = np.sort(np.unique(x_date[:, 0]))
    x_new = torch.empty((0, 270))
    x_date_new = np.empty((0, 2))
    y_new = torch.tensor([])
    for each_time in time_step:
        x_medium = x[x_date[:, 0] == each_time]
        x_date_medium = x_date[x_date[:, 0] == each_time]
        y_medium = y[x_date[:, 0] == each_time]
        assert x_medium.shape[0] == y_medium.shape[0]
        assert x_medium.shape[0] == x_date_medium.shape[0]
        x_new = torch.cat((x_new.double(), x_medium), 0)
        x_date_new = np.concatenate((x_date_new, x_date_medium), 0)
        y_new = torch.cat((y_new.double(), y_medium), 0)
    return x_new, x_date_new, y_new


if __name__ == '__main__':
    trading_list, x_data_without_return1, filter_data, delist_data, ev_data, industry_data = read()
    data = pd.merge(x_data_without_return1.reset_index(), filter_data.reset_index(), how='left', on=['trading_date', 'wind_code'])

    # Data preparing
    data = data_cleaning(data, delist_data)
    data = sp_handle(data)
    data = calculate_return1(data)
    data = calculate_return_bn(data)
    data = data.groupby('wind_code').apply(lambda x: processing(x))

    # Every 1500 trading_date one period
    train_start_list = trading_list[:-1500:TRAIN_UPDATE_DAYS]
    train_end_list = []
    for each in train_start_list:
        train_end_list.append(trading_list[trading_list.index(each) + 1500])
    joblib.dump(train_start_list, 'train_start_list.pkl')
    joblib.dump(train_end_list, 'train_end_list.pkl')

    # Preparing train data: frequency = 2,
    for i in range(len(train_start_list)):
        PATH_X = 'data_train_x_part_{}.pkl'.format(i+1)
        PATH_X_DATE = 'data_train_x_date_part_{}.pkl'.format(i + 1)
        PATH_Y = 'data_train_y_part_{}.pkl'.format(i+1)
        data_x = torch.empty((0, 270))
        data_x_date = np.empty((0, 2))
        data_y = torch.tensor([])
        medium = data[(data['trading_date'] >= train_start_list[i]) & (data['trading_date'] < train_end_list[i])]
        for each in medium['wind_code'].unique().tolist():
            sub_df = medium.loc[each]
            sub_df_length = sub_df.shape[0]
            if sub_df_length < 30:
                continue
            else:
                # Handle x
                sub_df_x = np.array(sub_df[['open_price', 'high_price', 'low_price', 'close_price', 'vwap', 'volume', 'return1', 'turn', 'free_turn']]).T
                sub_df_x_date = np.array(sub_df.iloc[29::FREQUENCY][['trading_date', 'wind_code']])
                sub_df_x = torch.from_numpy(sub_df_x)

                # Split x
                split_sub_df_x = f.unfold(sub_df_x.unsqueeze(0).unsqueeze(0), kernel_size=(9, 30), stride=(1, FREQUENCY))
                B, W, L = split_sub_df_x.size()
                split_sub_df_x = split_sub_df_x.permute(0, 2, 1)
                split_sub_df_x = split_sub_df_x.squeeze(0)

                # Handle y
                sub_df_y = np.array(sub_df.iloc[29::FREQUENCY]['return5'])
                split_sub_df_y = torch.from_numpy(sub_df_y)

                assert split_sub_df_x.shape[0] == split_sub_df_y.shape[0]
                assert split_sub_df_x.shape[0] == sub_df_x_date.shape[0]
                # filter x,y where x contains -1000
                truncate_index = np.unique(np.where(split_sub_df_x == -1000)[0])
                split_sub_df_x = np.delete(split_sub_df_x, truncate_index, 0)
                sub_df_x_date = np.delete(sub_df_x_date, truncate_index, 0)
                split_sub_df_y = np.delete(split_sub_df_y, truncate_index, 0)
                # filter x,y where y contains -1000
                truncate_index = np.unique(np.where(split_sub_df_y == -1000)[0])
                split_sub_df_x = np.delete(split_sub_df_x, truncate_index, 0)
                sub_df_x_date = np.delete(sub_df_x_date, truncate_index, 0)
                split_sub_df_y = np.delete(split_sub_df_y, truncate_index, 0)

                assert split_sub_df_x.shape[0] == split_sub_df_y.shape[0]
                assert split_sub_df_x.shape[0] == sub_df_x_date.shape[0]
                data_x = torch.cat((data_x.double(), split_sub_df_x), 0)
                data_x_date = np.concatenate((data_x_date, sub_df_x_date), 0)
                data_y = torch.cat((data_y.double(), split_sub_df_y), 0)
                # print(data_x.shape)
                # print(data_y.shape)

        data_x_new, date_x_new, data_y_new = get_new_data(data_x_date, data_x, data_y)
        if i == 0:
            print(data_x_new[486695])
        assert data_x_new.shape == data_x.shape
        assert date_x_new.shape == data_x_date.shape
        assert data_y_new.shape == data_y.shape
        joblib.dump(np.array(data_x_new), PATH_X)
        joblib.dump(date_x_new, PATH_X_DATE)
        joblib.dump(np.array(data_y_new), PATH_Y)

        print("Training data part {} finished !".format(i+1))

    # Preparing test data: frequency = 5
    for i in range(len(train_end_list) - 1):
        PATH_X = 'data_test_x_part_{}.pkl'.format(i + 1)
        PATH_X_DATE = 'data_test_x_date_part_{}.pkl'.format(i + 1)
        PATH_Y = 'data_test_y_part_{}.pkl'.format(i + 1)
        data_x = torch.empty((0, 270))
        data_x_date = np.empty((0, 2))
        data_y = torch.tensor([])
        medium = data[(data['trading_date'] >= train_end_list[i]) & (data['trading_date'] < train_end_list[i + 1])]
        for each in medium['wind_code'].unique().tolist():
            sub_df = medium.loc[each]
            sub_df_length = sub_df.shape[0]
            if sub_df_length != TRAIN_UPDATE_DAYS:
                continue
            else:
                # Handle x
                sub_df_x = np.array(sub_df[['open_price', 'high_price', 'low_price', 'close_price', 'vwap', 'volume',
                                            'return1', 'turn', 'free_turn']]).T
                sub_df_x_date = np.array(sub_df.iloc[29::FREQUENCY_BACKTEST][['trading_date', 'wind_code']])
                sub_df_x = torch.from_numpy(sub_df_x)

                # Split x
                split_sub_df_x = f.unfold(sub_df_x.unsqueeze(0).unsqueeze(0), kernel_size=(9, 30), stride=(1, FREQUENCY_BACKTEST))
                B, W, L = split_sub_df_x.size()
                split_sub_df_x = split_sub_df_x.permute(0, 2, 1)
                split_sub_df_x = split_sub_df_x.squeeze(0)

                # Handle y
                sub_df_y = np.array(sub_df.iloc[29::FREQUENCY_BACKTEST]['return5'])
                split_sub_df_y = torch.from_numpy(sub_df_y)

                assert split_sub_df_x.shape[0] == split_sub_df_y.shape[0]
                assert split_sub_df_x.shape[0] == sub_df_x_date.shape[0]
                # filter x,y where x contains -1000
                truncate_index = np.unique(np.where(split_sub_df_x == -1000)[0])
                split_sub_df_x = np.delete(split_sub_df_x, truncate_index, 0)
                sub_df_x_date = np.delete(sub_df_x_date, truncate_index, 0)
                split_sub_df_y = np.delete(split_sub_df_y, truncate_index, 0)
                # filter x,y where y contains -1000
                truncate_index = np.unique(np.where(split_sub_df_y == -1000)[0])
                split_sub_df_x = np.delete(split_sub_df_x, truncate_index, 0)
                sub_df_x_date = np.delete(sub_df_x_date, truncate_index, 0)
                split_sub_df_y = np.delete(split_sub_df_y, truncate_index, 0)

                assert split_sub_df_x.shape[0] == split_sub_df_y.shape[0]
                assert split_sub_df_x.shape[0] == sub_df_x_date.shape[0]
                data_x = torch.cat((data_x.double(), split_sub_df_x), 0)
                data_x_date = np.concatenate((data_x_date, sub_df_x_date), 0)
                data_y = torch.cat((data_y.double(), split_sub_df_y), 0)

        data_x_new, date_x_new, data_y_new = get_new_data(data_x_date, data_x, data_y)
        assert data_x_new.shape == data_x.shape
        assert date_x_new.shape == data_x_date.shape
        assert data_y_new.shape == data_y.shape
        joblib.dump(np.array(data_x_new), PATH_X)
        joblib.dump(date_x_new, PATH_X_DATE)
        joblib.dump(np.array(data_y_new), PATH_Y)

        print("Testing data part {} finished !".format(i + 1))





