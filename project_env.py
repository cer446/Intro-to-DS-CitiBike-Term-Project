import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_datetime(df):
    # Turn date columns from string to datetime64
    date_columns = [
        'last_reported',
        'traffic_0_asof',
        'traffic_1_asof',
        'traffic_2_asof'
    ]
    for c in date_columns:
        df[c] = pd.to_datetime(df[c], infer_datetime_format=True)

def scale_bikes_and_docks(data, capacity):
    data['num_bikes_available_scaled'] = data['num_bikes_available'] / capacity
    data['num_bikes_disabled_scaled'] = data['num_bikes_disabled'] / capacity
    data['num_docks_available_scaled'] = data['num_docks_available'] / capacity
    data['num_docks_disabled_scaled'] = data['num_docks_disabled'] / capacity

def get_time_features(df):
    df['day_of_week'] = df['last_reported'].dt.dayofweek
    df['hour_of_day'] = df['last_reported'].dt.hour + df['last_reported'].dt.minute / 60
    df['is_weekend'] = (df['day_of_week'] >= 5) * 1.0

def set_outdated_traffic_info_to_mean(data, plot=True):
    # Hide traffic information that is outdated
    limit_m = 5.0
    for i in [0, 1, 2]:
        speed_col = 'traffic_{}_speed'.format(i)
        datediff = data['last_reported'] - data['traffic_{}_asof'.format(i)]
        datediff_m = datediff / np.timedelta64(1, 'm')
        have_data = datediff_m < limit_m
        good_speeds = data[have_data][speed_col]
        mean = np.mean(good_speeds)
        if plot:
            print(speed_col, 'good records:', len(good_speeds), '- mean:', mean)

        speeds = data[speed_col].copy()
        speeds[~have_data] = mean
        data[speed_col + '_scrub'] = speeds

    if plot:
        data[['traffic_0_speed_scrub', 'traffic_1_speed_scrub', 'traffic_2_speed_scrub']].hist()
        data[['traffic_0_speed_scrub', 'traffic_1_speed_scrub', 'traffic_2_speed_scrub']].describe()

def drop_columns(data):
    return data.drop([
        'Unnamed: 0',
        'eightd_has_available_keys',
        'summary',
        'traffic_0_linkId',
        'traffic_1_linkId',
        'traffic_2_linkId',
        'traffic_0_asof', # no longer useful
        'traffic_1_asof',
        'traffic_2_asof',
        'traffic_0_speed', # unclean speeds
        'traffic_1_speed',
        'traffic_2_speed',
        'time',
        'icon',
        'weather_ts',
        'is_installed',
        'is_renting',
        'is_returning',
        'nearestStormBearing',
        'precipIntensityError',
        'precipType',
        'num_bikes_available',
        'num_docks_available',
        'num_bikes_disabled',
        'num_docks_disabled',
        'num_docks_disabled_scaled',

        'last_reported',
        'station_id',

        'traffic_0_distance',
        'traffic_1_distance',
        'traffic_2_distance',
        'traffic_0_travel_time',
        'traffic_1_travel_time',
        'traffic_2_travel_time',
        ], axis=1)

def drop_target_columns(data, leave_in='y_60m'):
    return data.drop([x for x in [
        'y_10m',
        'y_15m',
        'y_30m',
        'y_45m',
        'y_60m',
        'y_90m',
        'y_120m',
        ] if x != leave_in], axis=1)

def bucket_y_variable(
        data,
        target='y_60m',
        threshold_empty=.05,
        threshold_full=.95,
        plot=True):
    data = drop_target_columns(data, leave_in=target)
    raw_target = data[target].copy()
    raw_target[raw_target < 0] = None
    if plot:
        plt.figure()
        plt.plot(raw_target)
        plt.show()

    # Filter to data points that have valid target variables
    data_valid = data[data[target] >= 0].copy()
    if plot:
        print('# rows dropped because of missing Y: {} / {}'.format(
            len(data) - len(data_valid), len(data)))

    yvar = np.zeros(len(data_valid))
    yvar[(data_valid[target] < threshold_empty).as_matrix()] = -1
    yvar[(data_valid[target] > threshold_full).as_matrix()] = 1
    data_valid['y'] = yvar
    if plot:
        plt.figure()
        plt.plot(data_valid[target].as_matrix(), 'g')
        plt.plot(yvar, 'b')
        plt.ylim(-3, 3)
        plt.show()

    return data_valid.drop([target], axis=1)

def load(station_id, log=True):
    data = pd.read_csv('per_station/{}.csv'.format(station_id))
    if log:
        print('# of raw records:', len(data))
        #print(data.columns)
    parse_datetime(data)
    scale_bikes_and_docks(data, 61)
    get_time_features(data)

    if log:
        data_random_slice = data.iloc[np.random.permutation(len(data))[:15]]
        print(data_random_slice[[
            'last_reported', 
            'num_bikes_available_scaled', 
            'day_of_week', 
            'hour_of_day',
            'is_weekend']])

    set_outdated_traffic_info_to_mean(data, plot=log)
    data_drop = drop_columns(data)
    if log:
        #print(data_drop.as_matrix().shape)
        print(data_drop.columns)

    return data_drop

def split(data, train=0.7, dev=0.2, inorder=True, log=True):
    assert train + dev <= 1.0

    if not inorder:
        shuffle = data.copy()
        shuffle = shuffle.iloc[np.random.permutation(len(shuffle))]
        shuffle.reset_index(drop=True)
        data = shuffle

    train_size = int(train * len(data))
    dev_size = int(dev * len(data))
    test_size = len(data) - train_size - dev_size

    print('(train, dev, test):', (train_size, dev_size, test_size))

    training = data[:train_size]
    dev = data[train_size:(train_size + dev_size)]
    test = data[(train_size + dev_size):]

    def make_xy(dataset):
        dX = dataset.drop('y', axis=1)
        dy = dataset['y']
        return dX, dy

    training_X, training_y = make_xy(training)
    dev_X, dev_y = make_xy(dev)
    test_X, test_y = make_xy(test)

    return {
            'train': (training_X, training_y),
            'dev': (dev_X, dev_y),
            'test': (test_X, test_y),
            }

def merge_training(split_data, more_training):
    more_split = split(more_training, train=1.0, dev=0.0, log=False)

    orig_t_X, orig_t_y = split_data['train']
    more_t_X, more_t_y = more_split['train']

    merged_X = pd.concat([orig_t_X, more_t_X])
    merged_y = pd.concat([orig_t_y, more_t_y])

    return {
            'train': (merged_X, merged_y),
            'dev': split_data['dev'],
            'test': split_data['test']
            }
