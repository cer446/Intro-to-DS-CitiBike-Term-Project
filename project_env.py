import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytz

def parse_datetime(df):
    eastern = pytz.timezone('US/Eastern')
    # Turn date columns from string to datetime64
    date_columns = [
        'last_reported',
        'traffic_0_asof',
        'traffic_1_asof',
        'traffic_2_asof'
    ]
    for c in date_columns:
        timestamps = df[c].as_matrix()
        dates = pd.to_datetime(timestamps, unit='s').tz_localize(pytz.utc)
        dates_local = dates.tz_convert(eastern)
        df[c] = dates_local

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
        'Unnamed: 0.1',
        'bike_timestamp',
        'bike_station_id',
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
        #'time',
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

        #'last_reported',
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
        threshold_empty=.1,
        threshold_full=.9,
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

def load(station_id, mode='train', log=True):
    data = pd.read_csv('per_station_{}/{}.csv'.format(mode, station_id))
    if log:
        print('# of raw records:', len(data))
        #print(data.columns)
    parse_datetime(data)
    scale_bikes_and_docks(data, 61)
    get_time_features(data)

    if log:
        data_random_slice = data.iloc[np.random.permutation(len(data))[:3]]
        print(data_random_slice.T)

    set_outdated_traffic_info_to_mean(data, plot=log)
    data_drop = drop_columns(data)
    if log:
        #print(data_drop.as_matrix().shape)
        print(data_drop.columns)

    return data_drop.dropna()

def load_split_bucket(
        station_id, 
        target='y_60m',
        threshold_empty=.1,
        threshold_full=.9,
        log=False):

    def load_and_bucket(mode):
        return bucket_y_variable(
            load(station_id, mode=mode, log=log),
            target=target,
            threshold_empty=threshold_empty,
            threshold_full=threshold_full,
            plot=log)

    train = load_and_bucket('train')
    dev = load_and_bucket('dev')
    test = load_and_bucket('test')

    def make_xy(dataset):
        dX = dataset.drop(['y', 'last_reported'], axis=1)
        dy = dataset['y']
        dt = dataset['last_reported']
        return dX, dy, dt

    training_X, training_y, training_t = make_xy(train)
    dev_X, dev_y, dev_t = make_xy(dev)
    test_X, test_y, test_t = make_xy(test)

    return {
            'train': (training_X, training_y),
            'dev': (dev_X, dev_y),
            'test': (test_X, test_y),
            'train_times': training_t,
            'dev_times': dev_t,
            'test_times': test_t
            }

def merge_training(split_data, more_split):
    orig_t_X, orig_t_y = split_data['train']
    more_t_X, more_t_y = more_split['train']

    merged_X = pd.concat([orig_t_X, more_t_X])
    merged_y = pd.concat([orig_t_y, more_t_y])

    return {
            'train': (merged_X, merged_y),
            'dev': split_data['dev'],
            'test': split_data['test'],
            'train_times': split_data['train_times'],
            'dev_times': split_data['dev_times'],
            'test_times': split_data['test_times'],
            }

def binarize(data, target=1):
    '''Set target to 1 to predict full vs not full (1 = full), or -1 to
    predict empty vs not empty (1 = empty)'''
    def binarize_split(split):
        X, y = split
        return (X, (y == target).astype(np.int64))
    
    return {
        'train': binarize_split(data['train']),
        'dev': binarize_split(data['dev']),
        'test': binarize_split(data['test']),
        'train_times': data['train_times'],
        'dev_times': data['dev_times'],
        'test_times': data['test_times'],
    }

def max_precision_for_recall(curve, target_recall=0.95):
    '''Returns the max precision, threshold and recall of that precision,
    across all models with recall at least a certain value.'''

    precisions, recalls, thresholds = curve
    max_pre, max_rec, max_thresh = (-1, None, None)
    # The list is ordered on decreasing recall.
    for i, precision in enumerate(precisions):
        if recalls[i] < target_recall:
            break
        if precision > max_pre:
            # Sacrifice some recall to increase precision.
            max_pre, max_rec, max_thresh = precision, recalls[i], thresholds[i]
    # The model with the highest precision with recall >= target_recall
    return max_pre, max_rec, max_thresh

def precision_at_recall(curve, target_recall=0.95):
    '''Returns the precision, threshold and recall nearest (while still greater)
    to a target recall value.'''

    precisions, recalls, thresholds = curve
    pre, rec, thresh = (None, None, None)
    for i, precision in enumerate(precisions):
        if recalls[i] < target_recall:
            break
        pre, rec, thresh = precision, recalls[i], thresholds[i]
    return pre, rec, thresh
