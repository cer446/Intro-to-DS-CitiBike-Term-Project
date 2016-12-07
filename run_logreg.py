import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import math
import project_env
import json
import os

'''Loads and Processes Stations.Json file'''

with open('stations.json') as data_file:    
    data = json.load(data_file)
stations = pd.read_json('stations.json', 'index').T
stations_df = pd.DataFrame.from_dict(stations['stationBeanList'][0], orient = 'index').T
for i in range(1, 664):
    stations_df = pd.concat([stations_df, pd.DataFrame.from_dict(stations['stationBeanList'][i], orient = 'index').T])
stations_df = stations_df.set_index('id')

files = os.listdir('per_station_dev')

station_ids = []
for i in range(0, len(files)):
    station_ids.append(int(files[i][0:len(files[i])- 4]))
    
stations_df = stations_df.loc[station_ids, :]

def distance(station_a_id, station_b_id):
    '''Finds distance between two stations'''
    longitude_difference = stations_df.loc[station_a_id, 'longitude'] - stations_df.loc[station_b_id, 'longitude']
    latitude_difference = stations_df.loc[station_a_id, 'latitude'] - stations_df.loc[station_b_id, 'latitude']
    distance = math.sqrt(longitude_difference**2 + latitude_difference**2)
    return distance

def closest_stations(station_a_id, num):
    '''Finds closest station based on distance'''
    distances = pd.Series(data=None, index = stations_df.index)
    for station_id in stations_df.index:
        distances[station_id] = distance(station_a_id, station_id)
    distances = distances.sort_values()
    return distances.iloc[1:num+1]

def read_more_data(station_id, target):
    return project_env.load_split_bucket(station_id, target=target)

def add_closest_stations(split_data, station_id, target, num_stations=10, empty=True):
    '''takes split_data for one station and its station_id, splits, merges the closest stations' data and binarizes all'''
    station_ids_to_concat = list(closest_stations(station_id, num_stations).index[0:num_stations])
    station_ids_data = [read_more_data(sid, target) for sid in station_ids_to_concat]
    appended_multiple = split_data
    for df in station_ids_data:
        appended_multiple = project_env.merge_training(appended_multiple, df)
    binary = (empty==True)*-1 + (empty!=True)*1
    appended_multiple = project_env.binarize(appended_multiple, binary)
    return appended_multiple

def append_binarize(specs):
    '''Prepares the logistic regression with the input parameters'''
    
    if specs.num_append > 0:
        data = add_closest_stations(specs.split_data, specs.stationid, specs.target, num_stations=specs.num_append, empty=specs.empty)
    
    if specs.empty == True and specs.num_append==0:
        data = project_env.binarize(specs.split_data, -1) #the append_one function binarizes the data
    elif specs.empty == False and specs.num_append==0:
        data = project_env.binarize(specs.split_data, 1)
    
    return data

def do_logreg(specs, plot=False):

    split_binarized_data = append_binarize(specs)
    
    train_X, train_y = split_binarized_data['train']
    dev_X, dev_y = split_binarized_data['dev']
    
    logreg = LogisticRegression(penalty=specs.penalty, C=specs.C)
    scaler = sklearn.preprocessing.StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    
    if specs.squares:
        train_X_scaled = np.concatenate([
                train_X_scaled,
                np.square(train_X_scaled)], axis=1)
    print('X shape:', train_X_scaled.shape)
    
    #print(pd.DataFrame(train_X_scaled).describe().T)
    logreg.fit(train_X_scaled, train_y)

    dev_X_scaled = scaler.transform(dev_X)
    if specs.squares:
        dev_X_scaled = np.concatenate([
                dev_X_scaled,
                np.square(dev_X_scaled)], axis=1)
        
    dev_pred = logreg.predict(dev_X_scaled)
    dev_decision = logreg.predict_proba(dev_X_scaled)[:,1]
    acc = sklearn.metrics.accuracy_score(dev_y, dev_pred)
    print('Evaluating on dev set of {} examples'.format(len(dev_y)))
    print('Accuracy:', acc)
    
    print(sklearn.metrics.confusion_matrix(dev_y, dev_pred))

    if plot:
        plt.figure()
        plt.plot(dev_y.as_matrix()[100:500], 'b')
        plt.plot(dev_pred[100:500], 'g')
        plt.plot(0.5 * (dev_y.as_matrix() - dev_pred)[100:500], 'r')
        plt.ylim(-3, 3)
        plt.show()
    
    return logreg, scaler, dev_decision


def format_plot(target_recall, empty=True):
    if empty==True:
        plt.xlim([0.9, 1.0])
        plt.ylim([0.4, 0.7])
    if empty==False:
        plt.xlim([0.9, 1.0])
        plt.ylim([0, 0.7])
    plt.axvline(x=target_recall, color='k', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.title('Precision-Recall Curves')