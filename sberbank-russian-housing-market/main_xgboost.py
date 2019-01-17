import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
import xgboost as xgb

def main():
    # Init Model
    param = {'max_depth': 5,
             'eta': 0.1,
             'subsample': 0.8,
             'colsample_bytree': 0.8,
             'objective': 'reg:linear',
             'eval_metric': 'rmse',
             'gamma': 0,
             'silent': 1
            }
    num_round = 1000
    rs = RobustScaler()

    # Import data
    trainSet = pd.read_csv('input/train.csv',
                           index_col='id', parse_dates=['timestamp'],
                           infer_datetime_format=True,
                           true_values='yes', false_values='no')
    testSet = pd.read_csv('input/test.csv',
                          index_col='id', parse_dates=['timestamp'],
                          infer_datetime_format=True,
                          true_values='yes', false_values='no')
    macroSet = pd.read_csv('input/macro.csv',
                           index_col='timestamp', parse_dates=['timestamp'],
                           infer_datetime_format=True,
                           true_values='yes', false_values='no')
    fixSet = pd.read_excel('input/BAD_ADDRESS_FIX.xlsx') \
               .drop_duplicates('id') \
               .set_index('id')
    testIndices = testSet.index.values

    # Fixed data
    trainSet = pd.merge_ordered(trainSet, macroSet, on='timestamp', how='left')
    testSet = pd.merge_ordered(testSet, macroSet, on='timestamp', how='left')
    trainSet.update(fixSet, overwrite=True)
    testSet.update(fixSet, overwrite=True)
    badDataSet = set(trainSet[trainSet.kremlin_km < 0.1].index)
    trainSet.drop(badDataSet, inplace=True)

    # Drop useless data
    trainSet.drop(['timestamp'], axis=1, inplace=True)
    testSet.drop(['timestamp'], axis=1, inplace=True)
    for column in trainSet.drop(['price_doc'], axis=1).columns:
        if trainSet[column].dtype == type(object):
            colAll = pd.concat([trainSet[column].astype(str),
                                testSet[column].astype(str)])
            le = LabelEncoder()
            le.fit(colAll)
            trainSet[column] = le.transform(trainSet[column].astype(str))
            testSet[column] = le.transform(testSet[column].astype(str))
    testSet = testSet[trainSet.drop(['price_doc'], axis=1).columns.values]

    # Split X, Y
    trainY = np.log1p(trainSet.pop('price_doc').values)
    trainX = trainSet.values
    testX = testSet.values

    # Preprocessing
    print('Preprocessing...')
    trainX = rs.fit_transform(trainX)
    testX = rs.transform(testX)
    
    # Train
    print('Training...')
    trainData = xgb.DMatrix(trainX, label=trainY)
    record = xgb.cv(param, trainData, num_round,
                    early_stopping_rounds=20)
    print(record)
    bst = xgb.train(param, trainData, num_round)
    
    # Test
    print('Testing...')
    testX = xgb.DMatrix(testX)
    testY = np.exp(bst.predict(testX)) - 1
    with open('output/submission.csv', 'w') as f:
        f.write('id,price_doc\n')
        for id, result in zip(testIndices, testY):
            f.write('{},{}\n'.format(id, result))

if __name__ == '__main__':
    main()
