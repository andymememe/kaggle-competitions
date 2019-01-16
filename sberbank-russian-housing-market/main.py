import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler

def main():
    # Init Model
    imputer = SimpleImputer(strategy='median')
    gbr = GradientBoostingRegressor()
    param_grid = {'n_estimators':[100],
                  'learning_rate': [0.1, 0.05, 0.02],
                  'max_depth':[4, 6],
                  'min_samples_leaf':[3, 5, 9, 17],
                  'max_features':[1.0, 0.3, 0.1]
    }
    clf = GridSearchCV(estimator=gbr,
                       cv=5,
                       param_grid=param_grid,
                       n_jobs=-1)
    rs = RobustScaler()

    # Import data
    trainSet = pd.read_csv('input/train.csv',
                           index_col='id', parse_dates=['timestamp'],
                           infer_datetime_format=True)
    testSet = pd.read_csv('input/test.csv',
                          index_col='id', parse_dates=['timestamp'],
                          infer_datetime_format=True)
    macroSet = pd.read_csv('input/macro.csv',
                           index_col='timestamp', parse_dates=['timestamp'],
                           infer_datetime_format=True)
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
    trainY = trainSet.pop('price_doc').values
    trainX = trainSet.values
    testX = testSet.values

    # Preprocessing
    print('Preprocessing...')
    trainX = imputer.fit_transform(trainX)
    testX = imputer.transform(testX)
    trainX = rs.fit_transform(trainX)
    testX = rs.transform(testX)
    
    # Train
    print('Training...')
    clf.fit(trainX, trainY)
    print('Total score:', clf.best_score_)
    
    # Test
    print('Testing...')
    testY = clf.predict(testX)
    with open('output/submission.csv', 'w') as f:
        f.write('id,price_doc\n')
        for id, result in zip(testIndices, testY):
            f.write('{},{}\n'.format(id, result))

if __name__ == '__main__':
    main()
