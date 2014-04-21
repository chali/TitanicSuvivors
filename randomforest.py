import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

def prepared_data(data_frame):
    numerical_feature_names = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_feature_names = ['Embarked', 'Pclass', 'Sex']

    categorical_features = data_frame[categorical_feature_names]
    encoder = DictVectorizer(sparse=False)
    categorical_features_encoded = encoder.fit_transform(categorical_features.T.to_dict().values())

    numerical_features = data_frame[numerical_feature_names].as_matrix().astype(np.float)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    numerical_features_without_nan = imp.fit_transform(numerical_features)
    numerical_features_scaled = preprocessing.scale(numerical_features_without_nan)

    return np.concatenate((numerical_features_scaled, categorical_features_encoded), axis= 1)

trainData = pd.read_csv("train.csv")
Y_train = trainData['Survived']
X_train = prepared_data(trainData)

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(X_train, Y_train.as_matrix())

testData = pd.read_csv("test.csv")
X_test = prepared_data(testData)
Y_pred = forest.predict(X_test)

ids = testData['PassengerId']
output_file = open("prediction.csv", "wb")
open_file_object = csv.writer(output_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, Y_pred))
output_file.close()
print "Done"