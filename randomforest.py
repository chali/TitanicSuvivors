import pandas as pd
import numpy as np
import csv as csv
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

def prepared_data(data_frame):
    numerical_feature_names = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_feature_names = ['Embarked', 'Pclass']
    sex_feature = np.asmatrix(data_frame['Sex'] == 'male').T.astype(np.float)

    categorical_features = data_frame[categorical_feature_names]
    encoder = DictVectorizer(sparse=False)
    categorical_features_encoded = encoder.fit_transform(categorical_features.T.to_dict().values())

    numerical_features = data_frame[numerical_feature_names].as_matrix().astype(np.float)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    numerical_features_without_nan = imp.fit_transform(numerical_features)
    numerical_features_scaled = preprocessing.scale(numerical_features_without_nan)
    numerical_features_normalized = preprocessing.normalize(numerical_features_scaled, norm='l2')

    return np.concatenate((numerical_features_normalized, categorical_features_encoded, sex_feature), axis= 1)

data = pd.read_csv("train.csv")
Y_data = data['Survived'].as_matrix().astype(np.int)
X_data = prepared_data(data)

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X_data, Y_data, test_size=0.3, random_state=0)

cls = RandomForestClassifier(n_estimators=100)
cls = cls.fit(X_train, Y_train)
print('Accuracy: ' + cls.score(X_test, Y_test).astype('str'))

testData = pd.read_csv("test.csv")
X_test = prepared_data(testData)
Y_pred = cls.predict(X_test)

ids = testData['PassengerId']
output_file = open("prediction.csv", "wb")
open_file_object = csv.writer(output_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, Y_pred))
output_file.close()
print "Done"