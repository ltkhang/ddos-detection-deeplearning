import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils.np_utils import to_categorical, normalize
from sklearn.utils import shuffle
import pickle
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
PATH = './ml-model/'


def train(model_name, model, data_x, data_y):
    print('Train', model_name)
    t1 = time.time()
    model.fit(data_x, data_y)
    print('Training time:', time.time() - t1)
    pickle.dump(model, open(PATH + model_name + '_model.pkl', 'wb'))
    return model


def test(model_name, model, test_x, test_y):
    pred_y = model.predict(test_x)
    matrix = confusion_matrix(test_y, pred_y)
    tn = matrix[0][0]
    fn = matrix[1][0]
    fp = matrix[0][1]
    tp = matrix[1][1]
    acc = (tp + tn) / (tn + fn + fp + tp)
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * ((pre * rec) / (pre + rec))
    print('{} \t {:.2f} \t {:.2f} \t {:.2f} \t {:.2f} \t'.format(model_name, acc * 100, pre * 100, rec * 100, f1 * 100))

dropped_cols = ['Dst Port', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count', 'Fwd Byts/b Avg',
                'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
                ]
mask_label = {
    'Benign': 'Benign',
    'DoS attacks-GoldenEye': 'DDoS-Attack',
    'DoS attacks-Slowloris': 'DDoS-Attack',
    'DoS attacks-SlowHTTPTest': 'DDoS-Attack',
    'DoS attacks-Hulk': 'DDoS-Attack',
    'DDoS attacks-LOIC-HTTP': 'DDoS-Attack',
    'DDOS attack-LOIC-UDP': 'DDoS-Attack',
    'DDOS attack-HOIC': 'DDoS-Attack'
}

print('Training')

train_csv = './dataset/idsX_train_clean.csv'
df = pd.read_csv(train_csv)
df = df.dropna()
df = shuffle(df)
df['Label'].replace(mask_label, inplace=True)
y = df.pop('Label')
X = df.drop(columns=dropped_cols, axis=1)
del [df]
X[X < 0] = 0
encoder = LabelEncoder()
data_y = encoder.fit_transform(y)
data_x = normalize(X.to_numpy())
del [X, y]

lin_svc = LinearSVC()
nb = GaussianNB()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

lin_svc = train('lsvm', lin_svc, data_x, data_y)
nb = train('nb', nb, data_x, data_y)
dt = train('dt', dt, data_x, data_y)
rf = train('rf', rf, data_x, data_y)
del [data_x, data_y]

print('Testing')
df_test = pd.read_csv('./dataset/idsX_test_clean.csv')
df_test = df_test.dropna()
df_test['Label'].replace(mask_label, inplace=True)
y = df_test.pop('Label')
X = df_test.drop(columns=dropped_cols, axis=1)
del [df_test]
X[X < 0] = 0
test_y = encoder.transform(y)
test_x = normalize(X.to_numpy())
print('Model \t Acc \t Pre \t Rec \t F1-score')
test('LSVM', lin_svc, test_x, test_y)
test('NB', nb, test_x, test_y)
test('DT', dt, test_x, test_y)
test('RF', rf, test_x, test_y)
del [X, y]

