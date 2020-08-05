import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils.np_utils import to_categorical, normalize
from sklearn.utils import shuffle
import pickle
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tensorflow.keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.metrics import ConfusionMatrixDisplay

PATH = './dnn-model/'


def get_model(inputDim, outputDim):
    model = Sequential()
    model.add(Dense(inputDim, activation='relu', input_shape=(inputDim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(outputDim, activation='softmax'))
    print('Category')
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model


dropped_cols = ['Dst Port', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count', 'Fwd Byts/b Avg',
                    'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg']

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
y = encoder.fit_transform(y)
data_y = to_categorical(y)
data_x = normalize(X.to_numpy())
del [X, y]
inputDim = len(data_x[0])
outputDim = data_y.shape[1]
print(data_y.shape)

model = get_model(inputDim, outputDim)
model.summary()
model_json = model.to_json()
with open(PATH + "dnn-model.json", "w") as json_file:
    json_file.write(model_json)
plot_model(model, to_file=PATH + 'model-dnn.png', show_layer_names=True, show_shapes=True)
train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, stratify=data_y, test_size=0.2)
filepath=PATH + "dnn-weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit(x=train_x, y=train_y, epochs=10, batch_size=2000, validation_data=(val_x, val_y), verbose=2, callbacks=[checkpoint])
del [data_x, data_y, train_x, train_y, val_x, val_y]
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'], '--')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(PATH + 'dnn-history-acc.png')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], '--')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(PATH + 'dnn-history-loss.png')
plt.show()
print('Testing')
df_test = pd.read_csv('./dataset/idsX_test_clean.csv')
df_test = df_test.dropna()
df_test['Label'].replace(mask_label, inplace=True)
y = df_test.pop('Label')
X = df_test.drop(columns=dropped_cols, axis=1)
del [df_test]
X[X < 0] = 0
test_y = to_categorical(encoder.transform(y))
test_x = normalize(X.to_numpy())
del [X, y]
scores = model.evaluate(test_x, test_y, verbose=1)
print(model.metrics_names)
acc, loss = scores[1]*100, scores[0]*100
print('accuracy: {:.3f}%: loss: {:.2f}'.format(acc, loss))

pred_y = model.predict(test_x)
pred_y = np.argmax(pred_y, axis=1)
true_y = np.argmax(test_y, axis=1)

print('Confusion matrix')
matrix = confusion_matrix(true_y, pred_y)
print(matrix)
tn = matrix[0][0]
fn = matrix[1][0]
fp = matrix[0][1]
tp = matrix[1][1]
acc = (tp + tn) / (tn + fn + fp + tp)
pre = tp / (tp + fp)
rec = tp / (tp + fn)
f1 = 2 * ((pre * rec) / (pre + rec))
print('{} \t {:.2f} \t {:.2f} \t {:.2f} \t {:.2f} \t'.format('DNN', acc * 100, pre * 100, rec * 100, f1 * 100))


