"""NOTES: Batch data is different each time in keras, which result in slight differences in results."""
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, MaxPooling2D, Reshape, multiply, Conv2D, GlobalAveragePooling2D, Dense, Multiply
from keras.models  import  Model, load_model
from tensorflow.keras.layers import Input
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def plot(history):
    """Plot performance curve"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["loss"], "r-", label="Train")
    axes[0].plot(history["val_loss"], "b-", label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[1].plot(history["accuracy"], "r-",  label="Train")
    axes[1].plot(history["val_accuracy"], "b-", label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig('Training.png', dpi=fig.dpi)
    

fs = 100
def load_data(path):
    # No processing
    x_train = np.load(path +'x_tr.npy',allow_pickle=True)
    Y_train = np.load(path + 'y_tr.npy',allow_pickle=True)
    x_test = np.load(path + 'x_te.npy',allow_pickle=True)
    Y_test = np.load(path + 'y_te.npy',allow_pickle=True)
    
    x_train = x_train.transpose(0, 2, 1)  # transpose working correctly
    x_test = x_test.transpose(0, 2, 1)
    x_train = x_train.reshape(x_train.shape[0], 12, 1000, 1)  # Add another channel
    x_test = x_test.reshape(x_test.shape[0], 12, 1000, 1)

    dir ={'NORM':0,'MI':1,'STTC':2,'CD':3,'HYP':4}
    y_test = np.zeros((x_test.shape[0],5))
    for i in range(x_test.shape[0]):
        for j in Y_test[i]:
            idx = dir[j]
            y_test[i][idx]=1

    y_train = np.zeros((x_train.shape[0],5))
    for i in range(x_train.shape[0]):
        for j in Y_train[i]:
            idx = dir[j]
            y_train[i][idx]=1
            
    # add another channel
    x_train = x_train.reshape(x_train.shape[0], 12, 1000, 1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    x_test  = x_test.reshape(x_test.shape[0], 12, 1000, 1)
    
    print("x_train :", x_train.shape)
    print("y_train :", y_train.shape)
    print("x_val :", x_val.shape)
    print("y_val :", y_val.shape)
    print("x_test  :", x_test.shape)
    print("y_test  :", y_test.shape)
    print('Data loaded')
    return x_train, x_val, x_test, y_train, y_val, y_test

def lr_schedule(epoch, lr):
    if epoch > 30 and (epoch - 1) % 10 == 0:
        lr *= 0.1
    print("Learning rate: ", lr)
    return lr


def create_model(input_shape, weight=1e-3):
    input1 = Input(shape=input_shape)
    x1 = Conv2D(16, kernel_size=11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(input1)
    x1 = Conv2D(24, kernel_size=11, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x1)
    x1 = MaxPooling2D(pool_size=3, padding="same")(x1)
    x1 = Conv2D(32, kernel_size=11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x1)
    x1 = MaxPooling2D(pool_size=5, padding="same")(x1)

    # Channel-wise attention module
    squeeze = GlobalAveragePooling2D()(x1)
    excitation = Dense(16, activation='relu')(squeeze)
    excitation = Dense(32, activation='sigmoid')(excitation)
    excitation = Reshape((1,1,32))(excitation)
    scale = Multiply()([x1, excitation])
    x = GlobalAveragePooling2D()(scale)
    dp = Dropout(0.5)(x)
    outputs = Dense(5, activation='sigmoid')(dp)
    model = Model(inputs=input1, outputs=outputs)
    return model

if __name__ == "__main__":
    # load_data
    path = '/scratch/xc2627/DS/1.0.1/'
    x_train, x_val, x_test, y_train, y_val, y_test= load_data(path)
    # training
    model = create_model(x_train.shape[1:])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = 'weights.best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    lr_scheduler = LearningRateScheduler(lr_schedule)
    callbacks_list = [lr_scheduler, checkpoint]
    start_time = time.time()
    history = model.fit(x_train, y_train, batch_size=128, epochs=100,
                         validation_data=(x_val, y_val), callbacks=callbacks_list)
    
    plot(history.history)
    model = load_model(filepath)
   
    
    from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, accuracy_score, auc


    def sklearn_metrics(y_true, y_pred):
        y_bin = np.copy(y_pred)
        y_bin[y_bin >= 0.5] = 1
        y_bin[y_bin < 0.5] = 0

        # Compute area under precision-Recall curve
        auc_sum = 0
        for i in range(5):
            precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
            auc_sum += auc(recall, precision)

        print("Accuracy        : {:.2f}".format(accuracy_score(y_true.flatten(), y_bin.flatten()) * 100))
        print("Macro AUC score : {:.2f}".format(roc_auc_score(y_true, y_pred, average='macro') * 100))
        print('AUPRC           : {:.2f}'.format((auc_sum / 5) * 100))
        print("Micro F1 score  : {:.2f}".format(f1_score(y_true, y_bin, average='micro') * 100))
    
#     start_time = time.time()
    y_pred_train = model.predict(x_train)
    y_pred_test  = model.predict(x_test)
#     end_time = time.time()
    
#     print('time cost:%d s'%(end_time-start_time))

    print("Train")
    sklearn_metrics(y_train, y_pred_train)
    print("\nTest")
    sklearn_metrics(y_test, y_pred_test)
