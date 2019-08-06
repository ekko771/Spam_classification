import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, precision_recall_curve, confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from sklearn.utils.multiclass import unique_labels
import json
from model.lstm_model import Lstm_model
import matplotlib.pyplot as plt
import itertools
from utils.setup_logger import logger
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def read_config(file_name=None):
    with open(file_name, 'r') as reader:
        config = json.loads(reader.read())
    return config


def read_data(file_name=None, skip_row=[0]):
    data = pd.read_csv(file_name, skiprows=skip_row)
    return data


def train(config, x_train, x_test, y_train, y_test):
    model = Lstm_model(config)
    if config['save_model']:
        saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        iteration = int(len(X_train) / config['batch_size'])
        for epoch in range(config['epochs']):
            epoch_loss = 0
            i = 0
            for i in range(iteration):
                start = i
                end = i + config['batch_size']
                batch_x = np.array(
                    X_train[start:end]).reshape(-1, config['time_step'], config['num_feature'])
                batch_y = np.array(y_train[start:end]).reshape(-1, 1)
                _, loss = sess.run([model.train_op, model.loss], feed_dict={
                    model.xs: batch_x, model.ys: batch_y})
                epoch_loss += loss
                i += config['batch_size']

            logger.info("Epoch{} train loss is {}".format(epoch, epoch_loss))

            #logger.info("Epoch{} valid loss is {}".format(epoch, valid_loss))
        y_pred = tf.round(tf.nn.sigmoid(model.outputs)).eval(
            {model.xs: np.array(X_test).reshape(-1, config['time_step'], config['num_feature']), model.ys: np.array(y_test).reshape(-1, 1)})
        if 'saver' in locals():
            save_path = saver.save(sess, config['model_location'])
        f1 = f1_score(np.array(y_test), y_pred, average='macro')
        y_test = y_test
        y_pred = y_pred.astype(int).ravel()
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_true=y_test, y_pred=y_pred)
        precision = precision_score(y_true=y_test, y_pred=y_pred)
        logger.info("F1 Score: {}".format(f1))
        logger.info("Accuracy Score: {}".format(accuracy))
        logger.info("Recall: {}".format(recall))
        logger.info("Precision: {}".format(precision))
        ###
        train_pred = tf.round(tf.nn.sigmoid(model.outputs)).eval(
            {model.xs: np.array(x_train).reshape(-1, config['time_step'], config['num_feature']), model.ys: np.array(y_train).reshape(-1, 1)})
        f1 = f1_score(np.array(y_train), train_pred, average='macro')
        y_train = y_train
        train_pred = train_pred.astype(int).ravel()
        accuracy = accuracy_score(y_train, train_pred)
        recall = recall_score(y_true=y_train, y_pred=train_pred)
        precision = precision_score(y_true=y_train, y_pred=train_pred)
        logger.info("train F1 Score: {}".format(f1))
        logger.info("train Accuracy Score: {}".format(accuracy))
        logger.info("train Recall: {}".format(recall))
        logger.info("train Precision: {}".format(precision))
        ###
        precision, recall, thresholds = precision_recall_curve(
            y_test, y_pred)
        area = auc(recall, precision)
        logger.info("Area Under PR Curve(AP): {}".format(area))
        class_names = ["ham", "spam"]
        y_test = y_test
        y_pred = y_pred.astype(int).ravel()
        cm = confusion_matrix(y_test, y_pred)
        logger.info("confusion_matrix: {}".format(cm))
        plot_confusion_matrix(cm, target_names=class_names, normalize=False,
                              title='Confusion matrix, without normalization')
        plot_confusion_matrix(cm, target_names=class_names,
                              title='Normalized confusion matrix')


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.show()


def data_process(data, train_percentage=0.7):
    texts = []
    labels = []
    print(data['Category'])
    for i, label in enumerate(data['Category']):
        texts.append(data['Message'][i])
        if label == 'ham':
            labels.append(0)
        else:
            labels.append(1)
    texts = np.asarray(texts)
    labels = np.asarray(labels)
    logger.info("number of texts : {}".format(len(texts)))
    logger.info("number of labels : {}".format(len(labels)))
    # number of words used as features
    max_features = 10000
    # cut off the words after seeing 500 words in each document(email)
    maxlen = 500
    # we will use train_percentage of data as training, data_size-train_percentage as validation data
    training_samples = int(len(texts) * train_percentage)
    validation_samples = int(len(texts) - training_samples)
    logger.info(
        "The number of training {0}, validation {1} ".format(training_samples, validation_samples))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    logger.info("Found {} unique words: ".format(len(word_index)))
    data = pad_sequences(sequences, maxlen=maxlen)
    logger.info("data shape: {}".format(data.shape))
    np.random.seed(0)
    # shuffle data
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    X_train = data[:training_samples]
    Y_train = labels[:training_samples]
    X_test = data[training_samples:]
    Y_test = labels[training_samples:]
    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
    config = read_config('./config/config.json')
    logger.info('config : %s', config)
    orign_data = read_data(
        './data/SPAM text message 20170820 - Data.csv', skip_row=None)
    logger.info('data.head(5) : %s', orign_data.head(5))
    X_train, X_test, Y_train, Y_test = data_process(
        orign_data, train_percentage=0.8)
    train(config['build_model'], X_train, X_test, Y_train, Y_test)
