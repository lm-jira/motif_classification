import argparse
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    MaxPool1D,
    Bidirectional,
    LSTM,
    Flatten,
    Dense,
)

from utils import common

BATCH_SIZE = 1000


def train():
    train_data, train_label = common.load_mat("data", mode="train")
    # valid_data, valid_label = common.load_mat("data", mode="valid")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logdir')

    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop")

    model.fit(train_data, train_label, batch_size=BATCH_SIZE,
              epochs=60, shuffle=True,
              callbacks=[tensorboard_callback])
    model.save(args.model_path)


def test():
    test_data, test_label = common.load_mat("data", mode="test")
    model.load_weights(args.model_path)

    results = model.evaluate(test_data,
                             test_label,
                             show_accuracy=True)
    print(results)


def predict():
    pass


if __name__ == "__main__":
    model = tf.keras.Sequential()
    model.add(Conv1D(input_shape=(BATCH_SIZE, 4),
                     filters=320,
                     kernel_size=26,
                     padding="valid",
                     activation="relu"))

    model.add(MaxPool1D(pool_size=13, strides=13))
    # model.add(Bidirectional(LSTM(units=10, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=30, return_sequences=True)))
    model.add(Flatten())
    # model.add(Dense(units=925, activation="relu"))
    model.add(Dense(units=1500, activation="relu"))
    model.add(Dense(units=919, activation="sigmoid"))

    model.summary()

    parser = argparse.ArgumentParser(
        description="Motif function classification")
    parser.add_argument("mode", choices=["train", "test", "predict"],
                        help="What do you want to do with model?")
    parser.add_argument("--model_path", default="./model.h5",
                        help="Name of saved model \
                        (output for train, input for test)")
    args = parser.parse_args()

    target_function = globals()[args.mode]
    target_function()
