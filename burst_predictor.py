import hist as hist
import np as np
from src.lstm_utils import *

def build_training():
    np.random.seed(42)
    window_len = 100
    test_size = 0.2
    zero_base = True
    lstm_neurons = 50
    epochs = 10
    batch_size = 16
    loss = 'mse'
    dropout = 0.2
    optimizer = 'adam'
    train, test, X_train, X_test, y_train, y_test = \
        prepare_data(hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

    model = build_lstm_model(
        X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1,
        shuffle=True)
    return model, history

def plot_history(history):
    plt.plot(history.history['loss'], 'r', linewidth=2, label='Train loss')
    plt.plot(history.history['val_loss'], 'g', linewidth=2, label='Validation loss')
    plt.title('LSTM')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.show()

def model_stat(hist, target_col, model):
    train, test, X_train, X_test, y_train, y_test = \
        prepare_data(hist, target_col)
    targets = test[target_col][50:]
    preds = model.predict(X_test).squeeze()
    mean_absolute_error(preds, y_test)
    return targets, preds

if __name__ == '__main__':
    model, his = build_training()
    plot_history(his)
