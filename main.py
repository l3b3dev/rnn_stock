from itertools import cycle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from TrainingPipeline import TrainingPipeline

sns.set_style("darkgrid")

if __name__ == '__main__':

    data_dir = "data"

    # LSTM configurations
    window = 180
    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 50

    pipeline = TrainingPipeline(input_dim, hidden_dim, output_dim, num_layers, num_epochs)
    shiller, spx = pipeline.load_data(data_dir)

    # graph them next to each other
    fig, ax1 = plt.subplots(figsize=(12.5, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('S&P500', color=color)
    plt.plot(spx["Date"], spx.iloc[:, 7], color, label="Index")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'
    ax2.set_ylabel('Shiller P/E', color=color)  # we already handled the x-label with ax1
    plt.plot(shiller["Date"], shiller["Value"], color, label="Shiller P/E")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # now normalize price
    price = pipeline.normalize_price(spx)
    # perform 20%-80% test train split on 180 sliding window
    x_train, y_train, x_test, y_test = pipeline.train_test_split(price, window)

    # train it
    y_train_pred, hist = pipeline.train(x_train, y_train)

    # plot the situation during training
    y_train_pred, y_train = pipeline.inverse_transform_pair(y_train_pred, y_train)
    predict, original = pd.DataFrame(y_train_pred), pd.DataFrame(y_train)

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 2, 1)
    ax = sns.lineplot(x=original.index, y=original[0], label="Data", color='royalblue')
    ax = sns.lineplot(x=predict.index, y=predict[0], label="Training Prediction (LSTM)", color='tomato')
    ax.set_title('Stock price', size=14, fontweight='bold')
    ax.set_xlabel("Days", size=14)
    ax.set_ylabel("Cost (USD)", size=14)

    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=hist, color='royalblue')
    ax.set_xlabel("Epoch", size=14)
    ax.set_ylabel("Loss", size=14)
    ax.set_title("Training Loss", size=14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)
    plt.show()

    # now predict on test data
    y_test_pred = pipeline.predict(x_test)
    y_test_pred, y_test = pipeline.inverse_transform_pair(y_test_pred, y_test)

    # get RMSE
    trainScore = pipeline.compute_RMSE(y_train, y_train_pred)
    testScore = pipeline.compute_RMSE(y_test, y_test_pred)

    print('Train Score: %.2f RMSE' % (trainScore))
    print('Test Score: %.2f RMSE' % (testScore))

    # now calculate accuracy as (predicted_price-actual_price)/actual_price
    trainAccuracy = pipeline.compute_accuracy(y_train_pred, y_train)
    testAccuracy = pipeline.compute_accuracy(y_test_pred, y_test)

    std_default = '0.0'
    # graph accuracy
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 2, 1)
    ax = sns.lineplot(x=trainAccuracy.index, y=trainAccuracy[std_default], label="Training prediction error", color='royalblue')
    ax.set_title('Training prediction error', size=14, fontweight='bold')
    ax.set_xlabel("Days", size=14)
    ax.set_ylabel("Error", size=14)

    plt.subplot(1, 2, 2)
    ax = sns.lineplot(x=testAccuracy.index, y=testAccuracy[std_default], label="Test prediction error", color='tomato')
    ax.set_xlabel("Days", size=14)
    ax.set_ylabel("Error", size=14)
    ax.set_title("Test prediction error", size=14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)
    plt.show()

    # collect predicted results
    result = pipeline.collect(window, price, y_train_pred, y_test_pred)

    plt.figure(figsize=(12.5, 6))
    ax = sns.lineplot(x=result.index, y=result[0], label="Train prediction", color='royalblue')
    ax = sns.lineplot(x=result.index, y=result[1], label="Test Prediction", color='tomato')
    ax = sns.lineplot(x=result.index, y=result[2], label="Actual", color='green')
    ax.set_title('Prediction results', size=14, fontweight='bold')
    ax.set_xlabel("Days", size=14)
    ax.set_ylabel("Cost (USD)", size=14)
    plt.show()

    # corrupt all test data with Gaussian noise
    sdevs = [0., 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    # per instructions in step 4, test set is all data from Jan 1, 1980
    filter_date = '1-1-1980'
    x_test, y_test, filtered_spx = pipeline.get_test_data(spx, window, filter_date)
    accuracy_df = pipeline.compute_noise_accuracy(sdevs, x_test, y_test, filtered_spx["Date"] )

    # plot accuracies on same graph
    colors = cycle(['blue', 'green', 'red', 'purple', 'brown', 'cyan', 'tomato', 'black', 'yellow','gold'])
    plt.figure(figsize=(12.5, 6))
    for col, c, sd in zip(accuracy_df, colors, sdevs):
        if col == "Date": continue
        ax = sns.lineplot(x=accuracy_df.index, y=accuracy_df[col], label=f"Train prediction error for sd: {sd}", color=c)
    ax.set_title('Prediction results', size=14, fontweight='bold')
    ax.set_xlabel("Days", size=14)
    ax.set_ylabel("Error", size=14)
    plt.show()

    #write the accuracy table as CSV for step 6b
    accuracy_df.to_csv(f'{data_dir}/noise_accuracy.csv', index=False)
