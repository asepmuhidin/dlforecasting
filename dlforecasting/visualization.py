import matplotlib.pyplot as plt


def plot_loss(history):
    """
    Plot training loss history.
    """

    plt.figure(figsize=(10, 5))

    plt.plot(history)

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.show()


def plot_predictions(actual, predicted):
    """
    Plot actual vs predicted values.
    """

    plt.figure(figsize=(12, 5))

    plt.plot(actual, label='Actual')

    plt.plot(predicted, label='Prediction')

    plt.title("Forecasting Result")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.grid(True)

    plt.show()


def plot_future_forecast(
    historical,
    future,
    figsize=(12,5)
):
    """
    Plot future forecasting result.
    """

    plt.figure(figsize)

    plt.plot(
        range(len(historical)),
        historical,
        label='Historical'
    )

    plt.plot(
        range(
            len(historical),
            len(historical) + len(future)
        ),
        future,
        label='Future Forecast'
    )

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.grid(True)

    plt.show()