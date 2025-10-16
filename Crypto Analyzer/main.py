import requests
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Fetch current prices from CoinGecko API
def fetch_price():
    response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd')
    return response.json()

# Fetch historical data for a given cryptocurrency from CoinGecko API
def fetch_historical_data(coin):
    response = requests.get(f'https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=30')
    return [price[1] for price in response.json()['prices']]

# Calculate moving average
def moving_average(prices, window):
    return pd.Series(prices).rolling(window=window).mean().iloc[-1]

# Calculate MACD
def calculate_macd(prices):
    short_ema = pd.Series(prices).ewm(span=12, adjust=False).mean()
    long_ema = pd.Series(prices).ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd.iloc[-1], signal_line.iloc[-1], macd, signal_line

# Calculate Relative Strength Index (RSI)
def calculate_rsi(prices, period=14):
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# Calculate Bollinger Bands
def bollinger_bands(prices, window=20, num_std_dev=2):
    sma = moving_average(prices, window)
    rolling_std = pd.Series(prices).rolling(window).std()
    upper_band = sma + (rolling_std * num_std_dev)
    lower_band = sma - (rolling_std * num_std_dev)
    return upper_band.iloc[-1], lower_band.iloc[-1], sma

# Perform linear regression for price prediction
def predict_price(prices):
    X = np.array(range(len(prices))).reshape(-1, 1)  # Time as feature
    y = np.array(prices).reshape(-1, 1)  # Prices as target
    model = LinearRegression()
    model.fit(X, y)
    
    next_time = np.array([[len(prices)]])  # Next time point
    predicted_price = model.predict(next_time)[0][0]
    return predicted_price

# Plotting the chart with indicators
def plot_chart(prices, coin, entry_point=None, take_profit=None, stop_loss=None):
    plt.figure(figsize=(14, 7))
    plt.plot(prices, label=f'{coin} Price', color='blue')
    
    # Calculate indicators
    short_ma = moving_average(prices, window=5)
    long_ma = moving_average(prices, window=20)
    upper_band, lower_band, sma = bollinger_bands(prices)
    
    plt.plot(sma, label='SMA (20)', color='orange')
    plt.fill_between(range(len(prices)), lower_band, upper_band, color='lightgray', label='Bollinger Bands')
    
    if entry_point:
        plt.scatter(len(prices) - 1, entry_point, color='green', label='Entry Point', marker='^', s=100)
    if take_profit:
        plt.scatter(len(prices) - 1, take_profit, color='gold', label='Take Profit', marker='o', s=100)
    if stop_loss:
        plt.scatter(len(prices) - 1, stop_loss, color='red', label='Stop Loss', marker='x', s=100)
    
    plt.title(f'{coin} Price Chart with Indicators')
    plt.xlabel('Days')
    plt.ylabel('Price in USD')
    plt.legend()
    plt.show()

# Generate trade signals and suggestions
def generate_trade_signals(prices, coin):
    current_price = prices[-1]
    
    short_ma = moving_average(prices, window=5)
    long_ma = moving_average(prices, window=20)
    rsi = calculate_rsi(prices)
    macd, signal_line, macd_values, signal_values = calculate_macd(prices)
    upper_band, lower_band, sma = bollinger_bands(prices)

    predicted_price = predict_price(prices)

    entry_point = None
    exit_point = None
    stop_loss = None
    take_profit = None
    expected_profit = None
    buy_suggestion = None

    # Buy signal conditions
    if current_price < lower_band and rsi < 30:
        entry_point = current_price
        stop_loss = entry_point * 0.95
        take_profit = entry_point * 1.10
    elif short_ma > long_ma and macd > signal_line:
        entry_point = current_price
        stop_loss = entry_point * 0.95
        take_profit = entry_point * 1.10
    elif current_price > long_ma and rsi < 50:
        entry_point = current_price
        stop_loss = entry_point * 0.95
        take_profit = entry_point * 1.10

    # Calculate expected profit if entry point is set
    if entry_point and take_profit:
        expected_profit = take_profit - entry_point

    # Suggestion when entry point is stable
    if entry_point is None:
        if current_price > long_ma:
            buy_suggestion = f"Consider buying {coin} on a pullback to the recent support level."

    # Sell signal conditions
    if current_price > upper_band and rsi > 70:
        exit_point = current_price

    # Messages to display
    messages = []
    if entry_point:
        messages.append(f"{coin.title()} - Entry Point: ${entry_point:.2f}, Stop Loss: ${stop_loss:.2f}, Take Profit: ${take_profit:.2f}")
        if expected_profit is not None:
            messages.append(f"{coin.title()} - Expected Profit: ${expected_profit:.2f}")
    if exit_point:
        messages.append(f"{coin.title()} - Exit Point: ${exit_point:.2f}")
    messages.append(f"{coin.title()} - Predicted Price: ${predicted_price:.2f}")
    if buy_suggestion:
        messages.append(buy_suggestion)

    if not messages:
        messages.append(f"{coin.title()} price trend is stable.")

    # Plot the chart with indicators
    plot_chart(prices, coin, entry_point, take_profit, stop_loss)

    return messages

# Main function to run the application
def main():
    bitcoin_historical_prices = fetch_historical_data('bitcoin')
    ethereum_historical_prices = fetch_historical_data('ethereum')

    while True:
        prices = fetch_price()
        bitcoin_price = prices['bitcoin']['usd']
        ethereum_price = prices['ethereum']['usd']

        bitcoin_historical_prices.append(bitcoin_price)
        ethereum_historical_prices.append(ethereum_price)
        
        if len(bitcoin_historical_prices) > 30:
            bitcoin_historical_prices.pop(0)
        if len(ethereum_historical_prices) > 30:
            ethereum_historical_prices.pop(0)

        bitcoin_signals = generate_trade_signals(bitcoin_historical_prices, 'Bitcoin')
        ethereum_signals = generate_trade_signals(ethereum_historical_prices, 'Ethereum')

        print(f'Bitcoin: ${bitcoin_price}, Signals: {", ".join(bitcoin_signals)}')
        print(f'Ethereum: ${ethereum_price}, Signals: {", ".join(ethereum_signals)}')

        time.sleep(1800)  # Check every 30 minutes (1800 seconds)

if __name__ == "__main__":
    main()