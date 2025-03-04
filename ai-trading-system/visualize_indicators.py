#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data.technical_analysis import TechnicalAnalysisProcessor

# Create sample price data with a trend
dates = pd.date_range(start='2023-01-01', periods=100, freq='1D')
# Create a price series with an uptrend and then a downtrend
price = np.concatenate([
    np.linspace(100, 150, 50),  # Uptrend
    np.linspace(150, 120, 50)   # Downtrend
])
# Add some noise
price = price + np.random.normal(0, 3, 100)

data = pd.DataFrame({
    'open': price - np.random.normal(1, 0.5, 100),
    'high': price + np.random.normal(2, 1, 100),
    'low': price - np.random.normal(2, 1, 100),
    'close': price,
    'volume': np.random.normal(1000000, 200000, 100) + price * 1000
}, index=dates)

# Make sure high is always >= open, close, low
data['high'] = data[['open', 'close', 'high']].max(axis=1)
# Make sure low is always <= open, close, high
data['low'] = data[['open', 'close', 'low']].min(axis=1)

# Initialize the processor
processor = TechnicalAnalysisProcessor()

# Process the data
result = processor.process_data(data)

# Create plots
plt.figure(figsize=(12, 16))

# Plot 1: Price and Moving Averages
plt.subplot(4, 1, 1)
plt.plot(result.index, result['close'], label='Price')
plt.plot(result.index, result['sma_20'], label='SMA 20')
plt.plot(result.index, result['sma_50'], label='SMA 50')
plt.plot(result.index, result['sma_200'], label='SMA 200')
plt.title('Price and Moving Averages')
plt.legend()
plt.grid(True)

# Plot 2: RSI
plt.subplot(4, 1, 2)
plt.plot(result.index, result['rsi'])
plt.axhline(y=70, color='r', linestyle='--')  # Overbought line
plt.axhline(y=30, color='g', linestyle='--')  # Oversold line
plt.title('Relative Strength Index (RSI)')
plt.grid(True)
plt.ylim(0, 100)

# Plot 3: MACD
plt.subplot(4, 1, 3)
plt.plot(result.index, result['macd'], label='MACD')
plt.plot(result.index, result['macd_signal'], label='Signal Line')
plt.bar(result.index, result['macd_hist'], alpha=0.5, label='Histogram')
plt.title('MACD')
plt.legend()
plt.grid(True)

# Plot 4: Bollinger Bands
plt.subplot(4, 1, 4)
plt.plot(result.index, result['close'], label='Price')
plt.plot(result.index, result['bb_upper'], label='Upper Band')
plt.plot(result.index, result['bb_middle'], label='Middle Band')
plt.plot(result.index, result['bb_lower'], label='Lower Band')
plt.title('Bollinger Bands')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('technical_indicators.png')
plt.close()

print("Visualization completed! Check 'technical_indicators.png' for the charts.")
