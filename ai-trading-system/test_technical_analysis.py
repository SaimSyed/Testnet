#!/usr/bin/env python3

import pandas as pd
import numpy as np
from data.technical_analysis import TechnicalAnalysisProcessor

# Create sample price data
dates = pd.date_range(start='2023-01-01', periods=100, freq='1D')
data = pd.DataFrame({
    'open': np.random.normal(100, 5, 100),
    'high': np.random.normal(105, 5, 100),
    'low': np.random.normal(95, 5, 100),
    'close': np.random.normal(100, 5, 100),
    'volume': np.random.normal(1000000, 200000, 100)
}, index=dates)

# Make sure high is always >= open, close, low
data['high'] = data[['open', 'close', 'high']].max(axis=1)
# Make sure low is always <= open, close, high
data['low'] = data[['open', 'close', 'low']].min(axis=1)

# Initialize the processor
processor = TechnicalAnalysisProcessor()

# Process the data
result = processor.process_data(data)

# Display results
print(f"Original data shape: {data.shape}")
print(f"Processed data shape: {result.shape}")
print("\nCalculated indicators:")
for col in sorted(result.columns):
    if col not in ['open', 'high', 'low', 'close', 'volume']:
        print(f"- {col}")

# Show sample of the processed data
print("\nSample of processed data (last 3 rows):")
sample_cols = ['close', 'sma_20', 'rsi', 'macd', 'composite_signal']
print(result[sample_cols].tail(3))

# Test RAG data generation
rag_data = processor.prepare_rag_data(result)
print("\nRAG Data Sample:")
print(f"Price: ${rag_data['price']:.2f}")
print(f"RSI: {rag_data['indicators']['rsi']:.2f}")
print(f"Short-term trend: {rag_data['trends']['short_term']}")
