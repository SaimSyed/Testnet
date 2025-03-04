# AI Trading System with RAG-Enhanced Pattern Recognition

A complete cryptocurrency trading system with Retrieval Augmented Generation (RAG) for pattern recognition, powered by Google's Gemini AI.

## Overview

This system integrates advanced technical analysis with large language model capabilities to identify trading opportunities across multiple timeframes. It uses historical pattern matching through a RAG system to enhance trading signals and make more informed decisions.

### Key Features

- Multi-timeframe data collection from Binance
- Technical analysis with traditional indicators (RSI, MACD, Bollinger Bands, etc.)
- RAG-based pattern recognition for historical context
- Risk management with proper position sizing
- Paper trading execution on Binance testnet
- Continuous feedback loop for learning from trade outcomes
- Visualization tools for signals and performance

## System Architecture

The system is organized into the following modules:

1. **Data Collection & Processing** (`data/`)
   - `collector.py`: Fetches market data from exchanges
   - `technical_analysis.py`: Calculates technical indicators

2. **RAG System** (`rag/`)
   - `market_context.py`: Gathers and formats multi-timeframe market data
   - `signal_enricher.py`: Enhances signals using Gemini AI and similar pattern retrieval

3. **Risk Management** (`risk/`)
   - `position_sizing.py`: Implements risk-based position sizing

4. **Trading Strategy** (`strategy/`)
   - `trading_strategy.py`: Implements various technical trading strategies

5. **Execution System**
   - `trading_execution.py`: Handles trade execution with Binance API
   - `main.py`: Main entry point for the system

6. **Visualization**
   - `visualize_indicators.py`: Visualizes technical indicators
   - `visualize_results.py`: Visualizes trading results and signals

## Setup

### Prerequisites

- Python 3.8+
- Binance account (for paper trading)
- Google AI API key (for Gemini)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/ai-trading-system.git
   cd ai-trading-system
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the system
   - Edit `config.json` with your API keys and settings
   ```json
   {
     "exchange": {
       "id": "binance",
       "testnet": true,
       "api_key": "YOUR_BINANCE_API_KEY",
       "api_secret": "YOUR_BINANCE_API_SECRET"
     },
     "gemini": {
       "api_key": "YOUR_GEMINI_API_KEY"
     }
   }
   ```

### Running the System

The system can be run in various modes:

1. **Data Collection Test**
   ```bash
   python main.py --mode data
   ```

2. **Technical Analysis Test**
   ```bash
   python main.py --mode technical
   ```

3. **Market Context Test**
   ```bash
   python main.py --mode context
   ```

4. **Signal Enrichment Test**
   ```bash
   python main.py --mode signal
   ```

5. **Trading Execution Test**
   ```bash
   python main.py --mode execution
   ```

6. **Continuous Trading**
   ```bash
   python main.py --mode continuous --interval 15
   ```

### Paper Trading Execution

The system connects to Binance testnet for paper trading:

```bash
python trading_execution.py --mode cycle
```

To view current positions:
```bash
python trading_execution.py --mode positions
```

To close a specific position:
```bash
python trading_execution.py --close BTC/USDT
```

## Adding Custom Strategies

Create a new strategy by extending the base `Strategy` class in `strategy/trading_strategy.py`:

```python
class MyCustomStrategy(Strategy):
    def __init__(self, param1=default1, param2=default2, weight=1.0):
        super().__init__(weight)
        self.param1 = param1
        self.param2 = param2
    
    def generate_signal(self, data):
        # Your strategy logic here
        signal = self._create_signal_template()
        
        # Add signal logic and set direction, strength, confidence
        
        return signal
```

Then add your strategy to the `StrategyManager`:

```python
strategy_manager = StrategyManager()
strategy_manager.add_strategy(MyCustomStrategy(param1=value1, param2=value2))
```

## System Workflow

1. **Data Collection**: Fetch market data across multiple timeframes
2. **Technical Analysis**: Calculate indicators and generate base signals
3. **Market Context**: Create context snapshots with multi-timeframe data
4. **RAG Enhancement**: Find similar historical patterns and adjust confidence
5. **Position Sizing**: Calculate appropriate position size based on risk parameters
6. **Trade Execution**: Execute trades on the exchange (or paper trading)
7. **Feedback Loop**: Update RAG database with trade outcomes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [CCXT](https://github.com/ccxt/ccxt) for exchange connectivity
- [Google Gemini AI](https://ai.google.dev/) for pattern analysis
- [pandas-ta](https://github.com/twopirllc/pandas-ta) for technical indicators inspiration