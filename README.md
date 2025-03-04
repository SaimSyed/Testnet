AI Trading System with RAG-Enhanced Pattern Recognition
A complete cryptocurrency trading system with Retrieval Augmented Generation (RAG) for pattern recognition, powered by Google's Gemini AI.

Overview
This system integrates advanced technical analysis with large language model capabilities to identify trading opportunities across multiple timeframes. It uses historical pattern matching through a RAG system to enhance trading signals and make more informed decisions.

Key Features
Multi-timeframe data collection from Binance
Technical analysis with traditional indicators (RSI, MACD, Bollinger Bands, etc.)
RAG-based pattern recognition for historical context
Risk management with proper position sizing
Paper trading execution on Binance testnet
Continuous feedback loop for learning from trade outcomes
Visualization tools for signals and performance
System Architecture
The system is organized into the following modules:

Data Collection & Processing (data/)

collector.py: Fetches market data from exchanges
technical_analysis.py: Calculates technical indicators
RAG System (rag/)

market_context.py: Gathers and formats multi-timeframe market data
signal_enricher.py: Enhances signals using Gemini AI and similar pattern retrieval
Risk Management (risk/)

position_sizing.py: Implements risk-based position sizing
Trading Strategy (strategy/)

trading_strategy.py: Implements various technical trading strategies
Execution System

trading_execution.py: Handles trade execution with Binance API
main.py: Main entry point for the system
Visualization

visualize_indicators.py: Visualizes technical indicators
visualize_results.py: Visualizes trading results and signals
Setup
Prerequisites
Python 3.8+
Binance account (for paper trading)
Google AI API key (for Gemini)
Installation
Clone the repository

git clone https://github.com/yourusername/ai-trading-system.git
cd ai-trading-system
Install dependencies

pip install -r requirements.txt
Configure the system

Edit config.json with your API keys and settings
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
Running the System
The system can be run in various modes:

Data Collection Test

python main.py --mode data
Technical Analysis Test

python main.py --mode technical
Market Context Test

python main.py --mode context
Signal Enrichment Test

python main.py --mode signal
Trading Execution Test

python main.py --mode execution
Continuous Trading

python main.py --mode continuous --interval 15
Paper Trading Execution
The system connects to Binance testnet for paper trading:

python trading_execution.py --mode cycle
To view current positions:

python trading_execution.py --mode positions
To close a specific position:

python trading_execution.py --close BTC/USDT
Adding Custom Strategies
Create a new strategy by extending the base Strategy class in strategy/trading_strategy.py:

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
Then add your strategy to the StrategyManager:

strategy_manager = StrategyManager()
strategy_manager.add_strategy(MyCustomStrategy(param1=value1, param2=value2))
System Workflow
Data Collection: Fetch market data across multiple timeframes
Technical Analysis: Calculate indicators and generate base signals
Market Context: Create context snapshots with multi-timeframe data
RAG Enhancement: Find similar historical patterns and adjust confidence
Position Sizing: Calculate appropriate position size based on risk parameters
Trade Execution: Execute trades on the exchange (or paper trading)
Feedback Loop: Update RAG database with trade outcomes
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
CCXT for exchange connectivity
Google Gemini AI for pattern analysis
pandas-ta for technical indicators inspiration# Testnet