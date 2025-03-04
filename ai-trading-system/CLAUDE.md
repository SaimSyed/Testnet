# AI Trading System Guidelines

## Commands
- Run tests: `python -m unittest discover`
- Run specific test: `python -m unittest test_file.TestClass.test_method`
- Run strategy test: `python test_strategy.py`
- Run technical analysis test: `python test_technical_analysis.py`
- Visualize results: `python visualize_results.py`
- Analyze market: `python analyze_market.py`

## Code Style
- **Imports**: System → third-party → local modules
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Typing**: Use type hints from typing module (Dict, List, Optional)
- **Docstrings**: Google-style format with Args/Returns sections
- **Error handling**: Use try/except blocks for API calls and data processing
- **Modules**: Organize code in logical modules (data, strategy, risk, rag)
- **Testing**: Write unittest-based tests with descriptive method names
- **Comments**: Add comments for complex algorithms and trading logic

## Project Structure
- `/data`: Data collection and technical analysis
- `/strategy`: Trading strategies implementation
- `/risk`: Position sizing and risk management
- `/rag`: Market context and signal enrichment