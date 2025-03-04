#!/usr/bin/env python3
"""
AI Trading System - Main Entry Point

This script serves as the entry point for the AI Trading System, integrating
all components: data collection, technical analysis, RAG-based context analysis,
trading strategy execution, and visualization.
"""

import argparse
import logging
import json
import time
from datetime import datetime
from pathlib import Path

# Import local modules
from data.collector import MarketDataCollector
from data.technical_analysis import TechnicalAnalysisProcessor
from rag.market_context import MarketContext
from rag.signal_enricher import SignalEnricher
from risk.position_sizing import PositionSizer
from strategy.trading_strategy import StrategyManager, CrossoverStrategy, RsiStrategy
from trading_execution import TradingExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")

def setup_directories():
    """Create necessary directories for the system."""
    directories = [
        Path("logs"),
        Path("data/signals"),
        Path("data/trades"),
        Path("data/orders"),
        Path("data/context_db"),
        Path("data/vector_db"),
        Path("data/processed"),
        Path("data/visualizations")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.info("Created data storage directories")

def run_data_collection_test(config):
    """
    Test data collection functionality.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting AI Trading System - Data Collection Test")
    
    # Create a data collector instance
    collector = MarketDataCollector(
        exchange_id=config.get("exchange", {}).get("id", "binance"),
        testnet=config.get("exchange", {}).get("testnet", True)
    )
    
    # Get available trading symbols
    base_currency = config.get("trading", {}).get("base_currency", "USDT")
    symbols = collector.get_available_symbols(quote_currency=base_currency)
    logger.info(f"Found {len(symbols)} trading pairs with {base_currency}")
    
    if symbols:
        # Use first available symbol for testing
        symbol = symbols[0]
        logger.info(f"Testing with symbol: {symbol}")
        
        # Fetch data for multiple timeframes
        timeframes = config.get("rag", {}).get("timeframes", ["15m", "1h", "4h"])
        logger.info(f"Fetching data for timeframes: {timeframes}")
        
        for tf in timeframes:
            # Fetch OHLCV data
            df = collector.fetch_ohlcv(symbol, timeframe=tf, limit=5)
            logger.info(f"\n{tf} OHLCV Data for {symbol}:")
            logger.info(f"\n{df}")
    
    logger.info("Data collection test completed")

def run_technical_analysis_test(config):
    """
    Test technical analysis functionality.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting AI Trading System - Technical Analysis Test")
    
    # Create data collector and technical analysis processor
    collector = MarketDataCollector(
        exchange_id=config.get("exchange", {}).get("id", "binance"),
        testnet=config.get("exchange", {}).get("testnet", True)
    )
    
    ta_processor = TechnicalAnalysisProcessor()
    
    # Get a symbol to analyze
    symbol = config.get("trading", {}).get("symbols", [])[0] if config.get("trading", {}).get("symbols") else "BTC/USDT"
    
    # Fetch hourly data
    df = collector.fetch_ohlcv(symbol, timeframe="1h", limit=100)
    
    if not df.empty:
        logger.info(f"Fetched {len(df)} hourly candles for {symbol}")
        
        # Process technical indicators
        processed_df = ta_processor.process_data(df)
        
        # Show the most recent data with indicators
        logger.info(f"Recent data with indicators for {symbol}:")
        logger.info(f"\n{processed_df.tail().to_string()}")
        
        # Get summary for RAG system
        summary = ta_processor.prepare_rag_data(processed_df)
        logger.info(f"Technical analysis summary for RAG:\n{json.dumps(summary, indent=2)}")
    else:
        logger.warning(f"Could not fetch data for {symbol}")
    
    logger.info("Technical analysis test completed")

def run_market_context_test(config):
    """
    Test market context gathering and storage.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting AI Trading System - Market Context Test")
    
    # Create market context analyzer
    market_context = MarketContext(config_path="config.json")
    
    # Get a symbol to analyze
    symbol = config.get("trading", {}).get("symbols", [])[0] if config.get("trading", {}).get("symbols") else "BTC/USDT"
    
    # Gather market context
    context = market_context.gather_market_context(symbol)
    
    if context:
        logger.info(f"Gathered market context for {symbol}")
        
        # Display top-level information
        logger.info(f"Timeframes: {list(context.get('timeframes', {}).keys())}")
        logger.info(f"Patterns detected: {[p.get('pattern') for p in context.get('patterns', [])]}")
        logger.info(f"Short-term trend: {context.get('aggregated_signals', {}).get('short_term_trend')}")
        logger.info(f"Medium-term trend: {context.get('aggregated_signals', {}).get('medium_term_trend')}")
        logger.info(f"Volatility: {context.get('aggregated_signals', {}).get('volatility')}")
        logger.info(f"Momentum: {context.get('aggregated_signals', {}).get('momentum')}")
        
        # Find similar patterns
        if context.get("similar_contexts"):
            logger.info(f"Found {len(context.get('similar_contexts'))} similar historical contexts")
            for i, similar in enumerate(context.get("similar_contexts")[:2]):
                logger.info(f"Similar context {i+1}: Similarity score = {similar.get('similarity_score', 0):.3f}")
                logger.info(f"  Timestamp: {similar.get('timestamp')}")
                logger.info(f"  Patterns: {[p.get('pattern') for p in similar.get('patterns', [])]}")
    else:
        logger.warning(f"Could not gather market context for {symbol}")
    
    logger.info("Market context test completed")

def run_signal_enrichment_test(config):
    """
    Test signal enrichment with RAG.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting AI Trading System - Signal Enrichment Test")
    
    # Create components
    market_context = MarketContext(config_path="config.json")
    signal_enricher = SignalEnricher(config_path="config.json")
    
    # Initialize strategy manager
    strategy_manager = StrategyManager()
    strategy_manager.add_strategy(CrossoverStrategy(fast_period=20, slow_period=50))
    strategy_manager.add_strategy(RsiStrategy(period=14))
    
    # Create data collector and technical analysis processor
    collector = MarketDataCollector(
        exchange_id=config.get("exchange", {}).get("id", "binance"),
        testnet=config.get("exchange", {}).get("testnet", True)
    )
    
    ta_processor = TechnicalAnalysisProcessor()
    
    # Get a symbol to analyze
    symbol = config.get("trading", {}).get("symbols", [])[0] if config.get("trading", {}).get("symbols") else "BTC/USDT"
    
    # Fetch hourly data
    df = collector.fetch_ohlcv(symbol, timeframe="1h", limit=100)
    
    if not df.empty:
        # Process technical indicators
        processed_df = ta_processor.process_data(df)
        
        # Generate signals
        signals = strategy_manager.generate_signals(processed_df)
        
        if signals:
            logger.info(f"Generated {len(signals)} signals for {symbol}")
            
            # Enrich the first signal
            test_signal = signals[0]
            test_signal["symbol"] = symbol
            test_signal["timeframe"] = "1h"
            test_signal["timestamp"] = datetime.now().isoformat()
            
            logger.info(f"Original signal: {test_signal}")
            
            # Gather market context
            context = market_context.gather_market_context(symbol)
            
            # Enrich signal
            enriched_signal = signal_enricher.enrich_signal(test_signal, context)
            
            logger.info(f"Enriched signal confidence: {enriched_signal.get('confidence', 0):.3f} (original: {test_signal.get('original_confidence', 0):.3f})")
            
            # Display confidence factors
            if "confidence_factors" in enriched_signal:
                logger.info(f"Confidence factors: {json.dumps(enriched_signal.get('confidence_factors', {}), indent=2)}")
            
            # Display Gemini analysis if available
            if "analysis" in enriched_signal:
                logger.info(f"Signal analysis:\n{enriched_signal.get('analysis')}")
        else:
            logger.warning(f"No signals generated for {symbol}")
    else:
        logger.warning(f"Could not fetch data for {symbol}")
    
    logger.info("Signal enrichment test completed")

def run_trading_execution_test(config):
    """
    Test trading execution functionality.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting AI Trading System - Trading Execution Test")
    
    # Create trading executor
    executor = TradingExecutor(config_path="config.json")
    
    # Load previous state
    executor.load_state()
    
    # Display current positions
    if executor.open_positions:
        logger.info(f"Current open positions: {list(executor.open_positions.keys())}")
    else:
        logger.info("No open positions")
    
    # Run a trading cycle
    try:
        logger.info("Running trading cycle")
        result = executor.run_trading_cycle()
        
        logger.info(f"Trading cycle completed at {result['timestamp']}")
        logger.info(f"Positions updated: {result['positions_updated']}")
        logger.info(f"Stop losses triggered: {result['stop_losses_triggered']}")
        logger.info(f"New analyses: {len(result['new_analyses'])}")
        logger.info(f"New trades: {len(result['new_trades'])}")
        
        for trade in result['new_trades']:
            logger.info(f"Trade: {trade['symbol']} {trade['direction']} with confidence {trade.get('confidence', 0):.3f} - Status: {trade['status']}")
    
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}")
    
    logger.info("Trading execution test completed")

def run_continuous_trading(config, interval_minutes=15):
    """
    Run the trading system continuously.
    
    Args:
        config: Configuration dictionary
        interval_minutes: Trading cycle interval in minutes
    """
    logger.info(f"Starting AI Trading System - Continuous Trading (Interval: {interval_minutes} minutes)")
    
    # Create trading executor
    executor = TradingExecutor(config_path="config.json")
    
    # Load previous state
    executor.load_state()
    
    # Display initial positions
    if executor.open_positions:
        logger.info(f"Initial open positions: {list(executor.open_positions.keys())}")
    else:
        logger.info("No initial open positions")
    
    try:
        cycle_count = 0
        while True:
            cycle_count += 1
            start_time = time.time()
            
            logger.info(f"Starting trading cycle {cycle_count}")
            
            # Run a trading cycle
            result = executor.run_trading_cycle()
            
            # Log results
            logger.info(f"Cycle {cycle_count} completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"Open positions: {result['positions_updated']}")
            logger.info(f"New trades: {len(result['new_trades'])}")
            
            # Save results to a timestamped file
            cycle_time = datetime.now()
            result_file = Path(f"data/cycles/cycle_{cycle_time.strftime('%Y%m%d_%H%M%S')}.json")
            result_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Wait for the next cycle
            next_cycle_time = start_time + (interval_minutes * 60)
            sleep_time = max(0, next_cycle_time - time.time())
            
            if sleep_time > 0:
                logger.info(f"Waiting {sleep_time:.2f} seconds for next cycle")
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("Continuous trading stopped by user")
    except Exception as e:
        logger.error(f"Error in continuous trading: {e}")
        import traceback
        traceback.print_exc()

def load_config(config_path="config.json"):
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def main():
    """Main entry point for the AI Trading System."""
    parser = argparse.ArgumentParser(description="AI Trading System")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=[
        "data", "technical", "context", "signal", "execution", "continuous"
    ], default="data", help="Test mode")
    parser.add_argument("--interval", type=int, default=15, help="Trading cycle interval in minutes (for continuous mode)")
    
    args = parser.parse_args()
    
    # Create necessary directories
    setup_directories()
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute based on mode
    if args.mode == "data":
        run_data_collection_test(config)
    elif args.mode == "technical":
        run_technical_analysis_test(config)
    elif args.mode == "context":
        run_market_context_test(config)
    elif args.mode == "signal":
        run_signal_enrichment_test(config)
    elif args.mode == "execution":
        run_trading_execution_test(config)
    elif args.mode == "continuous":
        run_continuous_trading(config, args.interval)
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()