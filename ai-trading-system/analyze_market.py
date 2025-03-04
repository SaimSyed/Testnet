#!/usr/bin/env python3
"""
Main analysis script for the AI trading system.

This script integrates all components of the AI trading system to:
1. Collect market data
2. Perform technical analysis
3. Generate trading signals
4. Enrich signals with market context
5. Calculate position sizes based on risk parameters
"""

import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import system components
from data.collector import MarketDataCollector
from data.technical_analysis import TechnicalAnalysisProcessor
from strategy.trading_strategy import TradingStrategy
from rag.signal_enricher import SignalEnricher
from risk.position_sizing import PositionSizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('analyze_market')

class TradingSystemAnalyzer:
    """
    Main class for the AI trading system that integrates all components.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the TradingSystemAnalyzer with configuration parameters.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        logger.info("Initializing trading system components...")
        
        # Initialize components
        self.data_collector = MarketDataCollector(config_path)
        self.technical_analyzer = TechnicalAnalysisProcessor(config_path)
        self.trading_strategy = TradingStrategy(config_path)
        self.signal_enricher = SignalEnricher(config_path)
        self.position_sizer = PositionSizer(config_path)
        
        # Get symbols and timeframes from config
        self.symbols = self.config.get("symbols", ["BTC/USDT"])
        self.timeframes = self.config.get("timeframes", ["1d"])
        
        logger.info(f"Trading system initialized with {len(self.symbols)} symbols and {len(self.timeframes)} timeframes")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing configuration parameters
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading configuration: {e}")
            raise Exception(f"Error loading configuration: {e}")
    
    def run_analysis(self, symbols: Optional[List[str]] = None, 
                   timeframes: Optional[List[str]] = None,
                   save_results: bool = True) -> Dict:
        """
        Run the complete analysis pipeline.
        
        Args:
            symbols: List of trading symbols to analyze (overrides config)
            timeframes: List of timeframes to analyze (overrides config)
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary containing analysis results
        """
        symbols = symbols or self.symbols
        timeframes = timeframes or self.timeframes
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "timeframes": timeframes,
            "signals": [],
            "positions": []
        }
        
        logger.info(f"Starting analysis for {len(symbols)} symbols and {len(timeframes)} timeframes")
        
        # Process each symbol and timeframe
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"Analyzing {symbol} on {timeframe} timeframe")
                
                try:
                    # 1. Collect market data
                    market_data = self.data_collector.fetch_data(symbol, timeframe)
                    
                    # 2. Perform technical analysis
                    indicators = self.technical_analyzer.calculate_indicators(market_data)
                    
                    # 3. Generate trading signals
                    signals = self.trading_strategy.generate_signals(market_data, indicators, symbol, timeframe)
                    
                    if signals:
                        # 4. Enrich signals with market context
                        enriched_signals = self.signal_enricher.enrich_signals(signals)
                        
                        # 5. Calculate position sizes
                        positions = self.position_sizer.calculate_portfolio_allocation(enriched_signals)
                        
                        # Add to results
                        results["signals"].extend(enriched_signals)
                        results["positions"].extend(positions)
                        
                        # Log signals and positions
                        for signal, position in zip(enriched_signals, positions):
                            logger.info(f"Signal: {signal['direction']} {symbol} ({signal['signal_type']}) - "
                                      f"Confidence: {signal['confidence']:.2f}")
                            logger.info(f"Position: Size: {position['position_size']:.6f}, "
                                      f"Value: ${position['position_value']:.2f}")
                    else:
                        logger.info(f"No signals generated for {symbol} on {timeframe} timeframe")
                        
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} on {timeframe} timeframe: {e}")
        
        # Save results if requested
        if save_results and results["signals"]:
            self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict) -> None:
        """
        Save analysis results to disk.
        
        Args:
            results: Dictionary containing analysis results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"analysis_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Analysis results saved to {output_file}")

def main():
    """Main function to run the trading system analysis."""
    parser = argparse.ArgumentParser(description="AI Trading System Analysis")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols to analyze (overrides config)")
    parser.add_argument("--timeframes", type=str, nargs="+", help="Timeframes to analyze (overrides config)")
    parser.add_argument("--no-save", action="store_true", help="Don't save analysis results")
    
    args = parser.parse_args()
    
    try:
        # Initialize and run the trading system
        analyzer = TradingSystemAnalyzer(args.config)
        results = analyzer.run_analysis(
            symbols=args.symbols,
            timeframes=args.timeframes,
            save_results=not args.no_save
        )
        
        # Print summary
        print("\n===== Analysis Summary =====")
        print(f"Analyzed {len(results['symbols'])} symbols across {len(results['timeframes'])} timeframes")
        print(f"Generated {len(results['signals'])} signals")
        print(f"Calculated {len(results['positions'])} positions")
        
        # Print top signals by confidence
        if results["signals"]:
            top_signals = sorted(results["signals"], key=lambda x: x["confidence"], reverse=True)[:3]
            print("\nTop Signals by Confidence:")
            for i, signal in enumerate(top_signals, 1):
                print(f"{i}. {signal['direction'].upper()} {signal['symbol']} - "
                    f"Confidence: {signal['confidence']:.2f} - "
                    f"Type: {signal['signal_type']}")
        
    except Exception as e:
        logger.error(f"Error running trading system: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()