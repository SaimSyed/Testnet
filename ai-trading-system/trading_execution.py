#!/usr/bin/env python3
"""
Trading Execution System for AI Trading System.

This module implements the execution layer for algorithmic trading strategies,
connecting with the Binance testnet API to execute paper trades based on signals.
"""

import json
import time
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import ccxt
import pandas as pd

# Import local modules
from data.collector import MarketDataCollector
from data.technical_analysis import TechnicalAnalysisProcessor
from rag.market_context import MarketContext
from rag.signal_enricher import SignalEnricher
from risk.position_sizing import PositionSizer
from strategy.trading_strategy import StrategyManager, CrossoverStrategy, RsiStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trading_execution')

class TradingExecutor:
    """
    Executes trading strategies on Binance testnet with proper risk management.
    
    This class is responsible for:
    1. Collecting real-time market data
    2. Processing signals from trading strategies
    3. Enriching signals with RAG-based context
    4. Determining position sizes using risk management
    5. Executing trades on the exchange
    6. Tracking and managing open positions
    7. Updating the feedback loop with trade outcomes
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the TradingExecutor.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.config_path = config_path
        
        # Initialize exchange connection
        self._initialize_exchange()
        
        # Initialize components
        self.data_collector = MarketDataCollector(
            exchange_id=self.config["exchange"]["id"],
            testnet=self.config["exchange"]["testnet"]
        )
        self.ta_processor = TechnicalAnalysisProcessor()
        self.market_context = MarketContext(config_path)
        self.signal_enricher = SignalEnricher(config_path)
        self.position_sizer = PositionSizer(config_path)
        
        # Initialize strategy manager and add strategies
        self.strategy_manager = StrategyManager()
        self._initialize_strategies()
        
        # Initialize internal state
        self.open_positions = {}
        self.order_history = []
        self.trade_history = []
        
        # Create logging and data directories
        self._initialize_directories()
        
        logger.info("Trading Executor initialized successfully")
    
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
                config = json.load(f)
                
                # Ensure necessary sections exist
                required_sections = ["exchange", "trading", "risk_management", "rag"]
                for section in required_sections:
                    if section not in config:
                        config[section] = {}
                
                return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _initialize_exchange(self) -> None:
        """Initialize connection to the exchange."""
        try:
            exchange_id = self.config["exchange"]["id"]
            exchange_class = getattr(ccxt, exchange_id)
            
            # Prepare exchange options
            options = {
                'enableRateLimit': True
            }
            
            # Add API credentials if available
            if "api_key" in self.config["exchange"] and "api_secret" in self.config["exchange"]:
                options['apiKey'] = self.config["exchange"]["api_key"]
                options['secret'] = self.config["exchange"]["api_secret"]
            
            # Determine market type (spot or futures)
            if self.config["exchange"].get("use_futures", False):
                options['options'] = {
                    'defaultType': 'future'
                }
            
            # Create exchange instance
            self.exchange = exchange_class(options)
            
            # Set testnet mode if applicable
            if self.config["exchange"]["testnet"] and hasattr(self.exchange, 'set_sandbox_mode'):
                self.exchange.set_sandbox_mode(True)
                logger.info(f"Connected to {exchange_id} testnet")
            else:
                logger.info(f"Connected to {exchange_id} live market")
            
            # Load markets
            self.exchange.load_markets()
            logger.info(f"Loaded {len(self.exchange.markets)} markets")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    def _initialize_strategies(self) -> None:
        """Initialize and add trading strategies to the manager."""
        # Add a simple moving average crossover strategy
        self.strategy_manager.add_strategy(
            CrossoverStrategy(
                fast_period=20,
                slow_period=50,
                weight=1.0
            )
        )
        
        # Add RSI strategy
        self.strategy_manager.add_strategy(
            RsiStrategy(
                period=14,
                overbought=70,
                oversold=30,
                weight=1.0
            )
        )
        
        logger.info(f"Initialized {len(self.strategy_manager.strategies)} trading strategies")
    
    def _initialize_directories(self) -> None:
        """Create necessary directories for logs and data storage."""
        directories = [
            Path("logs"),
            Path("data/signals"),
            Path("data/trades"),
            Path("data/orders"),
            Path("data/context_db"),
            Path("data/vector_db"),
            Path("data/processed")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("Created data storage directories")
    
    def get_tradable_symbols(self) -> List[str]:
        """
        Get list of tradable symbols based on configuration.
        
        Returns:
            List of symbol strings
        """
        # Use configured symbols if available
        if "symbols" in self.config["trading"] and self.config["trading"]["symbols"]:
            return self.config["trading"]["symbols"]
        
        # Otherwise, get symbols from exchange with the configured base currency
        base_currency = self.config["trading"].get("base_currency", "USDT")
        available_symbols = self.data_collector.get_available_symbols(quote_currency=base_currency)
        
        # Limit to top 5 by volume if no specific symbols configured
        if len(available_symbols) > 5:
            # Could implement volume-based filtering here
            return available_symbols[:5]
        
        return available_symbols
    
    def analyze_market(self, symbol: str) -> Dict:
        """
        Analyze market data and generate signals with context enrichment.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary containing market analysis and signals
        """
        logger.info(f"Analyzing market for {symbol}")
        
        # Collect market data across timeframes
        market_data = {}
        for timeframe in self.config["rag"]["timeframes"]:
            df = self.data_collector.fetch_ohlcv(
                symbol, 
                timeframe=timeframe, 
                limit=self.config["rag"]["lookback_periods"]
            )
            
            if not df.empty:
                # Process technical indicators
                df = self.ta_processor.process_data(df)
                market_data[timeframe] = df
                logger.info(f"Collected and processed {len(df)} {timeframe} candles for {symbol}")
            else:
                logger.warning(f"No data returned for {symbol} on {timeframe} timeframe")
        
        if not market_data:
            logger.error(f"Could not collect any market data for {symbol}")
            return {}
        
        # Generate signals from strategies
        signals = []
        for timeframe, df in market_data.items():
            if len(df) > 20:  # Ensure enough data for indicators
                # Generate signals from all strategies
                timeframe_signals = self.strategy_manager.generate_signals(df)
                
                # Add symbol and timeframe to signals
                for signal in timeframe_signals:
                    signal["symbol"] = symbol
                    signal["timeframe"] = timeframe
                    signal["timestamp"] = datetime.now().isoformat()
                    signals.append(signal)
        
        # Calculate composite signal
        composite_value = self.strategy_manager.get_composite_signal(signals)
        
        # Create market context
        market_context = self.market_context.create_market_snapshot(symbol, market_data)
        
        # Store market context for future reference
        self.market_context.store_context(market_context)
        
        # Enrich signals with RAG context
        enriched_signals = []
        for signal in signals:
            # Only enrich significant signals
            if signal.get("strength", 0) > 0.5 and signal.get("direction") in ["buy", "sell"]:
                enriched_signal = self.signal_enricher.enrich_signal(signal, market_context)
                enriched_signals.append(enriched_signal)
        
        # Determine if a trade signal is present
        trade_signal = None
        if enriched_signals:
            # Find the signal with highest confidence and strength
            best_signal = max(
                enriched_signals, 
                key=lambda s: s.get("confidence", 0) * s.get("strength", 0)
            )
            
            # Check if signal meets minimum confidence threshold
            if best_signal.get("confidence", 0) >= self.config["trading"].get("min_confidence_for_trade", 0.65):
                trade_signal = best_signal
        
        # Prepare analysis result
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "market_data": {
                tf: df.tail(5).to_dict('records') for tf, df in market_data.items()
            },
            "signals": signals,
            "enriched_signals": enriched_signals,
            "trade_signal": trade_signal,
            "composite_signal": composite_value,
            "market_context": {
                "short_term_trend": market_context.get("aggregated_signals", {}).get("short_term_trend"),
                "medium_term_trend": market_context.get("aggregated_signals", {}).get("medium_term_trend"),
                "volatility": market_context.get("aggregated_signals", {}).get("volatility"),
                "momentum": market_context.get("aggregated_signals", {}).get("momentum"),
                "patterns": [p.get("pattern") for p in market_context.get("patterns", [])]
            }
        }
        
        return analysis
    
    def calculate_position_size(self, 
                                symbol: str, 
                                signal: Dict, 
                                current_price: float) -> Dict:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal dictionary
            current_price: Current market price
            
        Returns:
            Dictionary with position sizing information
        """
        # Get ATR for volatility-based stop placement
        atr = None
        for tf, data in signal.get("market_data", {}).items():
            if tf == "1h" and isinstance(data, pd.DataFrame) and "atr" in data.columns:
                atr = data["atr"].iloc[-1]
        
        # Determine stop loss price
        stop_distance = None
        if atr is not None:
            stop_multiplier = self.config["risk_management"].get("stop_loss_atr_multiplier", 2.5)
            stop_distance = atr * stop_multiplier
        else:
            # Default to fixed percentage stop
            stop_distance = current_price * 0.05  # 5% stop
        
        # Calculate stop loss price based on signal direction
        if signal.get("direction") == "buy":
            stop_loss = current_price - stop_distance
        else:
            stop_loss = current_price + stop_distance
        
        # Use signal confidence to adjust position size
        confidence = signal.get("confidence", 0.5)
        
        # Calculate position size
        position_info = self.position_sizer.calculate_position_size(
            symbol=symbol,
            entry_price=current_price,
            stop_loss=stop_loss,
            signal_strength=signal.get("strength", 1.0),
            market_volatility=atr,
            confidence=confidence
        )
        
        return position_info
    
    def execute_trade(self, symbol: str, signal: Dict, position_info: Dict) -> Dict:
        """
        Execute a trade based on signal and position sizing.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal dictionary
            position_info: Position sizing information
            
        Returns:
            Dictionary with trade execution details
        """
        direction = signal.get("direction")
        if direction not in ["buy", "sell"]:
            logger.warning(f"Invalid trade direction: {direction}")
            return {"status": "error", "message": "Invalid trade direction"}
        
        # Check if API credentials are configured
        if not self.config["exchange"].get("api_key") or not self.config["exchange"].get("api_secret"):
            logger.warning("API credentials not configured, skipping trade execution")
            return {
                "status": "simulated", 
                "message": "API credentials not configured",
                "symbol": symbol,
                "direction": direction,
                "position_size": position_info.get("position_size"),
                "entry_price": position_info.get("entry_price"),
                "timestamp": datetime.now().isoformat()
            }
        
        # Check if we already have an open position for this symbol
        if symbol in self.open_positions:
            existing_position = self.open_positions[symbol]
            # If position is in the same direction, skip
            if existing_position.get("direction") == direction:
                logger.info(f"Already have an open {direction} position for {symbol}, skipping")
                return {"status": "skipped", "message": f"Position already open in {direction} direction"}
            # If position is in the opposite direction, close it first
            else:
                logger.info(f"Closing existing {existing_position.get('direction')} position for {symbol}")
                self.close_position(symbol)
        
        # Check maximum open positions limit
        max_positions = self.config["trading"].get("max_open_positions", 3)
        if len(self.open_positions) >= max_positions and symbol not in self.open_positions:
            logger.warning(f"Maximum open positions limit ({max_positions}) reached, skipping trade")
            return {"status": "skipped", "message": "Maximum open positions limit reached"}
        
        try:
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker["last"]
            
            # Determine order side and type
            side = "buy" if direction == "buy" else "sell"
            order_type = "market"  # Use market orders for simplicity
            
            # Calculate position size
            amount = position_info.get("position_size", 0)
            
            # Set stop loss
            stop_loss = position_info.get("stop_price")
            
            # Set minimum order size
            min_notional = 10  # Minimum order value in USDT
            if amount * current_price < min_notional:
                amount = min_notional / current_price
                logger.info(f"Adjusted order size to meet minimum notional value: {amount:.6f} {symbol}")
            
            # Execute the order
            logger.info(f"Executing {side} {order_type} order for {amount:.6f} {symbol} at ~{current_price}")
            
            # In testnet/paper trading mode, we can simulate the order
            if self.config["exchange"]["testnet"]:
                # Simulate order execution
                order = {
                    "id": f"simulated_{int(time.time())}",
                    "symbol": symbol,
                    "side": side,
                    "type": order_type,
                    "price": current_price,
                    "amount": amount,
                    "cost": amount * current_price,
                    "fee": {
                        "cost": amount * current_price * 0.001,  # Assume 0.1% fee
                        "currency": symbol.split('/')[1]
                    },
                    "timestamp": int(time.time() * 1000),
                    "datetime": datetime.now().isoformat(),
                    "status": "closed"
                }
                logger.info(f"Simulated order executed: {order['id']}")
            else:
                # Real order execution (when using real testnet API)
                order = self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=None  # For market orders
                )
                logger.info(f"Order executed: {order['id']}")
            
            # Record the trade in open positions
            position = {
                "symbol": symbol,
                "direction": direction,
                "entry_price": current_price,
                "amount": amount,
                "stop_loss": stop_loss,
                "entry_time": datetime.now().isoformat(),
                "order_id": order["id"],
                "signal": {
                    "type": signal.get("signal_type"),
                    "confidence": signal.get("confidence"),
                    "strength": signal.get("strength"),
                    "context_id": signal.get("context", {}).get("id")
                }
            }
            
            # Store position
            self.open_positions[symbol] = position
            
            # Add to order history
            self.order_history.append(order)
            
            # Save current positions to disk
            self._save_positions()
            
            return {
                "status": "success",
                "order": order,
                "position": position
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {"status": "error", "message": str(e)}
    
    def close_position(self, symbol: str) -> Dict:
        """
        Close an open position.
        
        Args:
            symbol: Trading symbol of the position to close
            
        Returns:
            Dictionary with position closing details
        """
        if symbol not in self.open_positions:
            logger.warning(f"No open position found for {symbol}")
            return {"status": "error", "message": "No open position found"}
        
        position = self.open_positions[symbol]
        
        try:
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker["last"]
            
            # Determine closing side (opposite of position direction)
            side = "sell" if position["direction"] == "buy" else "buy"
            order_type = "market"
            amount = position["amount"]
            
            # Execute the closing order
            logger.info(f"Closing position: {side} {order_type} order for {amount:.6f} {symbol} at ~{current_price}")
            
            # In testnet/paper trading mode, we can simulate the order
            if self.config["exchange"]["testnet"]:
                # Simulate order execution
                order = {
                    "id": f"simulated_{int(time.time())}",
                    "symbol": symbol,
                    "side": side,
                    "type": order_type,
                    "price": current_price,
                    "amount": amount,
                    "cost": amount * current_price,
                    "fee": {
                        "cost": amount * current_price * 0.001,  # Assume 0.1% fee
                        "currency": symbol.split('/')[1]
                    },
                    "timestamp": int(time.time() * 1000),
                    "datetime": datetime.now().isoformat(),
                    "status": "closed"
                }
                logger.info(f"Simulated closing order executed: {order['id']}")
            else:
                # Real order execution (when using real testnet API)
                order = self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=None  # For market orders
                )
                logger.info(f"Closing order executed: {order['id']}")
            
            # Calculate P&L
            entry_price = position["entry_price"]
            pnl = 0
            
            if position["direction"] == "buy":
                pnl = (current_price - entry_price) * amount
            else:
                pnl = (entry_price - current_price) * amount
            
            # Record the closed trade
            closed_trade = {
                "symbol": symbol,
                "direction": position["direction"],
                "entry_price": entry_price,
                "exit_price": current_price,
                "amount": amount,
                "entry_time": position["entry_time"],
                "exit_time": datetime.now().isoformat(),
                "pnl": pnl,
                "pnl_percent": (pnl / (entry_price * amount)) * 100,
                "success": pnl > 0,
                "entry_order_id": position["order_id"],
                "exit_order_id": order["id"],
                "signal": position.get("signal", {})
            }
            
            # Add to trade history
            self.trade_history.append(closed_trade)
            
            # Remove from open positions
            del self.open_positions[symbol]
            
            # Save current positions and trade history
            self._save_positions()
            self._save_trade_history()
            
            # Update the feedback loop with trade outcome
            self._update_feedback_loop(closed_trade)
            
            return {
                "status": "success",
                "order": order,
                "closed_trade": closed_trade
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {"status": "error", "message": str(e)}
    
    def check_stop_losses(self) -> List[Dict]:
        """
        Check all open positions for stop loss triggers.
        
        Returns:
            List of closed position details
        """
        closed_positions = []
        
        for symbol, position in list(self.open_positions.items()):
            try:
                # Get current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker["last"]
                
                # Check if stop loss is triggered
                stop_triggered = False
                
                if position["direction"] == "buy" and current_price <= position["stop_loss"]:
                    stop_triggered = True
                    logger.info(f"Stop loss triggered for {symbol} long position: {current_price} <= {position['stop_loss']}")
                elif position["direction"] == "sell" and current_price >= position["stop_loss"]:
                    stop_triggered = True
                    logger.info(f"Stop loss triggered for {symbol} short position: {current_price} >= {position['stop_loss']}")
                
                if stop_triggered:
                    # Close the position
                    result = self.close_position(symbol)
                    if result["status"] == "success":
                        closed_positions.append(result["closed_trade"])
                
            except Exception as e:
                logger.error(f"Error checking stop loss for {symbol}: {e}")
        
        return closed_positions
    
    def update_positions(self) -> None:
        """Update all open positions with current market prices and P&L."""
        for symbol, position in self.open_positions.items():
            try:
                # Get current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker["last"]
                
                # Calculate current P&L
                entry_price = position["entry_price"]
                amount = position["amount"]
                
                if position["direction"] == "buy":
                    pnl = (current_price - entry_price) * amount
                else:
                    pnl = (entry_price - current_price) * amount
                
                # Update position with current information
                position["current_price"] = current_price
                position["current_pnl"] = pnl
                position["current_pnl_percent"] = (pnl / (entry_price * amount)) * 100
                position["last_updated"] = datetime.now().isoformat()
                
                # Log position update
                logger.info(f"Updated {symbol} position: Price={current_price}, PnL={pnl:.2f} ({position['current_pnl_percent']:.2f}%)")
                
            except Exception as e:
                logger.error(f"Error updating position for {symbol}: {e}")
        
        # Save updated positions
        self._save_positions()
    
    def _save_positions(self) -> None:
        """Save current open positions to disk."""
        positions_file = Path("data/trades/open_positions.json")
        try:
            with open(positions_file, 'w') as f:
                json.dump(self.open_positions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
    
    def _save_trade_history(self) -> None:
        """Save trade history to disk."""
        trade_file = Path("data/trades/trade_history.json")
        try:
            with open(trade_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
    
    def _update_feedback_loop(self, closed_trade: Dict) -> None:
        """
        Update the RAG feedback loop with trade outcome.
        
        Args:
            closed_trade: Dictionary with closed trade information
        """
        logger.info(f"Updating feedback loop with trade outcome for {closed_trade['symbol']}")
        
        try:
            # Extract necessary information
            symbol = closed_trade["symbol"]
            entry_time = closed_trade["entry_time"]
            exit_time = closed_trade["exit_time"]
            entry_price = closed_trade["entry_price"]
            exit_price = closed_trade["exit_price"]
            direction = closed_trade["direction"]
            pnl = closed_trade["pnl"]
            
            # Extract context ID if available
            context_id = closed_trade.get("signal", {}).get("context_id")
            
            # Update the market context with trade outcome
            self.market_context.process_trade_outcome(
                symbol=symbol,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                direction=direction,
                profit_loss=pnl,
                context_id=context_id
            )
            
            logger.info(f"Successfully updated feedback loop for {symbol} trade")
            
        except Exception as e:
            logger.error(f"Error updating feedback loop: {e}")
    
    def load_state(self) -> None:
        """Load previously saved open positions and trade history."""
        positions_file = Path("data/trades/open_positions.json")
        trade_file = Path("data/trades/trade_history.json")
        
        try:
            if positions_file.exists():
                with open(positions_file, 'r') as f:
                    self.open_positions = json.load(f)
                logger.info(f"Loaded {len(self.open_positions)} open positions")
            
            if trade_file.exists():
                with open(trade_file, 'r') as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded {len(self.trade_history)} historical trades")
                
        except Exception as e:
            logger.error(f"Error loading saved state: {e}")
    
    def run_trading_cycle(self) -> Dict:
        """
        Run a complete trading cycle:
        1. Update positions
        2. Check stop losses
        3. Analyze markets
        4. Execute new trades
        
        Returns:
            Dictionary with cycle results
        """
        cycle_results = {
            "timestamp": datetime.now().isoformat(),
            "positions_updated": [],
            "stop_losses_triggered": [],
            "new_analyses": [],
            "new_trades": []
        }
        
        # Update current positions
        self.update_positions()
        cycle_results["positions_updated"] = list(self.open_positions.keys())
        
        # Check stop losses
        closed_positions = self.check_stop_losses()
        cycle_results["stop_losses_triggered"] = [p["symbol"] for p in closed_positions]
        
        # Get tradable symbols
        symbols = self.get_tradable_symbols()
        
        # Analyze markets and execute trades
        for symbol in symbols:
            # Skip if we already have an open position for this symbol
            if symbol in self.open_positions:
                continue
            
            # Skip if we've reached the maximum number of open positions
            max_positions = self.config["trading"].get("max_open_positions", 3)
            if len(self.open_positions) >= max_positions:
                break
            
            # Analyze market
            analysis = self.analyze_market(symbol)
            if not analysis:
                continue
            
            cycle_results["new_analyses"].append({
                "symbol": symbol,
                "composite_signal": analysis.get("composite_signal"),
                "trade_signal_present": analysis.get("trade_signal") is not None
            })
            
            # Check if we have a valid trade signal
            trade_signal = analysis.get("trade_signal")
            if trade_signal and trade_signal.get("direction") in ["buy", "sell"]:
                # Get current price
                current_price = self.data_collector.fetch_current_price(symbol)
                if not current_price:
                    logger.warning(f"Could not fetch current price for {symbol}")
                    continue
                
                # Calculate position size
                position_info = self.calculate_position_size(symbol, trade_signal, current_price)
                
                # Execute trade
                trade_result = self.execute_trade(symbol, trade_signal, position_info)
                
                cycle_results["new_trades"].append({
                    "symbol": symbol,
                    "direction": trade_signal.get("direction"),
                    "confidence": trade_signal.get("confidence"),
                    "status": trade_result.get("status")
                })
        
        # Save current state after cycle
        self._save_positions()
        self._save_trade_history()
        
        return cycle_results

def main():
    """Main function to run the trading execution system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Trading System - Trading Execution")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["cycle", "analyze", "positions"], default="cycle", 
                        help="Execution mode: run a trading cycle, analyze markets, or show positions")
    parser.add_argument("--symbol", type=str, help="Symbol to analyze (for analyze mode)")
    parser.add_argument("--close", type=str, help="Symbol to close position for")
    
    args = parser.parse_args()
    
    try:
        # Initialize trading executor
        executor = TradingExecutor(config_path=args.config)
        
        # Load previous state
        executor.load_state()
        
        # Handle close position request
        if args.close:
            symbol = args.close
            logger.info(f"Closing position for {symbol}")
            result = executor.close_position(symbol)
            print(f"Close position result: {result['status']}")
            return
        
        # Execute based on mode
        if args.mode == "cycle":
            logger.info("Running trading cycle")
            result = executor.run_trading_cycle()
            print(f"Trading cycle completed at {result['timestamp']}")
            print(f"Positions updated: {result['positions_updated']}")
            print(f"Stop losses triggered: {result['stop_losses_triggered']}")
            print(f"New trades: {len(result['new_trades'])}")
            
        elif args.mode == "analyze":
            if not args.symbol:
                symbols = executor.get_tradable_symbols()
                print(f"Available symbols: {symbols}")
                symbol = symbols[0] if symbols else None
                if not symbol:
                    print("No tradable symbols found")
                    return
            else:
                symbol = args.symbol
            
            print(f"Analyzing {symbol}...")
            analysis = executor.analyze_market(symbol)
            
            print(f"\nMarket Analysis for {symbol}:")
            print(f"Composite Signal: {analysis.get('composite_signal', 0):.2f}")
            print(f"Short-term Trend: {analysis.get('market_context', {}).get('short_term_trend')}")
            print(f"Medium-term Trend: {analysis.get('market_context', {}).get('medium_term_trend')}")
            print(f"Volatility: {analysis.get('market_context', {}).get('volatility')}")
            print(f"Patterns: {analysis.get('market_context', {}).get('patterns')}")
            
            # Check if there's a trade signal
            trade_signal = analysis.get("trade_signal")
            if trade_signal:
                print(f"\nTrade Signal: {trade_signal.get('direction')} with confidence {trade_signal.get('confidence', 0):.2f}")
                print(f"Signal Type: {trade_signal.get('signal_type')}")
                if "analysis" in trade_signal:
                    print(f"\nSignal Analysis:\n{trade_signal['analysis']}")
            else:
                print("\nNo actionable trade signal at this time")
            
        elif args.mode == "positions":
            print("\nCurrent Open Positions:")
            if not executor.open_positions:
                print("No open positions")
            else:
                for symbol, position in executor.open_positions.items():
                    pnl = position.get("current_pnl", 0)
                    pnl_percent = position.get("current_pnl_percent", 0)
                    print(f"{symbol}: {position['direction']} {position['amount']:.6f} @ {position['entry_price']} | "
                          f"PnL: {pnl:.2f} ({pnl_percent:.2f}%)")
            
            print("\nRecent Trade History:")
            for trade in executor.trade_history[-5:]:
                print(f"{trade['symbol']}: {trade['direction']} | "
                      f"Entry: {trade['entry_price']} | Exit: {trade['exit_price']} | "
                      f"PnL: {trade['pnl']:.2f} ({trade['pnl_percent']:.2f}%)")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()