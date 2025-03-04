#!/usr/bin/env python3
"""
Demo script for the PositionSizer module.

This script demonstrates how to use the PositionSizer module for different
position sizing methods and portfolio allocation.
"""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
from risk.position_sizing import PositionSizer

def create_demo_config():
    """Create a demo configuration file."""
    config = {
        "exchange": "binance",
        "testnet": True,
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "timeframes": ["1h", "4h", "1d"],
        "indicators": ["sma", "ema", "rsi", "macd", "bbands", "atr"],
        "strategies": {
            "sma_crossover": {
                "fast_period": 20,
                "slow_period": 50,
                "weight": 0.6
            },
            "rsi": {
                "period": 14,
                "overbought": 70,
                "oversold": 30,
                "weight": 0.4
            }
        },
        "risk_management": {
            "account_balance": 10000,
            "risk_per_trade": 0.02,
            "max_position_size": 0.2,
            "position_sizing_method": "fixed_risk"
        },
        "data_dir": "data/processed",
        "market_context_dir": "data/market_context"
    }
    
    with open("demo_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return "demo_config.json"

def demo_fixed_risk_position_sizing():
    """Demonstrate fixed risk position sizing."""
    print("\n===== Fixed Risk Position Sizing =====")
    
    # Create position sizer with fixed risk method
    config_path = create_demo_config()
    with open(config_path, "r") as f:
        config = json.load(f)
    
    config["risk_management"]["position_sizing_method"] = "fixed_risk"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    position_sizer = PositionSizer(config_path)
    
    # Example 1: BTC position with stop loss
    btc_position = position_sizer.calculate_position_size(
        symbol="BTC/USDT",
        entry_price=50000,
        stop_loss=48000,
        signal_strength=0.8,
        confidence=0.9
    )
    
    print("\nBTC Position with Stop Loss:")
    for key, value in btc_position.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Example 2: ETH position with no stop loss
    eth_position = position_sizer.calculate_position_size(
        symbol="ETH/USDT",
        entry_price=3000,
        signal_strength=0.7,
        confidence=0.8
    )
    
    print("\nETH Position without Stop Loss:")
    for key, value in eth_position.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

def demo_kelly_criterion_position_sizing():
    """Demonstrate Kelly Criterion position sizing."""
    print("\n===== Kelly Criterion Position Sizing =====")
    
    # Create position sizer with Kelly Criterion method
    config_path = create_demo_config()
    with open(config_path, "r") as f:
        config = json.load(f)
    
    config["risk_management"]["position_sizing_method"] = "kelly_criterion"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    position_sizer = PositionSizer(config_path)
    
    # Example: BTC position with stop loss and take profit
    btc_position = position_sizer.calculate_position_size(
        symbol="BTC/USDT",
        entry_price=50000,
        stop_loss=48000,
        take_profit=55000,
        confidence=0.75
    )
    
    print("\nBTC Position with Kelly Criterion:")
    for key, value in btc_position.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

def demo_volatility_adjusted_position_sizing():
    """Demonstrate volatility adjusted position sizing."""
    print("\n===== Volatility Adjusted Position Sizing =====")
    
    # Create position sizer with volatility adjusted method
    config_path = create_demo_config()
    with open(config_path, "r") as f:
        config = json.load(f)
    
    config["risk_management"]["position_sizing_method"] = "volatility_adjusted"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    position_sizer = PositionSizer(config_path)
    
    # Example 1: BTC position with low volatility
    btc_position_low_vol = position_sizer.calculate_position_size(
        symbol="BTC/USDT",
        entry_price=50000,
        market_volatility=0.02,  # 2% volatility
        confidence=0.8
    )
    
    print("\nBTC Position with Low Volatility (2%):")
    for key, value in btc_position_low_vol.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Example 2: BTC position with high volatility
    btc_position_high_vol = position_sizer.calculate_position_size(
        symbol="BTC/USDT",
        entry_price=50000,
        market_volatility=0.05,  # 5% volatility
        confidence=0.8
    )
    
    print("\nBTC Position with High Volatility (5%):")
    for key, value in btc_position_high_vol.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

def demo_portfolio_allocation():
    """Demonstrate portfolio allocation with multiple signals."""
    print("\n===== Portfolio Allocation =====")
    
    # Create position sizer
    config_path = create_demo_config()
    position_sizer = PositionSizer(config_path)
    
    # Create multiple trading signals
    signals = [
        {
            "symbol": "BTC/USDT",
            "entry_price": 50000,
            "stop_loss": 48000,
            "signal_strength": 0.9,
            "confidence": 0.8
        },
        {
            "symbol": "ETH/USDT",
            "entry_price": 3000,
            "stop_loss": 2850,
            "signal_strength": 0.7,
            "confidence": 0.75
        },
        {
            "symbol": "SOL/USDT",
            "entry_price": 120,
            "stop_loss": 110,
            "signal_strength": 0.8,
            "confidence": 0.7
        }
    ]
    
    # Calculate position sizes with portfolio constraints
    positions = position_sizer.calculate_portfolio_allocation(
        signals=signals,
        max_total_risk=0.05  # Maximum 5% total risk
    )
    
    print("\nPortfolio Allocation Results:")
    for i, position in enumerate(positions):
        print(f"\nPosition {i+1} - {position['symbol']}:")
        for key, value in position.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    
    # Calculate total portfolio risk
    total_risk = sum(position["position_risk"] for position in positions)
    print(f"\nTotal Portfolio Risk: {total_risk:.2%}")

def demo_position_updates():
    """Demonstrate position updates with price changes."""
    print("\n===== Position Updates =====")
    
    # Create position sizer
    config_path = create_demo_config()
    position_sizer = PositionSizer(config_path)
    
    # Create a position
    position = position_sizer.calculate_position_size(
        symbol="BTC/USDT",
        entry_price=50000,
        stop_loss=48000,
        signal_strength=0.9,
        confidence=0.8
    )
    
    print("\nOriginal Position:")
    for key, value in position.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Update position with price increase
    updated_position = position_sizer.update_position(
        position_info=position,
        current_price=52000,
        update_account_balance=True
    )
    
    print("\nUpdated Position (Price Increase to $52,000):")
    for key, value in updated_position.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Update position with price decrease
    updated_position = position_sizer.update_position(
        position_info=position,
        current_price=49000,
        update_account_balance=True
    )
    
    print("\nUpdated Position (Price Decrease to $49,000):")
    for key, value in updated_position.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    print("Position Sizing Module Demo")
    print("==========================")
    
    demo_fixed_risk_position_sizing()
    demo_kelly_criterion_position_sizing()
    demo_volatility_adjusted_position_sizing()
    demo_portfolio_allocation()
    demo_position_updates()
    
    # Clean up
    import os
    if os.path.exists("demo_config.json"):
        os.remove("demo_config.json")