#!/usr/bin/env python3
"""
Visualization utilities for trading signals and positions.

This script provides visualization functions for:
1. Trading signals on price charts
2. Position sizes and risk allocations
3. Confidence levels and adjustments from RAG
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import data collection for historical data
from data.collector import MarketDataCollector

def load_results(results_path: str) -> Dict:
    """
    Load analysis results from a JSON file.
    
    Args:
        results_path: Path to the results JSON file
        
    Returns:
        Dictionary containing analysis results
    """
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_signals_on_chart(signals: List[Dict], 
                          config_path: str = "config.json",
                          lookback_periods: int = 100,
                          save_fig: bool = False,
                          output_dir: str = "data/visualizations") -> None:
    """
    Plot trading signals on price charts.
    
    Args:
        signals: List of signal dictionaries
        config_path: Path to configuration file
        lookback_periods: Number of periods to look back for chart
        save_fig: Whether to save the figure to disk
        output_dir: Directory to save figures
    """
    if not signals:
        print("No signals to visualize")
        return
    
    # Group signals by symbol and timeframe
    signal_groups = {}
    for signal in signals:
        symbol = signal.get("symbol", "")
        timeframe = signal.get("timeframe", "1d")
        key = f"{symbol}_{timeframe}"
        
        if key not in signal_groups:
            signal_groups[key] = []
            
        signal_groups[key].append(signal)
    
    # Initialize market data collector
    data_collector = MarketDataCollector(config_path)
    
    # Create output directory if saving figures
    if save_fig:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot each group of signals
    for key, group_signals in signal_groups.items():
        symbol, timeframe = key.split("_")
        
        # Fetch historical data
        market_data = data_collector.fetch_data(symbol, timeframe, limit=lookback_periods)
        
        if market_data.empty:
            print(f"No data available for {symbol} on {timeframe} timeframe")
            continue
        
        # Convert to datetime index if not already
        if not isinstance(market_data.index, pd.DatetimeIndex):
            market_data.index = pd.to_datetime(market_data.index)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price chart
        market_data['close'].plot(ax=ax1, color='black', linewidth=1)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Plot signals
        for signal in group_signals:
            timestamp = signal.get("timestamp", "")
            direction = signal.get("direction", "").lower()
            confidence = signal.get("confidence", 0.5)
            
            # Convert timestamp to datetime
            try:
                signal_time = pd.to_datetime(timestamp)
                
                # Find closest price point
                closest_idx = market_data.index.get_indexer([signal_time], method='nearest')[0]
                signal_price = market_data.iloc[closest_idx]['close']
                
                # Determine marker properties based on signal type
                if direction == "buy":
                    marker = '^'
                    color = 'green'
                    offset = -0.02
                else:
                    marker = 'v'
                    color = 'red'
                    offset = 0.02
                
                # Scale marker size based on confidence
                markersize = 10 + (confidence * 10)
                
                # Plot signal marker
                ax1.plot(signal_time, signal_price * (1 + offset), 
                        marker=marker, markersize=markersize, 
                        color=color, alpha=0.7)
                
            except (ValueError, KeyError) as e:
                print(f"Error plotting signal at {timestamp}: {e}")
        
        # Plot confidence levels in bottom subplot
        timestamps = []
        confidences = []
        original_confidences = []
        colors = []
        
        for signal in group_signals:
            try:
                timestamps.append(pd.to_datetime(signal.get("timestamp", "")))
                confidences.append(signal.get("confidence", 0))
                original_confidences.append(signal.get("original_confidence", signal.get("confidence", 0)))
                
                # Determine color based on direction
                if signal.get("direction", "").lower() == "buy":
                    colors.append("green")
                else:
                    colors.append("red")
            except ValueError:
                continue
        
        # Plot confidence bars
        if timestamps:
            x_positions = range(len(timestamps))
            ax2.bar(x_positions, confidences, color=colors, alpha=0.7, label='Adjusted Confidence')
            ax2.plot(x_positions, original_confidences, 'o--', color='black', label='Original Confidence')
            
            # Add timestamp labels
            ax2.set_xticks(x_positions)
            ax2.set_xticklabels([ts.strftime('%m-%d %H:%M') for ts in timestamps], rotation=45)
            
            # Add confidence scale
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Confidence')
            ax2.legend()
        
        # Set titles and labels
        fig.suptitle(f'{symbol} Trading Signals - {timeframe} Timeframe', fontsize=16)
        ax1.set_title('Price Chart with Signals')
        ax1.set_ylabel('Price')
        ax2.set_title('Signal Confidence Levels')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show figure
        if save_fig:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_path = Path(output_dir) / f"{symbol}_{timeframe}_{timestamp}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {fig_path}")
        else:
            plt.show()
        
        plt.close(fig)

def plot_position_allocation(positions: List[Dict], 
                           save_fig: bool = False,
                           output_dir: str = "data/visualizations") -> None:
    """
    Plot position sizes and risk allocations.
    
    Args:
        positions: List of position dictionaries
        save_fig: Whether to save the figure to disk
        output_dir: Directory to save figures
    """
    if not positions:
        print("No positions to visualize")
        return
    
    # Extract position data
    symbols = [pos.get("symbol", "").split('/')[0] for pos in positions]
    position_values = [pos.get("position_value", 0) for pos in positions]
    position_risks = [pos.get("position_risk", 0) for pos in positions]
    
    # Calculate total position value
    total_value = sum(position_values)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot position values
    ax1.pie(position_values, labels=symbols, autopct='%1.1f%%', startangle=90,
           explode=[0.05] * len(symbols), shadow=True)
    ax1.set_title(f'Position Value Allocation (Total: ${total_value:.2f})')
    
    # Plot position risks
    risk_colors = ['green' if risk < 0.01 else
                  'blue' if risk < 0.02 else
                  'orange' if risk < 0.03 else
                  'red' for risk in position_risks]
    
    ax2.bar(symbols, position_risks, color=risk_colors)
    ax2.set_title('Position Risk Allocation')
    ax2.set_ylabel('Risk (fraction of portfolio)')
    ax2.set_ylim(0, max(position_risks) * 1.2)
    
    # Add risk reference lines
    ax2.axhline(y=0.01, linestyle='--', color='green', alpha=0.5, label='Low Risk (1%)')
    ax2.axhline(y=0.02, linestyle='--', color='blue', alpha=0.5, label='Medium Risk (2%)')
    ax2.axhline(y=0.03, linestyle='--', color='red', alpha=0.5, label='High Risk (3%)')
    ax2.legend()
    
    # Add position details as table
    position_table = []
    for pos in positions:
        symbol = pos.get("symbol", "")
        position_size = pos.get("position_size", 0)
        position_value = pos.get("position_value", 0)
        risk = pos.get("position_risk", 0)
        position_table.append([symbol, f"{position_size:.6f}", f"${position_value:.2f}", f"{risk*100:.2f}%"])
    
    fig.suptitle('Position and Risk Allocation', fontsize=16)
    
    # Add table below plots
    table_ax = plt.subplot(212)
    table_ax.axis('off')
    table = table_ax.table(
        cellText=position_table,
        colLabels=['Symbol', 'Size', 'Value', 'Risk'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show figure
    if save_fig:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = Path(output_dir) / f"position_allocation_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
    else:
        plt.show()
    
    plt.close(fig)

def plot_confidence_adjustments(signals: List[Dict],
                              save_fig: bool = False,
                              output_dir: str = "data/visualizations") -> None:
    """
    Plot confidence adjustments from RAG system.
    
    Args:
        signals: List of signal dictionaries
        save_fig: Whether to save the figure to disk
        output_dir: Directory to save figures
    """
    if not signals:
        print("No signals to visualize")
        return
    
    # Extract data for signals with confidence adjustments
    adjusted_signals = []
    for signal in signals:
        if "original_confidence" in signal and "confidence" in signal:
            adjusted_signals.append(signal)
    
    if not adjusted_signals:
        print("No signals with confidence adjustments found")
        return
    
    # Sort signals by adjustment magnitude
    adjusted_signals.sort(key=lambda x: abs(x.get("confidence", 0) - x.get("original_confidence", 0)), reverse=True)
    
    # Take top 10 signals with largest adjustments
    top_signals = adjusted_signals[:min(10, len(adjusted_signals))]
    
    # Extract data
    symbols = [f"{sig.get('symbol', '').split('/')[0]} ({sig.get('direction', '').upper()})" for sig in top_signals]
    original_confidences = [sig.get("original_confidence", 0) for sig in top_signals]
    adjusted_confidences = [sig.get("confidence", 0) for sig in top_signals]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot data
    x = range(len(symbols))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], original_confidences, width, label='Original Confidence', color='lightblue')
    ax.bar([i + width/2 for i in x], adjusted_confidences, width, label='Adjusted Confidence', color='darkblue')
    
    # Add labels and title
    ax.set_ylabel('Confidence')
    ax.set_title('Confidence Adjustments from Market Context')
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    
    # Add text showing adjustment
    for i, (orig, adj) in enumerate(zip(original_confidences, adjusted_confidences)):
        adjustment = adj - orig
        color = 'green' if adjustment > 0 else 'red'
        ax.annotate(f"{adjustment:+.2f}", 
                   xy=(i, max(orig, adj) + 0.02),
                   ha='center',
                   va='bottom',
                   color=color,
                   weight='bold')
    
    # Add explanation of context factors
    ax.text(0.5, -0.15, 
           "Context factors: Technical patterns, Historical signals, Market regime, Relevant news",
           transform=ax.transAxes, ha='center', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show figure
    if save_fig:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = Path(output_dir) / f"confidence_adjustments_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
    else:
        plt.show()
    
    plt.close(fig)

def main():
    """Main function to visualize trading system results."""
    parser = argparse.ArgumentParser(description="Visualize AI Trading System Results")
    parser.add_argument("--results", type=str, required=True, help="Path to analysis results JSON file")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--lookback", type=int, default=100, help="Number of periods to look back for charts")
    parser.add_argument("--save", action="store_true", help="Save figures to disk")
    parser.add_argument("--output-dir", type=str, default="data/visualizations", help="Directory to save figures")
    
    args = parser.parse_args()
    
    try:
        # Load results
        results = load_results(args.results)
        
        # Extract signals and positions
        signals = results.get("signals", [])
        positions = results.get("positions", [])
        
        print(f"Loaded {len(signals)} signals and {len(positions)} positions")
        
        # Plot signals on charts
        plot_signals_on_chart(
            signals=signals,
            config_path=args.config,
            lookback_periods=args.lookback,
            save_fig=args.save,
            output_dir=args.output_dir
        )
        
        # Plot position allocation
        plot_position_allocation(
            positions=positions,
            save_fig=args.save,
            output_dir=args.output_dir
        )
        
        # Plot confidence adjustments
        plot_confidence_adjustments(
            signals=signals,
            save_fig=args.save,
            output_dir=args.output_dir
        )
        
    except Exception as e:
        print(f"Error visualizing results: {e}")

if __name__ == "__main__":
    main()