import json
import numpy as np
from typing import Dict, Optional, Tuple, Union, List

class PositionSizer:
    """
    Position sizing and risk management module for trading strategies.
    
    This class implements various position sizing methods based on risk parameters
    and market conditions.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the PositionSizer with configuration parameters.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.account_balance = self.config["risk_management"]["account_balance"]
        self.risk_per_trade = self.config["risk_management"]["risk_per_trade"]
        self.max_position_size = self.config["risk_management"]["max_position_size"]
        self.position_sizing_method = self.config["risk_management"]["position_sizing_method"]
        
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
            raise Exception(f"Error loading configuration: {e}")
    
    def calculate_position_size(self, 
                               symbol: str,
                               entry_price: float,
                               stop_loss: Optional[float] = None,
                               take_profit: Optional[float] = None,
                               signal_strength: float = 1.0,
                               market_volatility: Optional[float] = None,
                               confidence: float = 1.0) -> Dict:
        """
        Calculate position size based on risk parameters and market conditions.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            entry_price: Entry price for the position
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            signal_strength: Strength of the trading signal (0.0 to 1.0)
            market_volatility: Market volatility indicator (e.g., ATR)
            confidence: Confidence level in the trade (0.0 to 1.0)
            
        Returns:
            Dict containing position sizing information
        """
        # Select position sizing method
        if self.position_sizing_method == "fixed_risk":
            position_info = self._fixed_risk_position_size(entry_price, stop_loss, signal_strength, confidence)
        elif self.position_sizing_method == "kelly_criterion":
            position_info = self._kelly_criterion_position_size(entry_price, take_profit, stop_loss, confidence)
        elif self.position_sizing_method == "volatility_adjusted":
            position_info = self._volatility_adjusted_position_size(entry_price, market_volatility, confidence)
        elif self.position_sizing_method == "fixed_percentage":
            position_info = self._fixed_percentage_position_size(signal_strength, confidence)
        else:
            raise ValueError(f"Unknown position sizing method: {self.position_sizing_method}")
        
        # Apply maximum position size constraint
        max_position_value = self.account_balance * self.max_position_size
        if position_info["position_value"] > max_position_value:
            position_info["position_value"] = max_position_value
            position_info["position_size"] = max_position_value / entry_price
            position_info["adjusted"] = True
            position_info["adjustment_reason"] = "max_position_size_exceeded"
        
        # Add symbol and entry price to the result
        position_info["symbol"] = symbol
        position_info["entry_price"] = entry_price
        
        return position_info
    
    def _fixed_risk_position_size(self, 
                                 entry_price: float, 
                                 stop_loss: Optional[float],
                                 signal_strength: float = 1.0,
                                 confidence: float = 1.0) -> Dict:
        """
        Calculate position size based on fixed risk per trade.
        
        Args:
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            signal_strength: Strength of the trading signal (0.0 to 1.0)
            confidence: Confidence level in the trade (0.0 to 1.0)
            
        Returns:
            Dict containing position sizing information
        """
        # Calculate risk amount
        risk_amount = self.account_balance * self.risk_per_trade * signal_strength * confidence
        
        # Calculate position size
        if stop_loss is not None:
            # Calculate risk per unit based on stop loss
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit == 0:
                raise ValueError("Stop loss cannot be equal to entry price")
            
            position_size = risk_amount / risk_per_unit
            position_value = position_size * entry_price
        else:
            # Default to 1% of risk if no stop loss specified
            position_value = risk_amount * 10  # Assuming 10% price movement without stop loss
            position_size = position_value / entry_price
        
        return {
            "position_size": position_size,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "sizing_method": "fixed_risk",
            "adjusted": False
        }
    
    def _kelly_criterion_position_size(self,
                                      entry_price: float,
                                      take_profit: Optional[float],
                                      stop_loss: Optional[float],
                                      confidence: float = 1.0) -> Dict:
        """
        Calculate position size based on the Kelly Criterion.
        
        Args:
            entry_price: Entry price for the position
            take_profit: Take profit price
            stop_loss: Stop loss price
            confidence: Confidence level in the trade (0.0 to 1.0)
            
        Returns:
            Dict containing position sizing information
        """
        if take_profit is None or stop_loss is None:
            raise ValueError("Take profit and stop loss are required for Kelly Criterion")
        
        # Calculate win/loss ratio and probability of winning (estimated from confidence)
        win_amount = (take_profit - entry_price) / entry_price
        loss_amount = (entry_price - stop_loss) / entry_price
        
        # Adjust win probability based on win/loss ratio and confidence
        # This is a simplified approach - in practice, this might be derived from backtesting
        win_probability = 0.5 + (confidence - 0.5) * 0.5  # Range from 0.25 to 0.75 based on confidence
        
        # Calculate Kelly fraction
        if loss_amount == 0:
            kelly_fraction = 1.0  # If no risk of loss, use maximum position size
        else:
            kelly_fraction = (win_probability * (1 + win_amount) - 1) / win_amount
            
        # Apply half-Kelly for more conservative sizing (common practice)
        kelly_fraction = max(0, kelly_fraction * 0.5)
        
        # Calculate position value and size
        position_value = self.account_balance * kelly_fraction * self.risk_per_trade
        position_size = position_value / entry_price
        
        return {
            "position_size": position_size,
            "position_value": position_value,
            "kelly_fraction": kelly_fraction,
            "win_probability": win_probability,
            "sizing_method": "kelly_criterion",
            "adjusted": False
        }
    
    def _volatility_adjusted_position_size(self,
                                          entry_price: float,
                                          market_volatility: Optional[float],
                                          confidence: float = 1.0) -> Dict:
        """
        Calculate position size based on market volatility.
        
        Args:
            entry_price: Entry price for the position
            market_volatility: Market volatility indicator (e.g., ATR)
            confidence: Confidence level in the trade (0.0 to 1.0)
            
        Returns:
            Dict containing position sizing information
        """
        if market_volatility is None or market_volatility <= 0:
            raise ValueError("Valid market volatility value is required for volatility_adjusted method")
        
        # Calculate risk amount
        risk_amount = self.account_balance * self.risk_per_trade * confidence
        
        # Calculate position size inversely proportional to volatility
        volatility_factor = 1.0 / market_volatility
        position_size = (risk_amount * volatility_factor) / entry_price
        position_value = position_size * entry_price
        
        return {
            "position_size": position_size,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "volatility_factor": volatility_factor,
            "sizing_method": "volatility_adjusted",
            "adjusted": False
        }
    
    def _fixed_percentage_position_size(self,
                                        signal_strength: float = 1.0,
                                        confidence: float = 1.0) -> Dict:
        """
        Calculate position size based on a fixed percentage of account balance.
        
        Args:
            signal_strength: Strength of the trading signal (0.0 to 1.0)
            confidence: Confidence level in the trade (0.0 to 1.0)
            
        Returns:
            Dict containing position sizing information
        """
        # Calculate position value as a percentage of account balance
        position_percentage = self.risk_per_trade * 10  # Typically 10x the risk percentage
        adjusted_percentage = position_percentage * signal_strength * confidence
        position_value = self.account_balance * adjusted_percentage
        
        return {
            "position_size": None,  # Will be calculated after entry price is known
            "position_value": position_value,
            "position_percentage": adjusted_percentage,
            "sizing_method": "fixed_percentage",
            "adjusted": False
        }
    
    def calculate_portfolio_allocation(self, 
                                      signals: List[Dict], 
                                      max_total_risk: float = 0.2) -> List[Dict]:
        """
        Calculate position sizes for multiple signals with portfolio constraints.
        
        Args:
            signals: List of dictionaries containing signal information
                Each signal should have: symbol, entry_price, stop_loss, 
                take_profit (optional), signal_strength, confidence
            max_total_risk: Maximum portfolio risk exposure (fraction of account)
            
        Returns:
            List of dictionaries with position sizing information for each signal
        """
        # Calculate initial position sizes
        positions = []
        total_risk = 0
        
        for signal in signals:
            position_info = self.calculate_position_size(
                symbol=signal["symbol"],
                entry_price=signal["entry_price"],
                stop_loss=signal.get("stop_loss"),
                take_profit=signal.get("take_profit"),
                signal_strength=signal.get("signal_strength", 1.0),
                market_volatility=signal.get("market_volatility"),
                confidence=signal.get("confidence", 1.0)
            )
            
            # Calculate risk for this position
            if "risk_amount" in position_info:
                position_risk = position_info["risk_amount"] / self.account_balance
            else:
                # Estimate risk for methods without explicit risk_amount
                position_risk = (position_info["position_value"] / self.account_balance) * self.risk_per_trade
            
            position_info["position_risk"] = position_risk
            total_risk += position_risk
            positions.append(position_info)
        
        # Adjust position sizes if total risk exceeds maximum
        if total_risk > max_total_risk and total_risk > 0:
            scaling_factor = max_total_risk / total_risk
            
            for position in positions:
                position["position_value"] *= scaling_factor
                if "position_size" in position and position["position_size"] is not None:
                    position["position_size"] *= scaling_factor
                if "risk_amount" in position:
                    position["risk_amount"] *= scaling_factor
                position["position_risk"] *= scaling_factor
                position["adjusted"] = True
                position["adjustment_reason"] = "portfolio_risk_management"
                
        return positions
    
    def update_position(self, 
                       position_info: Dict, 
                       current_price: float,
                       update_account_balance: bool = False) -> Dict:
        """
        Update position information based on current price.
        
        Args:
            position_info: Dictionary containing position information
            current_price: Current market price
            update_account_balance: Whether to update account balance based on P&L
            
        Returns:
            Updated position information
        """
        # Copy original position info
        updated_position = position_info.copy()
        
        # Calculate P&L
        entry_price = position_info["entry_price"]
        position_size = position_info["position_size"]
        pnl = (current_price - entry_price) * position_size
        pnl_percentage = (current_price / entry_price - 1) * 100
        
        # Update position info
        updated_position["current_price"] = current_price
        updated_position["pnl"] = pnl
        updated_position["pnl_percentage"] = pnl_percentage
        
        # Update account balance if requested
        if update_account_balance:
            original_balance = self.account_balance
            self.account_balance += pnl
            updated_position["previous_account_balance"] = original_balance
            updated_position["updated_account_balance"] = self.account_balance
        
        return updated_position