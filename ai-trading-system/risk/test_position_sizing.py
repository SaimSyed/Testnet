#!/usr/bin/env python3
"""
Test cases for the PositionSizer module.
"""

import unittest
import json
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from risk.position_sizing import PositionSizer

class TestPositionSizer(unittest.TestCase):
    """Test cases for the PositionSizer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary config file
        self.config = {
            "risk_management": {
                "account_balance": 10000,
                "risk_per_trade": 0.02,
                "max_position_size": 0.2,
                "position_sizing_method": "fixed_risk"
            }
        }
        
        # Create a temporary file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json')
        json.dump(self.config, self.temp_file)
        self.temp_file.close()
        
        # Create PositionSizer instance
        self.position_sizer = PositionSizer(self.temp_file.name)
    
    def tearDown(self):
        """Clean up after tests."""
        # Delete temporary file
        os.unlink(self.temp_file.name)
    
    def test_init(self):
        """Test initialization of PositionSizer."""
        self.assertEqual(self.position_sizer.account_balance, 10000)
        self.assertEqual(self.position_sizer.risk_per_trade, 0.02)
        self.assertEqual(self.position_sizer.max_position_size, 0.2)
        self.assertEqual(self.position_sizer.position_sizing_method, "fixed_risk")
    
    def test_fixed_risk_position_size(self):
        """Test fixed risk position sizing."""
        # Calculate position size with stop loss
        position = self.position_sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss=48000
        )
        
        # Expected risk amount: 10000 * 0.02 = 200
        # Expected risk per unit: 50000 - 48000 = 2000
        # Expected position size: 200 / 2000 = 0.1
        # Expected position value: 0.1 * 50000 = 5000
        
        self.assertEqual(position["symbol"], "BTC/USDT")
        self.assertEqual(position["entry_price"], 50000)
        self.assertAlmostEqual(position["risk_amount"], 200)
        self.assertAlmostEqual(position["position_size"], 0.1)
        self.assertAlmostEqual(position["position_value"], 5000)
        self.assertEqual(position["sizing_method"], "fixed_risk")
        self.assertFalse(position["adjusted"])
    
    def test_fixed_risk_position_size_with_signal_strength(self):
        """Test fixed risk position sizing with signal strength adjustment."""
        # Calculate position size with signal strength
        position = self.position_sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss=48000,
            signal_strength=0.5,
            confidence=0.8
        )
        
        # Expected risk amount: 10000 * 0.02 * 0.5 * 0.8 = 80
        # Expected risk per unit: 50000 - 48000 = 2000
        # Expected position size: 80 / 2000 = 0.04
        # Expected position value: 0.04 * 50000 = 2000
        
        self.assertAlmostEqual(position["risk_amount"], 80)
        self.assertAlmostEqual(position["position_size"], 0.04)
        self.assertAlmostEqual(position["position_value"], 2000)
    
    def test_max_position_size_constraint(self):
        """Test maximum position size constraint."""
        # Set up a position that would exceed max_position_size
        position = self.position_sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss=49900  # Very tight stop loss to force large position
        )
        
        # Maximum position value: 10000 * 0.2 = 2000
        # Maximum position size: 2000 / 50000 = 0.04
        
        self.assertAlmostEqual(position["position_value"], 2000)
        self.assertAlmostEqual(position["position_size"], 0.04)
        self.assertTrue(position["adjusted"])
        self.assertEqual(position["adjustment_reason"], "max_position_size_exceeded")
    
    def test_kelly_criterion_position_size(self):
        """Test Kelly Criterion position sizing."""
        # Set config to use Kelly Criterion
        with open(self.temp_file.name, 'w') as f:
            self.config["risk_management"]["position_sizing_method"] = "kelly_criterion"
            json.dump(self.config, f)
        
        position_sizer = PositionSizer(self.temp_file.name)
        
        # Calculate position size with Kelly Criterion
        position = position_sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss=48000,
            take_profit=55000,
            confidence=0.7
        )
        
        # Verify the position sizing method and check for reasonable results
        self.assertEqual(position["sizing_method"], "kelly_criterion")
        self.assertTrue(0 < position["position_value"] < 10000 * 0.2)  # Should be less than max position
        self.assertTrue(0 < position["kelly_fraction"] < 1)  # Kelly fraction should be between 0 and 1
        self.assertTrue(0.5 < position["win_probability"] < 0.75)  # Based on our confidence adjustment
    
    def test_volatility_adjusted_position_size(self):
        """Test volatility adjusted position sizing."""
        # Set config to use volatility adjusted method
        with open(self.temp_file.name, 'w') as f:
            self.config["risk_management"]["position_sizing_method"] = "volatility_adjusted"
            json.dump(self.config, f)
        
        position_sizer = PositionSizer(self.temp_file.name)
        
        # Calculate position with low volatility
        low_vol_position = position_sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            market_volatility=0.02,  # 2% volatility
            confidence=0.8
        )
        
        # Calculate position with high volatility
        high_vol_position = position_sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            market_volatility=0.05,  # 5% volatility
            confidence=0.8
        )
        
        # Position size should be inversely proportional to volatility
        # So low volatility should yield a larger position
        self.assertTrue(low_vol_position["position_value"] > high_vol_position["position_value"])
        self.assertEqual(low_vol_position["sizing_method"], "volatility_adjusted")
        self.assertEqual(high_vol_position["sizing_method"], "volatility_adjusted")
    
    def test_portfolio_allocation(self):
        """Test portfolio allocation with risk constraints."""
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
            }
        ]
        
        # Calculate positions with portfolio constraints
        positions = self.position_sizer.calculate_portfolio_allocation(
            signals=signals,
            max_total_risk=0.03  # Maximum 3% total risk
        )
        
        # Check that we have 2 positions
        self.assertEqual(len(positions), 2)
        
        # Calculate total risk and ensure it doesn't exceed max_total_risk
        total_risk = sum(position["position_risk"] for position in positions)
        self.assertLessEqual(total_risk, 0.03)
        
        # Check that positions were adjusted
        self.assertTrue(positions[0]["adjusted"])
        self.assertTrue(positions[1]["adjusted"])
        self.assertEqual(positions[0]["adjustment_reason"], "portfolio_risk_management")
    
    def test_update_position(self):
        """Test position update with price changes."""
        # Create a position
        position = self.position_sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss=48000
        )
        
        # Update position with price increase
        updated_position = self.position_sizer.update_position(
            position_info=position,
            current_price=52000,
            update_account_balance=True
        )
        
        # Calculate expected P&L
        expected_pnl = (52000 - 50000) * position["position_size"]
        expected_pnl_percentage = (52000 / 50000 - 1) * 100
        
        # Verify results
        self.assertEqual(updated_position["current_price"], 52000)
        self.assertAlmostEqual(updated_position["pnl"], expected_pnl)
        self.assertAlmostEqual(updated_position["pnl_percentage"], expected_pnl_percentage)
        self.assertEqual(updated_position["updated_account_balance"], 10000 + expected_pnl)
        
        # Update with price decrease
        updated_position = self.position_sizer.update_position(
            position_info=position,
            current_price=49000,
            update_account_balance=True
        )
        
        # Calculate expected P&L for decrease
        expected_pnl = (49000 - 50000) * position["position_size"]
        expected_pnl_percentage = (49000 / 50000 - 1) * 100
        
        # Verify results
        self.assertAlmostEqual(updated_position["pnl"], expected_pnl)
        self.assertAlmostEqual(updated_position["pnl_percentage"], expected_pnl_percentage)
        self.assertTrue(updated_position["pnl"] < 0)  # P&L should be negative

if __name__ == "__main__":
    unittest.main()