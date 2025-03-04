#!/usr/bin/env python3
"""
Signal Enricher Module for AI Trading System.

This module enhances trading signals with historical context using RAG (Retrieval
Augmented Generation) to identify similar market patterns and outcomes.
"""

import json
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import datetime
import google.generativeai as genai
from pathlib import Path

# Import local modules
from rag.market_context import MarketContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('signal_enricher')

class SignalEnricher:
    """
    Enhances trading signals with market context and historical pattern matching.
    
    This class uses RAG (Retrieval Augmented Generation) to:
    1. Find similar historical market contexts
    2. Analyze historical outcomes for similar patterns
    3. Adjust signal confidence based on historical success rates
    4. Generate enhanced trading signals with context-aware information
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the SignalEnricher.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.market_context = MarketContext(config_path)
        
        # Initialize flag for Gemini availability
        self.gemini_available = False
        
        # Initialize Gemini API
        api_key = self.config.get("gemini", {}).get("api_key", "")
        if not api_key:
            logger.warning("Gemini API key not found in config. RAG functionality will be limited.")
        else:
            try:
                genai.configure(api_key=api_key)
                self.gemini_available = True
                
                # Set up the model
                generation_config = {
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 4096,
                }
                
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
                
                # Use the model specified in the config
                model_name = self.config.get("gemini", {}).get("model", "gemini-1.5-pro")
                
                # Validate model name - ensure it's one of the supported models
                supported_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.0-pro"]
                if model_name not in supported_models:
                    logger.warning(f"Model '{model_name}' may not be supported. Using 'gemini-1.5-pro' as fallback.")
                    model_name = "gemini-1.5-pro"
                
                logger.info(f"Initializing Gemini with model: {model_name}")
                self.model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                logger.info("Gemini API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API with model '{model_name}': {e}")
                logger.error("Ensure your API key is correct and the specified model is accessible to your account")
                self.gemini_available = False
        
        # Create dirs for storing enriched signals
        self.signal_storage_path = Path(self.config.get("rag", {}).get("signal_db_path", "data/signals"))
        self.signal_storage_path.mkdir(parents=True, exist_ok=True)
    
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
            logger.warning(f"Error loading configuration: {e}. Using default values.")
            return {
                "gemini": {
                    "api_key": "",
                    "model": "gemini-1.5-pro"
                },
                "rag": {
                    "signal_db_path": "data/signals",
                    "min_similarity_score": 0.5,
                    "max_contexts": 5
                },
                "trading": {
                    "default_confidence": 0.5,
                    "min_confidence_for_trade": 0.6
                }
            }
    
    def enrich_signal(self, signal: Dict, market_data: Optional[Dict] = None) -> Dict:
        """
        Enrich a trading signal with market context and similar patterns.
        
        Args:
            signal: Original trading signal dictionary
            market_data: Optional pre-fetched market data
            
        Returns:
            Enriched signal with additional context and adjusted confidence
        """
        if not signal:
            logger.warning("Cannot enrich empty signal")
            return {}
        
        symbol = signal.get("symbol", "")
        if not symbol:
            logger.warning("Signal missing symbol information")
            return signal
        
        # Track original confidence for comparison
        original_confidence = signal.get("confidence", 0.5)
        signal["original_confidence"] = original_confidence
        
        # Gather market context if not provided
        if not market_data:
            market_context = self.market_context.gather_market_context(symbol)
        else:
            # If market data is provided but not in the right format, process it
            if isinstance(market_data, dict) and "timeframes" in market_data:
                market_context = market_data
            else:
                # Convert provided market data to context format
                market_context = self.market_context.create_market_snapshot(symbol, market_data)
                # Store for future use
                self.market_context.store_context(market_context)
        
        # If we couldn't get market context, return original signal
        if not market_context:
            logger.warning(f"Could not gather market context for {symbol}")
            return signal
        
        # Add basic context information to signal
        signal["context"] = {
            "timestamp": market_context.get("timestamp", ""),
            "timeframes": list(market_context.get("timeframes", {}).keys()),
            "patterns": [p.get("pattern") for p in market_context.get("patterns", [])],
            "short_term_trend": market_context.get("aggregated_signals", {}).get("short_term_trend"),
            "medium_term_trend": market_context.get("aggregated_signals", {}).get("medium_term_trend"),
            "volatility": market_context.get("aggregated_signals", {}).get("volatility"),
            "momentum": market_context.get("aggregated_signals", {}).get("momentum")
        }
        
        # Find similar historical contexts if not already included
        if "similar_contexts" not in market_context:
            similar_contexts = self.market_context.find_similar_contexts(
                market_context,
                top_n=self.config.get("rag", {}).get("max_contexts", 5),
                min_similarity=self.config.get("rag", {}).get("min_similarity_score", 0.5)
            )
            market_context["similar_contexts"] = similar_contexts
        
        # Extract similar contexts
        similar_contexts = market_context.get("similar_contexts", [])
        
        # Calculate confidence adjustment based on similar contexts
        if similar_contexts:
            confidence_adjustment = self._calculate_confidence_adjustment(
                signal, market_context, similar_contexts
            )
            
            # Apply confidence adjustment
            adjusted_confidence = original_confidence + confidence_adjustment
            # Ensure confidence is within valid range
            adjusted_confidence = max(0.1, min(0.95, adjusted_confidence))
            
            signal["confidence"] = adjusted_confidence
            signal["confidence_adjustment"] = confidence_adjustment
            signal["confidence_factors"] = self._get_confidence_factors(signal, market_context, similar_contexts)
        
        # Add similarity context summary
        signal["context"]["similar_patterns"] = [
            {
                "similarity_score": ctx.get("similarity_score", 0),
                "timestamp": ctx.get("timestamp", ""),
                "short_term_trend": ctx.get("aggregated_signals", {}).get("short_term_trend"),
                "patterns": [p.get("pattern") for p in ctx.get("patterns", [])][:3],
                "success_rate": ctx.get("success_rate", None)
            }
            for ctx in similar_contexts[:3]  # Include only top 3 for brevity
        ]
        
        # If Gemini is available, generate analysis with LLM
        if self.gemini_available:
            analysis = self._generate_analysis(signal, market_context, similar_contexts)
            if analysis:
                signal["analysis"] = analysis
        
        # Store the enriched signal for future reference
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        signal_file = self.signal_storage_path / f"{symbol.replace('/', '_')}_{timestamp}.json"
        try:
            with open(signal_file, 'w') as f:
                json.dump(signal, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving enriched signal: {e}")
        
        return signal
    
    def _calculate_confidence_adjustment(self, 
                                        signal: Dict, 
                                        current_context: Dict,
                                        similar_contexts: List[Dict]) -> float:
        """
        Calculate confidence adjustment based on similar historical contexts.
        
        Args:
            signal: Original trading signal
            current_context: Current market context
            similar_contexts: List of similar historical contexts
            
        Returns:
            Confidence adjustment value (-0.3 to +0.3)
        """
        if not similar_contexts:
            return 0.0
        
        # Initialize adjustment factors
        success_rate_factor = 0.0
        similarity_weighted_factor = 0.0
        trend_alignment_factor = 0.0
        pattern_match_factor = 0.0
        
        # Direction of the signal (1 for buy, -1 for sell)
        signal_direction = 1 if signal.get("direction", "").lower() in ["buy", "long"] else -1
        
        # Get total weight for weighted average
        total_similarity = sum(ctx.get("similarity_score", 0) for ctx in similar_contexts)
        if total_similarity == 0:
            return 0.0
        
        # Calculate weighted factors
        for ctx in similar_contexts:
            # Similarity score for this context
            similarity = ctx.get("similarity_score", 0)
            weight = similarity / total_similarity
            
            # Success rate from historical trades
            if "trade_outcomes" in ctx:
                outcomes = ctx.get("trade_outcomes", [])
                if outcomes:
                    successes = sum(1 for o in outcomes if o.get("success", False))
                    ctx_success_rate = successes / len(outcomes)
                    
                    # Adjust based on direction alignment
                    direction_matches = sum(1 for o in outcomes 
                                          if (o.get("direction", "").lower() in ["buy", "long"]) == (signal_direction > 0)
                                          and o.get("success", False))
                    direction_total = sum(1 for o in outcomes 
                                        if (o.get("direction", "").lower() in ["buy", "long"]) == (signal_direction > 0))
                    
                    if direction_total > 0:
                        direction_success_rate = direction_matches / direction_total
                        # More weight to direction-specific success rate
                        ctx_success_rate = (ctx_success_rate * 0.3) + (direction_success_rate * 0.7)
                    
                    # Scale to adjustment range (-0.2 to +0.2)
                    # If success rate is above 0.5, positive adjustment, otherwise negative
                    success_factor = (ctx_success_rate - 0.5) * 0.4
                    success_rate_factor += success_factor * weight
            
            # Trend alignment factor
            curr_trend = current_context.get("aggregated_signals", {}).get("short_term_trend")
            hist_trend = ctx.get("aggregated_signals", {}).get("short_term_trend")
            
            if curr_trend and hist_trend:
                trend_alignment = False
                
                # Check if trends align
                if signal_direction > 0:  # Buy signal
                    if (curr_trend in ["bullish", "strong_bullish"] and 
                        hist_trend in ["bullish", "strong_bullish"]):
                        trend_alignment = True
                else:  # Sell signal
                    if (curr_trend in ["bearish", "strong_bearish"] and 
                        hist_trend in ["bearish", "strong_bearish"]):
                        trend_alignment = True
                
                # Adjust based on trend alignment
                if trend_alignment:
                    trend_alignment_factor += 0.1 * weight
                elif ((curr_trend in ["bullish", "strong_bullish"] and hist_trend in ["bearish", "strong_bearish"]) or
                      (curr_trend in ["bearish", "strong_bearish"] and hist_trend in ["bullish", "strong_bullish"])):
                    # Opposite trends should reduce confidence
                    trend_alignment_factor -= 0.15 * weight
            
            # Pattern match factor
            curr_patterns = {p.get("pattern") for p in current_context.get("patterns", [])}
            hist_patterns = {p.get("pattern") for p in ctx.get("patterns", [])}
            
            if curr_patterns and hist_patterns:
                matching_patterns = curr_patterns.intersection(hist_patterns)
                
                # Calculate pattern influence
                bullish_patterns = {"golden_cross", "bullish_engulfing", "bullish_divergence"}
                bearish_patterns = {"death_cross", "bearish_engulfing", "bearish_divergence", "double_top"}
                
                for pattern in matching_patterns:
                    if (pattern in bullish_patterns and signal_direction > 0) or \
                       (pattern in bearish_patterns and signal_direction < 0):
                        pattern_match_factor += 0.05 * weight
        
        # Combine all factors with appropriate weights
        combined_adjustment = (
            success_rate_factor * 0.5 +   # Historical success has highest weight
            similarity_weighted_factor * 0.2 +
            trend_alignment_factor * 0.2 +
            pattern_match_factor * 0.1
        )
        
        # Limit adjustment to reasonable range
        return max(-0.3, min(0.3, combined_adjustment))
    
    def _get_confidence_factors(self, 
                               signal: Dict, 
                               current_context: Dict,
                               similar_contexts: List[Dict]) -> Dict:
        """
        Extract main factors that influenced confidence adjustment.
        
        Args:
            signal: Original trading signal
            current_context: Current market context
            similar_contexts: List of similar historical contexts
            
        Returns:
            Dictionary of confidence factors
        """
        factors = {}
        
        # Direction of the signal
        signal_direction = signal.get("direction", "").lower()
        
        # Success rates from similar contexts
        if similar_contexts:
            success_rates = []
            for ctx in similar_contexts:
                if "trade_outcomes" in ctx:
                    outcomes = ctx.get("trade_outcomes", [])
                    if outcomes:
                        successes = sum(1 for o in outcomes if o.get("success", False))
                        same_direction_successes = sum(1 for o in outcomes 
                                                     if o.get("direction", "").lower() == signal_direction 
                                                     and o.get("success", False))
                        
                        success_rates.append({
                            "similarity_score": ctx.get("similarity_score", 0),
                            "overall_success_rate": successes / len(outcomes),
                            "direction_success_rate": same_direction_successes / len(outcomes) if len(outcomes) > 0 else 0,
                            "trade_count": len(outcomes)
                        })
            
            if success_rates:
                factors["historical_success_rates"] = success_rates
        
        # Pattern matches
        current_patterns = [p.get("pattern") for p in current_context.get("patterns", [])]
        if current_patterns:
            factors["current_patterns"] = current_patterns
        
        # Trend alignment
        trends = {
            "current_short_term": current_context.get("aggregated_signals", {}).get("short_term_trend"),
            "current_medium_term": current_context.get("aggregated_signals", {}).get("medium_term_trend"),
        }
        
        if similar_contexts and "timeframes" in current_context:
            # Get most recent price action
            prices = {}
            for tf, data in current_context.get("timeframes", {}).items():
                if "close" in data:
                    prices[tf] = data["close"]
            
            if prices:
                factors["current_prices"] = prices
        
        # Market conditions
        market_conditions = {
            "volatility": current_context.get("aggregated_signals", {}).get("volatility"),
            "momentum": current_context.get("aggregated_signals", {}).get("momentum"),
        }
        
        if any(market_conditions.values()):
            factors["market_conditions"] = market_conditions
        
        # Support/resistance levels
        support_resistance = current_context.get("aggregated_signals", {}).get("support_resistance", [])
        if support_resistance:
            # Find closest levels
            sr_levels = []
            
            for level in support_resistance[:2]:  # Only include closest 2 levels
                sr_levels.append({
                    "type": level.get("type"),
                    "price": level.get("price"),
                    "distance_percent": level.get("distance_percent"),
                    "strength": level.get("strength")
                })
            
            if sr_levels:
                factors["key_levels"] = sr_levels
        
        return factors
    
    def _generate_analysis(self, 
                          signal: Dict, 
                          current_context: Dict,
                          similar_contexts: List[Dict]) -> str:
        """
        Generate analysis of the signal using Gemini.
        
        Args:
            signal: Original trading signal
            current_context: Current market context
            similar_contexts: List of similar historical contexts
            
        Returns:
            String containing the analysis
        """
        if not self.gemini_available:
            return ""
        
        try:
            # Prepare context for the LLM
            symbol = signal.get("symbol", "Unknown")
            signal_type = signal.get("signal_type", "Unknown")
            direction = signal.get("direction", "Unknown")
            confidence = signal.get("confidence", 0)
            original_confidence = signal.get("original_confidence", 0)
            
            # Format timeframe data
            timeframe_data = []
            for tf, data in current_context.get("timeframes", {}).items():
                if isinstance(data, dict) and "indicators" in data:
                    tf_info = {
                        "timeframe": tf,
                        "price": data.get("close"),
                        "indicators": data.get("indicators", {})
                    }
                    timeframe_data.append(tf_info)
            
            # Format similar contexts
            similar_context_data = []
            for ctx in similar_contexts[:3]:  # Limit to top 3 for clarity
                ctx_data = {
                    "similarity_score": ctx.get("similarity_score", 0),
                    "timestamp": ctx.get("timestamp", ""),
                    "trends": {
                        "short_term": ctx.get("aggregated_signals", {}).get("short_term_trend"),
                        "medium_term": ctx.get("aggregated_signals", {}).get("medium_term_trend")
                    },
                    "patterns": [p.get("pattern") for p in ctx.get("patterns", [])]
                }
                
                # Add outcomes if available
                if "trade_outcomes" in ctx:
                    outcomes = ctx.get("trade_outcomes", [])
                    if outcomes:
                        successes = sum(1 for o in outcomes if o.get("success", False))
                        success_rate = successes / len(outcomes)
                        ctx_data["trade_outcomes"] = {
                            "count": len(outcomes),
                            "success_rate": success_rate,
                            "example_outcomes": outcomes[:2]  # Include only 2 examples
                        }
                
                similar_context_data.append(ctx_data)
            
            # Prepare prompt for Gemini
            prompt = f"""
            As a trading analyst, analyze this signal and provide insights:
            
            TRADING SIGNAL:
            - Symbol: {symbol}
            - Type: {signal_type}
            - Direction: {direction}
            - Original Confidence: {original_confidence:.2f}
            - Adjusted Confidence: {confidence:.2f}
            
            CURRENT MARKET CONTEXT:
            - Patterns: {[p.get('pattern') for p in current_context.get('patterns', [])]}
            - Short-term Trend: {current_context.get('aggregated_signals', {}).get('short_term_trend')}
            - Medium-term Trend: {current_context.get('aggregated_signals', {}).get('medium_term_trend')}
            - Volatility: {current_context.get('aggregated_signals', {}).get('volatility')}
            - Momentum: {current_context.get('aggregated_signals', {}).get('momentum')}
            
            Support/Resistance Levels:
            {[{level.get('type'): level.get('price')} for level in current_context.get('aggregated_signals', {}).get('support_resistance', [])[:3]]}
            
            SIMILAR HISTORICAL CONTEXTS:
            {similar_context_data}
            
            Generate a concise, factual analysis of this trading signal. Focus on:
            1. Key reasons for the confidence adjustment
            2. Important technical factors from current market context
            3. Relevant insights from similar historical contexts
            4. Potential risks and considerations
            
            Keep your analysis under 400 words, focused strictly on objective market factors.
            """
            
            # Generate analysis with Gemini
            response = self.model.generate_content(prompt)
            
            if response:
                return response.text
            else:
                logger.warning("Gemini returned empty response")
                return ""
        except Exception as e:
            logger.error(f"Error generating analysis with Gemini: {e}")
            return ""
    
    def batch_enrich_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Enrich multiple signals, reusing market context where possible.
        
        Args:
            signals: List of trading signals to enrich
            
        Returns:
            List of enriched signals
        """
        if not signals:
            return []
        
        # Group signals by symbol to fetch market context once per symbol
        signals_by_symbol = {}
        for signal in signals:
            symbol = signal.get("symbol", "")
            if symbol:
                if symbol not in signals_by_symbol:
                    signals_by_symbol[symbol] = []
                signals_by_symbol[symbol].append(signal)
        
        enriched_signals = []
        
        # Process each symbol group
        for symbol, symbol_signals in signals_by_symbol.items():
            # Fetch market context once per symbol
            market_context = self.market_context.gather_market_context(symbol)
            
            # Enrich each signal with the same market context
            for signal in symbol_signals:
                enriched_signal = self.enrich_signal(signal, market_context)
                enriched_signals.append(enriched_signal)
        
        return enriched_signals
    
    def get_historical_signals(self, 
                              symbol: str, 
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              limit: int = 20) -> List[Dict]:
        """
        Retrieve historical signals for a symbol.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            limit: Maximum number of signals to return
            
        Returns:
            List of historical signals
        """
        symbol_safe = symbol.replace('/', '_')
        signal_files = list(self.signal_storage_path.glob(f"{symbol_safe}_*.json"))
        
        # Convert dates to datetime objects if provided
        start_dt = None
        end_dt = None
        
        if start_date:
            try:
                start_dt = datetime.datetime.fromisoformat(start_date)
            except ValueError:
                logger.warning(f"Invalid start date format: {start_date}")
        
        if end_date:
            try:
                end_dt = datetime.datetime.fromisoformat(end_date)
                # Set to end of day
                end_dt = end_dt.replace(hour=23, minute=59, second=59)
            except ValueError:
                logger.warning(f"Invalid end date format: {end_date}")
        
        # Load and filter signals
        signals = []
        
        for file_path in signal_files:
            try:
                # Parse timestamp from filename
                filename = file_path.name
                timestamp_str = filename.split('_', 1)[1].split('.')[0]
                file_dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                # Filter by date range if specified
                if (start_dt and file_dt < start_dt) or (end_dt and file_dt > end_dt):
                    continue
                
                # Load signal data
                with open(file_path, 'r') as f:
                    signal = json.load(f)
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error loading signal file {file_path}: {e}")
        
        # Sort by timestamp (newest first) and limit results
        signals.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
        return signals[:limit]