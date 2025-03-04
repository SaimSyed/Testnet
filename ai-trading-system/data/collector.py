import ccxt
import pandas as pd
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_collector')

class MarketDataCollector:
    def __init__(self, exchange_id='binance', testnet=True):
        self.exchange_id = exchange_id
        
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'
                }
            })
            
            if testnet and hasattr(self.exchange, 'set_sandbox_mode'):
                self.exchange.set_sandbox_mode(True)
                logger.info(f"Connected to {exchange_id} testnet")
            else:
                logger.info(f"Connected to {exchange_id} live market")
                
            self.exchange.load_markets()
            logger.info(f"Loaded {len(self.exchange.markets)} markets")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    def get_available_symbols(self, quote_currency='USDT'):
        symbols = []
        for symbol in self.exchange.markets:
            if symbol.endswith(f'/{quote_currency}'):
                symbols.append(symbol)
        
        return symbols
    
    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} {timeframe} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_current_price(self, symbol):
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            logger.info(f"Current price for {symbol}: {ticker['last']}")
            return ticker['last']
        except Exception as e:
            logger.error(f"Failed to fetch current price for {symbol}: {e}")
            return None
    
    def fetch_multi_timeframe_data(self, symbol, timeframes=['15m', '1h', '4h'], limit=100):
        result = {}
        for tf in timeframes:
            result[tf] = self.fetch_ohlcv(symbol, tf, limit)
            time.sleep(self.exchange.rateLimit / 1000)
        
        return result

if __name__ == "__main__":
    collector = MarketDataCollector(exchange_id='binance', testnet=True)
    
    symbols = collector.get_available_symbols()
    print(f"Available symbols: {symbols[:5]}...")
    
    if symbols:
        test_symbol = symbols[0]
        print(f"Testing with symbol: {test_symbol}")
        
        df = collector.fetch_ohlcv(test_symbol, timeframe='1h', limit=10)
        print("\nOHLCV Data:")
        print(df.head())
        
        price = collector.fetch_current_price(test_symbol)
        print(f"\nCurrent price for {test_symbol}: {price}")
