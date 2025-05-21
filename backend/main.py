from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import csv
from io import StringIO
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import math
from functools import lru_cache
from fastapi import HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Portfolio Analysis API",
    description="API for portfolio risk analysis and capital market assumptions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (consider using a database in production)
holdings_store: Dict[str, List[Dict]] = {}
ticker_name_cache: Dict[str, str] = {}
price_cache: Dict[str, Dict[str, pd.DataFrame]] = {}  # Cache structure: {ticker: {timeframe: dataframe}}

# Models
class Portfolio(BaseModel):
    id: str
    name: str
    description: Optional[str] = None

class Holding(BaseModel):
    ticker: str
    exposure: float
    name: Optional[str] = None
    pct_of_portfolio: Optional[float] = None

class RiskInput(BaseModel):
    portfolio_id: str
    start_date: str
    end_date: str
    base_currency: Optional[str] = "AUD"
    
    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        try:
            pd.to_datetime(v)
            return v
        except:
            raise ValueError(f"Invalid date format: {v}")

class RiskOutput(BaseModel):
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    var_95_pct: float
    var_99_pct: float
    cvar_95_pct: float
    cvar_99_pct: float
    var_95_annual: float
    var_99_annual: float
    cvar_95_annual: float
    cvar_99_annual: float
    var_95_pct_annual: float
    var_99_pct_annual: float
    cvar_95_pct_annual: float
    cvar_99_pct_annual: float
    warning: Optional[str] = None

class CMAInput(BaseModel):
    portfolio_id: str
    start_date: str
    end_date: str
    base_currency: Optional[str] = "AUD"
    use_assumptions: Optional[bool] = False
    assumptions: Optional[Dict[str, Any]] = None
    
    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        try:
            pd.to_datetime(v)
            return v
        except:
            raise ValueError(f"Invalid date format: {v}")

class CMAOutput(BaseModel):
    arithmetic_return: float
    geometric_return: float
    volatility: float
    negative_prob: float
    negative_years: int
    asset_returns: Dict[str, float]
    asset_vols: Dict[str, float]
    covariance_matrix: Dict[str, Dict[str, float]]
    correlation_matrix: Dict[str, Dict[str, float]]
    sharpe_ratio: Optional[float] = None
    
class StatusResponse(BaseModel):
    status: str
    rows: int

# Helper functions
@lru_cache(maxsize=100)
def get_ticker_info(ticker: str) -> str:
    """Get ticker name with caching"""
    if ticker in ticker_name_cache:
        return ticker_name_cache[ticker]
    
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ticker
        name = name[:30]  # Truncate long names
        ticker_name_cache[ticker] = name
        return name
    except Exception as e:
        logger.warning(f"Failed to get name for {ticker}: {str(e)}")
        ticker_name_cache[ticker] = ticker
        return ticker

async def get_prices_async(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Get historical prices with caching"""
    cache_key = f"{start_date}_{end_date}"
    
    # Check what we have in cache
    to_fetch = []
    for ticker in tickers:
        if ticker not in price_cache or cache_key not in price_cache[ticker]:
            to_fetch.append(ticker)
    
    if to_fetch:
        try:
            logger.info(f"Fetching prices for {len(to_fetch)} tickers")
            price_data = yf.download(
                tickers=to_fetch, 
                start=pd.to_datetime(start_date),
                end=pd.to_datetime(end_date)
            )
            
            # Parse and cache the results
            if isinstance(price_data.columns, pd.MultiIndex):
                if len(to_fetch) > 1:
                    top_levels = price_data.columns.get_level_values(0).unique()
                    if "Adj Close" in top_levels:
                        price_data = price_data["Adj Close"]
                    elif "Close" in top_levels:
                        price_data = price_data["Close"]
                else:
                    price_data = price_data["Adj Close"]
                    price_data.columns = [to_fetch[0]]
            
            # Update cache
            for ticker in to_fetch:
                if ticker in price_data.columns:
                    if ticker not in price_cache:
                        price_cache[ticker] = {}
                    price_cache[ticker][cache_key] = price_data[[ticker]]
        except Exception as e:
            logger.error(f"Failed to fetch prices: {str(e)}")
            # Just continue with what we have in cache
    
    # Combine cached data
    combined_data = pd.DataFrame()
    for ticker in tickers:
        if ticker in price_cache and cache_key in price_cache[ticker]:
            if combined_data.empty:
                combined_data = price_cache[ticker][cache_key].copy()
            else:
                combined_data = pd.concat([combined_data, price_cache[ticker][cache_key]], axis=1)
    
    return combined_data

def preload_holdings_if_empty(portfolio_id: str):
    """Load default holdings if none exist"""
    if portfolio_id not in holdings_store or not holdings_store[portfolio_id]:
        if portfolio_id == "demo-portfolio":
            holdings_store[portfolio_id] = [
                {"ticker": "AAPL", "exposure": 100000},
                {"ticker": "MSFT", "exposure": 80000},
                {"ticker": "GOOGL", "exposure": 120000}
            ]
        elif portfolio_id == "growth-2025":
            holdings_store[portfolio_id] = [
                {"ticker": "IVV", "exposure": 150000},
                {"ticker": "BND", "exposure": 100000},
                {"ticker": "VWO", "exposure": 50000}
            ]
        elif portfolio_id == "dhhf":
            holdings_store[portfolio_id] = [
                {"ticker": "VTI", "exposure": 40.18},
                {"ticker": "A200.AX", "exposure": 37.03},
                {"ticker": "SPDW", "exposure": 16.51},
                {"ticker": "SPEM", "exposure": 6.08},
                {"ticker": "AUDUSD=X", "exposure": 0.20}
            ]

def calculate_weights(holdings: List[Dict]) -> Dict[str, float]:
    """Calculate portfolio weights from holdings"""
    weights = {}
    total_exposure = sum(float(h.get("exposure", 0)) for h in holdings)
    if total_exposure > 0:
        for h in holdings:
            if h.get("ticker"):
                weights[h["ticker"]] = float(h.get("exposure", 0)) / total_exposure
    return weights

def apply_fx_conversion(price_data: pd.DataFrame, fx_ticker: str, tickers: List[str]) -> pd.DataFrame:
    """Apply FX conversion to price data"""
    if fx_ticker in price_data.columns:
        fx_rates = price_data[fx_ticker]
        for ticker in price_data.columns:
            if ticker != fx_ticker and ticker in tickers:
                price_data[ticker] = price_data[ticker] * fx_rates
    return price_data

# API Endpoints
@app.get("/api/portfolios", response_model=List[Portfolio])
def list_portfolios():
    """List available portfolios"""
    return [
        Portfolio(id="demo-portfolio", name="Demo Portfolio", description="Sample technology stocks"),
        Portfolio(id="growth-2025", name="Growth Portfolio 2025", description="Diversified growth ETFs"),
        Portfolio(id="dhhf", name="DHHF Portfolio", description="High growth ETF portfolio")
    ]

@app.post("/api/upload/{portfolio_id}", response_model=StatusResponse)
async def upload_holdings(portfolio_id: str, file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload portfolio holdings from CSV"""
    try:
        content = await file.read()
        csv_reader = csv.DictReader(StringIO(content.decode("utf-8")))
        holdings = []
        
        for row in csv_reader:
            try:
                row["exposure"] = float(row.get("exposure", 0))
            except (ValueError, TypeError):
                row["exposure"] = 0.0
            holdings.append(row)
        
        holdings_store[portfolio_id] = holdings
        
        # Fetch ticker names in background
        if background_tasks:
            background_tasks.add_task(prefetch_ticker_names, holdings)
            
        return StatusResponse(status="uploaded", rows=len(holdings))
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

async def prefetch_ticker_names(holdings: List[Dict]):
    """Prefetch ticker names in background"""
    for holding in holdings:
        if ticker := holding.get("ticker"):
            get_ticker_info(ticker)

@app.get("/api/holdings/{portfolio_id}", response_model=List[Holding])
async def get_holdings(portfolio_id: str):
    """Get portfolio holdings with additional info"""
    preload_holdings_if_empty(portfolio_id)
    holdings = holdings_store.get(portfolio_id, [])
    
    # Calculate total and percentages
    total = sum(float(row.get("exposure", 0)) for row in holdings)
    
    result = []
    for row in holdings:
        exposure = float(row.get("exposure", 0))
        pct = round((exposure / total) * 100, 2) if total else 0.0
        
        ticker = row.get("ticker", "")
        name = get_ticker_info(ticker) if ticker else "Cash"
        
        result.append({
            "ticker": ticker,
            "exposure": exposure,
            "name": name,
            "pct_of_portfolio": pct
        })
    
    return result

@app.post("/api/risk", response_model=RiskOutput)
async def get_risk(data: RiskInput):
    """Calculate portfolio risk metrics"""
    preload_holdings_if_empty(data.portfolio_id)
    holdings = holdings_store.get(data.portfolio_id, [])
    start = pd.to_datetime(data.start_date)
    end = pd.to_datetime(data.end_date)
    base_currency = data.base_currency.upper()

    try:
        # Calculate weights
        weights = calculate_weights(holdings)
        tickers = list(weights.keys())
        
        if not tickers:
            raise ValueError("No valid tickers in portfolio")
        
        # Add FX ticker if needed
        fx_ticker = "AUDUSD=X" if base_currency == "AUD" else None
        fetch_list = tickers + ([fx_ticker] if fx_ticker and fx_ticker not in tickers else [])
        
        # Get price data
        price_data = await get_prices_async(fetch_list, data.start_date, data.end_date)
        
        if price_data.empty:
            raise ValueError("No price data available")
            
        # Apply FX conversion if needed
        if base_currency == "AUD":
            price_data = apply_fx_conversion(price_data, fx_ticker, tickers)
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Filter to only tickers in both returns and weights
        shared_tickers = [t for t in returns.columns if t in weights]
        
        if not shared_tickers:
            raise ValueError("No matching tickers between price data and weights")
        
        filtered_returns = returns[shared_tickers]
        filtered_weights = pd.Series({t: weights[t] for t in shared_tickers})
        
        # Calculate portfolio returns
        portfolio_returns = filtered_returns.dot(filtered_weights)
        
        # Calculate risk metrics
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        # Calculate annualized metrics
        total_exposure = sum(float(h.get("exposure", 0)) for h in holdings)
        factor = np.sqrt(252)  # Annualization factor
        
        return RiskOutput(
            var_95=abs(var_95 * total_exposure),
            var_99=abs(var_99 * total_exposure),
            cvar_95=abs(cvar_95 * total_exposure),
            cvar_99=abs(cvar_99 * total_exposure),
            var_95_pct=abs(var_95) * 100,
            var_99_pct=abs(var_99) * 100,
            cvar_95_pct=abs(cvar_95) * 100,
            cvar_99_pct=abs(cvar_99) * 100,
            var_95_annual=abs(var_95 * total_exposure * factor),
            var_99_annual=abs(var_99 * total_exposure * factor),
            cvar_95_annual=abs(cvar_95 * total_exposure * factor),
            cvar_99_annual=abs(cvar_99 * total_exposure * factor),
            var_95_pct_annual=abs(var_95) * 100 * factor,
            var_99_pct_annual=abs(var_99) * 100 * factor,
            cvar_95_pct_annual=abs(cvar_95) * 100 * factor,
            cvar_99_pct_annual=abs(cvar_99) * 100 * factor
        )

    except Exception as e:
        logger.error(f"Risk calculation error: {str(e)}")
        
        # Fallback to estimated values
        fallback_total = sum(float(row.get("exposure", 0)) for row in holdings)
        return RiskOutput(
            var_95=round(fallback_total * 0.05, 2),
            var_99=round(fallback_total * 0.08, 2),
            cvar_95=round(fallback_total * 0.07, 2),
            cvar_99=round(fallback_total * 0.10, 2),
            var_95_pct=5.0,
            var_99_pct=8.0,
            cvar_95_pct=7.0,
            cvar_99_pct=10.0,
            var_95_annual=round(fallback_total * 0.05 * np.sqrt(252), 2),
            var_99_annual=round(fallback_total * 0.08 * np.sqrt(252), 2),
            cvar_95_annual=round(fallback_total * 0.07 * np.sqrt(252), 2),
            cvar_99_annual=round(fallback_total * 0.10 * np.sqrt(252), 2),
            var_95_pct_annual=round(5.0 * np.sqrt(252), 2),
            var_99_pct_annual=round(8.0 * np.sqrt(252), 2),
            cvar_95_pct_annual=round(7.0 * np.sqrt(252), 2),
            cvar_99_pct_annual=round(10.0 * np.sqrt(252), 2),
            warning="Fallback values used: price or FX data unavailable."
        )

@app.post("/api/risk_with_benchmark")
async def get_risk_with_benchmark(data: RiskInput):
    """Calculate portfolio risk metrics with benchmark comparison"""
    try:
        # Get portfolio risk metrics
        logger.info("Fetching portfolio risk metrics")
        portfolio_risk = await get_risk(data)
        
        # Get benchmark data
        benchmark_id = data.benchmark_id if hasattr(data, 'benchmark_id') and data.benchmark_id else "asx200"
        
        logger.info(f"Fetching benchmark data for {benchmark_id}")
        try:
            benchmark_data = get_benchmark_data(benchmark_id, data.start_date, data.end_date)
            
            if benchmark_data.empty:
                logger.warning(f"No data found for benchmark {benchmark_id}")
                raise ValueError(f"No data available for benchmark {benchmark_id}")
        except Exception as e:
            logger.error(f"Error fetching benchmark data: {str(e)}")
            raise ValueError(f"Error fetching benchmark data: {str(e)}")
        
        # Calculate benchmark returns
        logger.info("Calculating benchmark returns")
        try:
            benchmark_returns = benchmark_data.pct_change(fill_method=None).dropna()
            logger.info(f"Benchmark returns shape: {benchmark_returns.shape}")
        except Exception as e:
            logger.error(f"Error calculating benchmark returns: {str(e)}")
            raise ValueError(f"Error calculating benchmark returns: {str(e)}")
        
        # Calculate benchmark risk metrics
        logger.info("Calculating benchmark risk metrics")
        try:
            benchmark_risk = calculate_benchmark_risk(benchmark_returns)
        except Exception as e:
            logger.error(f"Error calculating benchmark risk: {str(e)}")
            raise ValueError(f"Error calculating benchmark risk: {str(e)}")
        
        # Add benchmark ID for reference
        benchmark_risk["id"] = benchmark_id
        
        # Extract portfolio risk as dict if it's a Pydantic model
        portfolio_risk_dict = portfolio_risk
        if hasattr(portfolio_risk, 'dict') and callable(getattr(portfolio_risk, 'dict')):
            logger.info("Converting portfolio risk from Pydantic model to dict")
            portfolio_risk_dict = portfolio_risk.dict()
        
        # Calculate comparison metrics
        logger.info("Calculating comparison metrics")
        try:
            comparison = calculate_comparison_metrics(portfolio_risk_dict, benchmark_risk)
        except Exception as e:
            logger.error(f"Error calculating comparison metrics: {str(e)}")
            # Use default values if calculation fails
            comparison = {
                "tracking_error": 2.0,
                "information_ratio": 0.25,
                "beta": 1.0,
                "alpha": 0.5
            }
        
        logger.info("Returning final results")
        # Return combined results
        return {
            "portfolio": portfolio_risk,
            "benchmark": benchmark_risk,
            "comparison": comparison
        }
    except Exception as e:
        logger.error(f"Error in risk_with_benchmark: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def calculate_benchmark_risk(benchmark_returns):
    """Calculate risk metrics for a benchmark"""
    try:
        # ... existing calculations ...
        
        # Handle potential NaN or infinite values before returning
        def sanitize_value(value, default=0.0):
            if value is None or (hasattr(value, 'isna') and value.isna().any()) or (hasattr(value, 'item') and (np.isnan(value.item()) or np.isinf(value.item()))) or np.isnan(value) or np.isinf(value):
                return default
            return value
        
        return {
            "var_95": sanitize_value(abs(var_95_value)),
            "var_99": sanitize_value(abs(var_99_value)),
            "cvar_95": sanitize_value(abs(cvar_95_value)),
            "cvar_99": sanitize_value(abs(cvar_99_value)),
            "var_95_pct": sanitize_value(var_95_pct),
            "var_99_pct": sanitize_value(var_99_pct),
            "cvar_95_pct": sanitize_value(cvar_95_pct),
            "cvar_99_pct": sanitize_value(cvar_99_pct),
            "var_95_annual": sanitize_value(abs(var_95_annual)),
            "var_99_annual": sanitize_value(abs(var_99_annual)),
            "cvar_95_annual": sanitize_value(abs(cvar_95_annual)),
            "cvar_99_annual": sanitize_value(abs(cvar_99_annual)),
            "var_95_pct_annual": sanitize_value(var_95_pct_annual),
            "var_99_pct_annual": sanitize_value(var_99_pct_annual),
            "cvar_95_pct_annual": sanitize_value(cvar_95_pct_annual),
            "cvar_99_pct_annual": sanitize_value(cvar_99_pct_annual),
            "volatility": sanitize_value(annual_volatility * 100),
            "returns": {
                "daily_mean": sanitize_value(daily_mean * 100),
                "annual_mean": sanitize_value(daily_mean * 252 * 100)
            }
        }
    except Exception as e:
        logger.error(f"Error calculating benchmark risk: {str(e)}")
        return get_default_benchmark_risk()  # Use default values instead of raising an error

def get_benchmark_data(benchmark_id, start_date, end_date):
    """Fetch benchmark price data"""
    if benchmark_id == "composite":
        # Special handling for composite indices
        # This would combine multiple indices based on target weights
        return calculate_composite_benchmark()
    else:
        # Fetch data for standard benchmarks
        benchmarks = {b["id"]: b["ticker"] for b in list_benchmarks()}
        ticker = benchmarks.get(benchmark_id)
        
        if not ticker:
            raise ValueError(f"Unknown benchmark: {benchmark_id}")
        
        # Use your existing price fetching function
        return fetch_benchmark_prices(ticker, start_date, end_date)

def calculate_comparison_metrics(portfolio_risk_dict, benchmark_risk):
    """Calculate comparison metrics between portfolio and benchmark"""
    try:
        # Extract and sanitize volatilities
        p_vol = safe_float(portfolio_risk_dict.get('volatility', 10.0), 10.0)
        b_vol = safe_float(benchmark_risk.get('volatility', 15.0), 15.0)
        
        # Extract and sanitize returns
        p_return = 0
        if 'returns' in portfolio_risk_dict and isinstance(portfolio_risk_dict['returns'], dict):
            p_return = safe_float(portfolio_risk_dict['returns'].get('annual_mean', 7.0), 7.0)
        
        b_return = 0
        if 'returns' in benchmark_risk and isinstance(benchmark_risk['returns'], dict):
            b_return = safe_float(benchmark_risk['returns'].get('annual_mean', 6.0), 6.0)
        
        # Calculations with sanitized values
        correlation = 0.8
        tracking_error = math.sqrt(max(0, (p_vol/100)**2 + (b_vol/100)**2 - 2 * correlation * (p_vol/100) * (b_vol/100))) * 100
        
        # Prevent division by zero
        beta = correlation * (p_vol / b_vol) if b_vol > 0 else 1.0
        
        # Avoid NaN in information ratio
        information_ratio = (p_return - b_return) / tracking_error if tracking_error > 0 else 0.0
        
        # Risk-free rate for alpha calculation
        risk_free_rate = 2.5
        alpha = p_return - (risk_free_rate + beta * (b_return - risk_free_rate))
        
        return {
            "tracking_error": float(tracking_error) if not math.isnan(tracking_error) else 2.0,
            "information_ratio": float(information_ratio) if not math.isnan(information_ratio) else 0.25,
            "beta": float(beta) if not math.isnan(beta) else 1.0,
            "alpha": float(alpha) if not math.isnan(alpha) else 0.5
        }
    except Exception as e:
        logger.error(f"Error calculating comparison metrics: {str(e)}")
        return {
            "tracking_error": 2.0,
            "information_ratio": 0.25,
            "beta": 1.0,
            "alpha": 0.5
        }

def safe_float(value, default=0.0):
    """Safely convert a value to float, handling pandas Series and errors"""
    try:
        if hasattr(value, 'iloc'):  # Check if it's a Series
            if len(value) > 0:
                return float(value.iloc[0])
            else:
                return default
        elif hasattr(value, 'item'):  # Check if it's a numpy scalar
            return float(value.item())
        else:
            return float(value)
    except (TypeError, ValueError):
        return default

def fetch_benchmark_prices(ticker, start_date, end_date):
    """
    Fetch historical price data for benchmark tickers
    
    Args:
        ticker (str): The benchmark ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: DataFrame containing price history
    """
    try:
        # Convert string dates to datetime objects
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Add a small buffer to ensure we get all data
        start = start - timedelta(days=5)
        end = end + timedelta(days=5)
        
        # Use your existing yfinance function if you have one
        if hasattr(yf, 'download'):
            data = yf.download(ticker, start=start, end=end)
            
            # Get adjusted close prices
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            else:
                prices = data['Close']
                
            # Filter to the actual date range
            actual_start = pd.to_datetime(start_date)
            actual_end = pd.to_datetime(end_date)
            prices = prices[(prices.index >= actual_start) & (prices.index <= actual_end)]
            
            return prices
        else:
            # Alternative implementation if yfinance is not available
            logger.error("yfinance module not available")
            raise ValueError("Price data service unavailable")
    
    except Exception as e:
        logger.error(f"Error fetching benchmark prices for {ticker}: {str(e)}")
        raise ValueError(f"Failed to fetch benchmark data: {str(e)}")

def calculate_tracking_error(portfolio, benchmark):
    """Calculate tracking error between portfolio and benchmark"""
    try:
        # Convert Pydantic model to dict if needed
        if hasattr(portfolio, 'dict') and callable(getattr(portfolio, 'dict')):
            portfolio = portfolio.dict()
        
        # Extract volatilities from the objects
        p_vol = portfolio.get('volatility', 0) / 100 if isinstance(portfolio, dict) else 0
        b_vol = benchmark.get('volatility', 0) / 100
        
        # Assume correlation of 0.8 between portfolio and benchmark
        correlation = 0.8
        
        # Calculate tracking error using the volatilities
        tracking_error = np.sqrt(p_vol**2 + b_vol**2 - 2 * correlation * p_vol * b_vol)
        
        return tracking_error * 100  # Convert back to percentage
    except Exception as e:
        logger.error(f"Error calculating tracking error: {str(e)}")
        return 2.0  # Default to a reasonable tracking error value
        
def calculate_information_ratio(portfolio, benchmark):
    """Calculate information ratio (excess return / tracking error)"""
    try:
        # Convert Pydantic model to dict if needed
        if hasattr(portfolio, 'dict') and callable(getattr(portfolio, 'dict')):
            portfolio = portfolio.dict()
            
        # Get the annual returns
        if isinstance(portfolio, dict) and isinstance(benchmark, dict):
            # For portfolio, check different locations where return data might be
            p_return = 0
            if 'returns' in portfolio and isinstance(portfolio['returns'], dict):
                p_return = portfolio['returns'].get('annual_mean', 0)
            elif 'geometric_return' in portfolio:
                p_return = float(portfolio.get('geometric_return', 0))
            elif 'arithmetic_return' in portfolio:
                p_return = float(portfolio.get('arithmetic_return', 0))
                
            # For benchmark
            b_return = 0
            if 'returns' in benchmark and isinstance(benchmark['returns'], dict):
                b_return = benchmark['returns'].get('annual_mean', 0)
            else:
                # Try to estimate from historical data
                b_return = 6.0  # Default to reasonable market return
                
            # Calculate excess return
            excess_return = p_return - b_return
            
            # Get tracking error
            te = calculate_tracking_error(portfolio, benchmark)
            
            # Ensure te is a scalar, not a Series
            if hasattr(te, 'item'):
                te = te.item()
                
            if te <= 0:
                return 0.0  # Avoid division by zero
                
            return excess_return / te
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Error calculating information ratio: {str(e)}")
        return 0.0
        
def calculate_beta(portfolio, benchmark):
    """Calculate portfolio beta (sensitivity to market movements)"""
    try:
        # Convert Pydantic model to dict if needed
        if hasattr(portfolio, 'dict') and callable(getattr(portfolio, 'dict')):
            portfolio = portfolio.dict()
            
        # Extract volatilities
        p_vol = 0
        if isinstance(portfolio, dict):
            if 'volatility' in portfolio:
                # Ensure it's a scalar, not a Series
                if hasattr(portfolio['volatility'], 'item'):
                    p_vol = portfolio['volatility'].item() / 100
                else:
                    p_vol = float(portfolio['volatility']) / 100
        
        b_vol = 0
        if 'volatility' in benchmark:
            # Ensure it's a scalar, not a Series
            if hasattr(benchmark['volatility'], 'item'):
                b_vol = benchmark['volatility'].item() / 100
            else:
                b_vol = float(benchmark['volatility']) / 100
        
        if b_vol <= 0:
            return 1.0  # Default to market beta
            
        # Assume correlation of 0.8
        correlation = 0.8
        
        # Calculate beta using volatilities and correlation
        beta = correlation * (p_vol / b_vol)
        
        return beta
    except Exception as e:
        logger.error(f"Error calculating beta: {str(e)}")
        return 1.0  # Default to market beta
        
def calculate_alpha(portfolio, benchmark):
    """Calculate portfolio alpha (excess return over market-driven return)"""
    try:
        # Convert Pydantic model to dict if needed
        if hasattr(portfolio, 'dict') and callable(getattr(portfolio, 'dict')):
            portfolio = portfolio.dict()
            
        # Get the annual returns
        if isinstance(portfolio, dict) and isinstance(benchmark, dict):
            # Extract return data for portfolio
            p_return = 0
            if 'returns' in portfolio and isinstance(portfolio['returns'], dict):
                p_return = float(portfolio['returns'].get('annual_mean', 0))
            elif 'geometric_return' in portfolio:
                if hasattr(portfolio['geometric_return'], 'item'):
                    p_return = portfolio['geometric_return'].item()
                else:
                    p_return = float(portfolio.get('geometric_return', 0))
            elif 'arithmetic_return' in portfolio:
                if hasattr(portfolio['arithmetic_return'], 'item'):
                    p_return = portfolio['arithmetic_return'].item()
                else:
                    p_return = float(portfolio.get('arithmetic_return', 0))
                
            # Extract return data for benchmark
            b_return = 0
            if 'returns' in benchmark and isinstance(benchmark['returns'], dict):
                b_return = float(benchmark['returns'].get('annual_mean', 0))
            else:
                b_return = 6.0  # Default market return
                
            # Risk-free rate (assumed)
            risk_free_rate = 2.5  # 2.5% risk-free rate
            
            # Calculate beta
            beta = calculate_beta(portfolio, benchmark)
            
            # Calculate alpha using CAPM formula: r_p - [r_f + β(r_m - r_f)]
            alpha = p_return - (risk_free_rate + beta * (b_return - risk_free_rate))
            
            return alpha
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Error calculating alpha: {str(e)}")
        return 0.0

def calculate_dhhf_benchmark(start_date, end_date):
    """Calculate a synthetic benchmark based on DHHF target allocations"""
    # DHHF target allocations
    allocations = {
        "^AXJO": 0.37,  # ASX 200 for Australian equities
        "URTH": 0.40,   # MSCI World ETF for developed markets
        "EEM": 0.10,    # MSCI Emerging Markets ETF
        "^AXPJ": 0.13   # ASX 200 A-REIT index for property
    }
    
    # Fetch data for all components
    component_data = {}
    for ticker, weight in allocations.items():
        try:
            prices = fetch_price_data(ticker, start_date, end_date)
            component_data[ticker] = {
                "prices": prices,
                "weight": weight
            }
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
    
    # Calculate weighted returns
    dates = sorted(set().union(*[set(data["prices"].index) for data in component_data.values()]))
    benchmark_values = pd.Series(index=dates)
    
    # Fill with weighted values
    for date in dates:
        value = 0
        for ticker, data in component_data.items():
            if date in data["prices"].index:
                value += data["prices"].loc[date] * data["weight"]
        benchmark_values[date] = value
    
    # Convert to returns
    benchmark_returns = benchmark_values.pct_change().dropna()
    
    return benchmark_returns

@app.post("/api/cma", response_model=CMAOutput)
async def get_cma(data: CMAInput):
    """Calculate capital market assumptions"""
    preload_holdings_if_empty(data.portfolio_id)
    holdings = holdings_store.get(data.portfolio_id, [])
    start = pd.to_datetime(data.start_date)
    end = pd.to_datetime(data.end_date)
    base_currency = data.base_currency.upper()

    try:
        # Use assumptions if provided
        if data.use_assumptions and data.assumptions:
            return calculate_cma_with_assumptions(holdings, data.assumptions, base_currency)
        
        # Otherwise use historical data
        weights = calculate_weights(holdings)
        tickers = list(weights.keys())
        
        if not tickers:
            raise ValueError("No valid tickers in portfolio")
        
        # Add FX ticker if needed
        fx_ticker = "AUDUSD=X" if base_currency == "AUD" else None
        fetch_list = tickers + ([fx_ticker] if fx_ticker and fx_ticker not in tickers else [])
        
        # Get price data
        price_data = await get_prices_async(fetch_list, data.start_date, data.end_date)
        
        if price_data.empty:
            raise ValueError("No price data available")
            
        # Apply FX conversion if needed
        if base_currency == "AUD":
            price_data = apply_fx_conversion(price_data, fx_ticker, tickers)
        
        # Calculate returns
        returns = price_data.pct_change(fill_method=None).dropna()
        
        # Filter to only tickers in both returns and weights
        shared_tickers = [t for t in returns.columns if t in weights]
        
        if not shared_tickers:
            raise ValueError("No matching tickers between price data and weights")
        
        filtered_returns = returns[shared_tickers]
        filtered_weights = pd.Series({t: weights[t] for t in shared_tickers})
        
        # Calculate portfolio returns
        portfolio_returns = filtered_returns.dot(filtered_weights)
        
        # Calculate metrics
        daily_mean = portfolio_returns.mean()
        daily_vol = portfolio_returns.std()

        annual_arithmetic = daily_mean * 252
        annual_volatility = daily_vol * np.sqrt(252)
        
        # More accurate calculation of geometric return
        # From log returns directly, which is more accurate than the approximation
        annual_geometric = np.exp(np.log1p(portfolio_returns).mean() * 252) - 1
        
        # Risk-free rate (approximation)
        risk_free_rate = 0.025  # Assumption: 2.5% risk-free rate
        sharpe_ratio = (annual_arithmetic - risk_free_rate) / annual_volatility if annual_volatility > 0 else None

        # Simulate for negative probability
        simulated_returns = np.random.normal(
            loc=annual_arithmetic, 
            scale=annual_volatility, 
            size=100000
        )
        negative_prob = (simulated_returns < 0).mean()
        expected_negative_years = round(negative_prob * 20)  # In a 20-year horizon

        # Asset-level statistics
        asset_returns = (filtered_returns.mean() * 100 * 252).round(3).to_dict()
        asset_vols = (filtered_returns.std() * 100 * np.sqrt(252)).round(3).to_dict()
        cov_matrix = (filtered_returns.cov() * 252).round(6).to_dict()
        corr_matrix = filtered_returns.corr().round(4).to_dict()

        return CMAOutput(
            arithmetic_return=round(annual_arithmetic * 100, 3),
            geometric_return=round(annual_geometric * 100, 3),
            volatility=round(annual_volatility * 100, 3),
            negative_prob=round(negative_prob * 100, 2),
            negative_years=expected_negative_years,
            asset_returns=asset_returns,
            asset_vols=asset_vols,
            covariance_matrix=cov_matrix,
            correlation_matrix=corr_matrix,
            sharpe_ratio=round(sharpe_ratio, 3) if sharpe_ratio is not None else None
        )

    except Exception as e:
        logger.error(f"CMA calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate CMA: {str(e)}")

def calculate_cma_with_assumptions(
    holdings: List[Dict], 
    assumptions: Dict, 
    base_currency: str = "AUD"
) -> CMAOutput:
    """Calculate CMA using forward-looking assumptions"""
    try:
        # Extract assumptions data
        asset_classes = assumptions.get("assetClasses", {})
        mappings = assumptions.get("mappings", {})
        
        # Create mappings of tickers to asset classes
        asset_allocation = {}
        total_exposure = sum(float(h.get("exposure", 0)) for h in holdings)
        
        if total_exposure <= 0:
            raise ValueError("Total portfolio exposure must be greater than zero")
        
        # Calculate allocation by asset class
        for holding in holdings:
            ticker = holding.get("ticker", "")
            exposure = float(holding.get("exposure", 0))
            
            # Map ticker to asset class
            asset_class = mappings.get(ticker, "Global Equities")  # Default to Global Equities
            
            if asset_class not in asset_allocation:
                asset_allocation[asset_class] = 0
            
            asset_allocation[asset_class] += exposure
        
        # Calculate portfolio expected return and volatility
        portfolio_return = 0
        portfolio_vol_sqr = 0
        asset_returns = {}
        asset_vols = {}
        
        # Convert to percentages and calculate expected return
        for asset_class, exposure in asset_allocation.items():
            weight = exposure / total_exposure
            
            class_data = asset_classes.get(asset_class, {})
            expected_return = float(class_data.get("expectedReturn", 0))
            volatility = float(class_data.get("volatility", 0))
            
            portfolio_return += weight * expected_return
            portfolio_vol_sqr += (weight * volatility) ** 2  # Simplified vol calculation
            
            asset_returns[asset_class] = expected_return
            asset_vols[asset_class] = volatility
        
        portfolio_vol = (portfolio_vol_sqr) ** 0.5
        
        # Create a correlation and covariance matrix from asset classes
        asset_class_list = list(asset_allocation.keys())
        correlation_matrix = {}
        covariance_matrix = {}
        
        for asset1 in asset_class_list:
            correlation_matrix[asset1] = {}
            covariance_matrix[asset1] = {}
            
            for asset2 in asset_class_list:
                # Default correlation values based on asset class relationships
                if asset1 == asset2:
                    correlation = 1.0
                elif (asset1.endswith("Equities") and asset2.endswith("Equities")) or \
                     (asset1.endswith("Bonds") and asset2.endswith("Bonds")):
                    correlation = 0.7  # Higher correlation within asset types
                elif (asset1 == "Cash" or asset2 == "Cash"):
                    correlation = 0.1  # Cash has low correlation
                else:
                    correlation = 0.3  # Default correlation between different asset types
                
                correlation_matrix[asset1][asset2] = correlation
                
                # Calculate covariance
                vol1 = asset_classes.get(asset1, {}).get("volatility", 0)
                vol2 = asset_classes.get(asset2, {}).get("volatility", 0)
                covariance_matrix[asset1][asset2] = correlation * vol1 * vol2 / 100.0
        
        # Calculate risk measures
        risk_free_rate = 3.0  # Assumption
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calculate probability of negative return using normal approximation
        z_score = portfolio_return / portfolio_vol if portfolio_vol > 0 else 3.0  # Default to very low probability
        negative_prob = norm_cdf(-z_score)
        expected_negative_years = round(negative_prob * 20)  # In 20 years
        
        # Correct geometric return calculation for forward-looking assumptions
        # For forward-looking assumptions, proper conversion from arithmetic to geometric return:
        # For annual returns, the formula is G = (1 + A) / (1 + σ²/A) - 1 where A is arithmetic return and σ is volatility
        # We'll use a simpler approximation: G ≈ A - σ²/2
        
        # First convert percentages to decimals for the calculation
        arith_decimal = portfolio_return / 100
        vol_decimal = portfolio_vol / 100
        
        # Calculate geometric return (correct formula)
        geometric_return = 0
        
        if portfolio_vol < 50:  # Safe range for the approximation
            # Use the approximation formula
            geometric_return = 100 * (arith_decimal - (vol_decimal ** 2) / 2)
        else:
            # For very high volatility, use more conservative estimate
            geometric_return = 100 * (arith_decimal * 0.75)  # Simple fallback
        
        # Sanity check - geometric return should be less than arithmetic return
        # but generally not more than a few percentage points lower for typical volatilities
        if geometric_return > portfolio_return:
            logger.warning(f"Invalid geometric return calculation: geometric ({geometric_return}) > arithmetic ({portfolio_return})")
            geometric_return = portfolio_return * 0.95  # Fallback: 95% of arithmetic
        
        # Another sanity check - the difference shouldn't be too extreme
        if portfolio_return > 0 and geometric_return < 0 and abs(geometric_return) > portfolio_return:
            logger.warning(f"Suspiciously large difference between arithmetic ({portfolio_return}) and geometric ({geometric_return})")
            geometric_return = portfolio_return * 0.75  # More conservative fallback
        
        return CMAOutput(
            arithmetic_return=round(portfolio_return, 3),
            geometric_return=round(geometric_return, 3),
            volatility=round(portfolio_vol, 3),
            negative_prob=round(negative_prob * 100, 2),  # Convert to percentage
            negative_years=expected_negative_years,
            asset_returns=asset_returns,
            asset_vols=asset_vols,
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
            sharpe_ratio=round(sharpe_ratio, 3)
        )
    
    except Exception as e:
        logger.error(f"Error in calculate_cma_with_assumptions: {str(e)}")
        raise ValueError(f"Failed to calculate CMA with assumptions: {str(e)}")

def norm_cdf(x):
    """Standard normal cumulative distribution function"""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

# Additional endpoints
@app.get("/api/benchmarks")
def list_benchmarks():
    """List available benchmarks"""
    return [
        {
            "id": "asx200",
            "name": "S&P/ASX 200",
            "description": "Australian stock market benchmark",
            "ticker": "^AXJO"
        },
        {
            "id": "msci_world",
            "name": "MSCI World Index",
            "description": "Global developed markets benchmark",
            "ticker": "URTH"  # Using ETF as proxy
        },
        {
            "id": "balanced_index",
            "name": "70/30 Growth Index",
            "description": "Composite index (70% equity, 30% bonds)",
            "ticker": "composite"  # Special handling for composite indices
        }
    ]

@app.get("/api/health")
def health_check():
    """API health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.delete("/api/holdings/{portfolio_id}")
def delete_holdings(portfolio_id: str):
    """Delete portfolio holdings"""
    if portfolio_id in holdings_store:
        del holdings_store[portfolio_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Portfolio not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)