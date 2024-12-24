import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set page config
st.set_page_config(page_title="Trading Strategy Analysis", layout="wide")

@st.cache_data
def fetch_data(ticker, start_date, end_date, market="A", frequency="daily", adjust=""):
    """
    获取股票历史数据，支持 A 股、港股和美股
    
    Parameters:
    -----------
    ticker : str
        股票代码
    start_date : str
        开始日期，格式：YYYYMMDD
    end_date : str
        结束日期，格式：YYYYMMDD
    market : str
        市场类型：'A' - A股, 'HK' - 港股, 'US' - 美股
    frequency : str
        数据频率：'daily', 'weekly', 'monthly'
    adjust : str
        复权类型：'qfq' - 前复权, 'hfq' - 后复权, None - 不复权
        
    Returns:
    --------
    pd.DataFrame
        包含股票历史数据的DataFrame
    """
    try:
        if market == "A":  # A 股
            # 获取 A 股的历史数据
            df = ak.stock_zh_a_hist(symbol=ticker, 
                                    period=frequency, 
                                    start_date=start_date, 
                                    end_date=end_date,
                                    adjust=adjust)
            
        elif market == "HK":  # 港股     
            df = ak.stock_hk_hist(symbol=ticker, 
                                  period=frequency, 
                                  start_date=start_date, 
                                  end_date=end_date,
                                  adjust=adjust)
            
        elif market == "US":  # 美股
            df = ak.stock_us_hist(symbol=ticker, 
                                  period=frequency,
                                  start_date=start_date, 
                                  end_date=end_date,
                                  adjust=adjust)
            
        else:
            raise ValueError("Invalid market type. Use 'A' for A-shares, 'HK' for Hong Kong stocks, or 'US' for US stocks.")

        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
             
        # 确保日期列为datetime类型
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 按日期排序
        df = df.sort_values('日期')
        
        return df
        
    except pd.errors.EmptyDataError:
        st.error(f"No data available for {ticker} in the specified date range")
        raise
    except ConnectionError:
        st.error("Network connection error. Please check your internet connection")
        raise
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        raise

def format_hk_stock_code(code):
    """Format HK stock code to 5 digits"""
    return code.zfill(5)

# Helper function to convert date to string format
def date_to_str(date):
    return date.strftime('%Y%m%d')

def preprocess_data(data):
    """
    数据预处理：获取Close列
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("输入必须是 pandas DataFrame")

    close_col = "收盘"
    return data, close_col

def implement_ma_strategy(data: pd.DataFrame, close_col, fast_period=50, slow_period=200):
    """
    实现双均线交叉策略：  
    - 快线上穿慢线时买入（金叉）  
    - 快线下穿慢线时卖出（死叉）
    """
    data['MA_Fast'] = data[close_col].rolling(window=fast_period).mean()
    data['MA_Slow'] = data[close_col].rolling(window=slow_period).mean()

    data['MA_Signal'] = 0
    data.loc[(data['MA_Fast'] > data['MA_Slow']) & 
             (data['MA_Fast'].shift(1) <= data['MA_Slow'].shift(1)), 'MA_Signal'] = 1
    data.loc[(data['MA_Fast'] < data['MA_Slow']) & 
             (data['MA_Fast'].shift(1) >= data['MA_Slow'].shift(1)), 'MA_Signal'] = -1

    data['Position'] = data['MA_Signal'].replace(0, np.nan).ffill().fillna(0)

    data['Return'] = data[close_col].pct_change()
    data['Strategy_Return'] = data['Position'].shift(1) * data['Return']

    return data

def implement_rsi_strategy(data, close_col, rsi_period=14, rsi_upper=70, rsi_lower=30):
    """
    实现RSI策略：
    - RSI高于70时卖出
    - RSI低于30时买入
    """
    delta = data[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    # 处理除零情况
    rs = gain / np.where(loss == 0, np.finfo(float).eps, loss)
    data['RSI'] = 100 - (100 / (1 + rs))

    data['RSI_Signal'] = 0
    data.loc[data['RSI'] < rsi_lower, 'RSI_Signal'] = 1
    data.loc[data['RSI'] > rsi_upper, 'RSI_Signal'] = -1

    data['Position'] = data['RSI_Signal'].replace(0, np.nan).ffill().fillna(0)

    data['Return'] = data[close_col].pct_change()
    data['Strategy_Return'] = data['Position'].shift(1) * data['Return']

    return data

def implement_macd_strategy(data, close_col, short_period=12, long_period=26, signal_period=9):
    """
    实现MACD策略：
    - MACD线上穿信号线时买入
    - MACD线下穿信号线时卖出
    """
    data['MACD'] = data[close_col].ewm(span=short_period, adjust=False).mean() - \
                   data[close_col].ewm(span=long_period, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()

    data['MACD_Signal'] = 0
    data.loc[(data['MACD'] > data['Signal_Line']) & 
             (data['MACD'].shift(1) <= data['Signal_Line'].shift(1)), 'MACD_Signal'] = 1
    data.loc[(data['MACD'] < data['Signal_Line']) & 
             (data['MACD'].shift(1) >= data['Signal_Line'].shift(1)), 'MACD_Signal'] = -1

    data['Position'] = data['MACD_Signal'].replace(0, np.nan).ffill().fillna(0)

    data['Return'] = data[close_col].pct_change()
    data['Strategy_Return'] = data['Position'].shift(1) * data['Return']

    return data

def implement_my_strategy(data, close_col, momentum_period=20):
    """
    实现动量策略：
    - 计算过去N天的回报率，选择表现最好的股票进行买入
    - 如果股票的回报率为正则买入，为负则卖出
    """
    data['Momentum'] = data[close_col].pct_change(periods=momentum_period)

    data['Momentum_Signal'] = 0
    data.loc[data['Momentum'] > 0, 'Momentum_Signal'] = 1
    data.loc[data['Momentum'] < 0, 'Momentum_Signal'] = -1

    data['Position'] = data['Momentum_Signal'].replace(0, np.nan).ffill().fillna(0)

    data['Return'] = data[close_col].pct_change()
    data['Strategy_Return'] = data['Position'].shift(1) * data['Return']

    return data

def backtest_strategy(data:pd.DataFrame, strategy_name):
    """
    回测策略：计算累计收益和最大回撤
    """
    data = data.copy()
    
    # 确保数据中包含必要的列
    required_columns = ['Return', 'Strategy_Return']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Need {required_columns}")

    data = data.dropna()

    # 计算策略和买入持有的累计收益
    data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()
    data['Buy_Hold_Return'] = (1 + data['Return']).cumprod()

    # 计算最大回撤
    data['Cumulative_Max'] = data['Cumulative_Return'].cummax()
    data['Drawdown'] = (data['Cumulative_Return'] / data['Cumulative_Max']) - 1
    max_drawdown = data['Drawdown'].min()

    # 计算其他指标
    total_return = data['Cumulative_Return'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(data)) - 1
    volatility = data['Strategy_Return'].std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0

    # 在返回时添加调试信息
    if 'Cumulative_Return' not in data.columns:
        st.error(f"Columns in data: {data.columns.tolist()}")
        raise ValueError("Cumulative_Return column not found after calculation")

    return data, max_drawdown, total_return, annual_return, sharpe_ratio

def analyze_current_signals(data: pd.DataFrame):
    """
    Analyze the current signals for all strategies.
    """
    if data.empty:
        return {}

    latest_data = data.iloc[-1]
    signals = {}

    if 'MA_Signal' in data.columns:
        signals['MA'] = 'Buy' if latest_data['MA_Signal'] == 1 else 'Sell' if latest_data['MA_Signal'] == -1 else 'Hold'

    if 'RSI' in data.columns:
        if latest_data['RSI'] < 30:
            signals['RSI'] = 'Buy (Oversold)'
        elif latest_data['RSI'] > 70:
            signals['RSI'] = 'Sell (Overbought)'
        else:
            signals['RSI'] = f'Hold (RSI: {latest_data["RSI"]:.2f})'

    if 'MACD_Signal' in data.columns:
        signals['MACD'] = 'Buy' if latest_data['MACD_Signal'] == 1 else 'Sell' if latest_data['MACD_Signal'] == -1 else 'Hold'

    if 'Momentum_Signal' in data.columns:
        signals['Momentum'] = 'Buy' if latest_data['Momentum_Signal'] == 1 else 'Sell' if latest_data['Momentum_Signal'] == -1 else 'Hold'

    return signals

# UI Components
st.title("Trading Strategy Analysis Dashboard")

col1, col2, col3, col_fq = st.columns(4)

with col1:
    market = st.selectbox("Select Market", ["A", "HK", "US"],help="A: Chinese A-Shares, HK: Hong Kong Shares, US: US Shares", index=0)
    
with col2:
    stock_code = st.text_input("Enter Stock Code")
    if market == "HK" and stock_code:
        stock_code = format_hk_stock_code(stock_code)

with col3:
    frequency = st.selectbox("Select Frequency", ["daily", "weekly", "monthly"], index=0)

# Date selection
col4, col5 = st.columns(2)
with col4:
    start_date = st.date_input(
        "Start Date",
        datetime.now() - timedelta(days=365*4)
    )
with col5:
    end_date = st.date_input(
        "End Date",
        datetime.now()
    )

with col_fq:
    adjust = st.selectbox("Adjust Type", 
                         options=[None, "qfq", "hfq"],
                         format_func=lambda x: {
                             "": "No Adjustment",
                             "qfq": "Forward",
                             "hfq": "Backward"
                         }.get(x),
                         index=1)  # 默认选择前复权

# Strategy parameters
st.subheader("Strategy Parameters")
col6, col7, col8, col9 = st.columns(4)

with col6:
    st.markdown(
        '<div title="Moving Average Strategy: Buy when the fast moving average (MA) crosses above the slow MA, sell when the fast MA crosses below the slow MA. The fast MA is a shorter-term average, while the slow MA is a longer-term average."><h5>Moving Average Parameters</h5></div>',
        unsafe_allow_html=True
    )
    ma_fast = st.number_input("Fast MA Period", value=50, min_value=1, help='The "Fast MA Period" refers to the shorter time period used to calculate the fast moving average (MA), which is more responsive to recent price changes and is used to identify short-term trends.')
    ma_slow = st.number_input("Slow MA Period", value=200, min_value=1, help='The "Slow MA Period" refers to the longer time period used to calculate the slow moving average (MA), which is less responsive to recent price changes and is used to identify long-term trends.')

with col7:
    st.markdown(
        '<div title="RSI (Relative Strength Index) Strategy: Buy when RSI falls below the lower bound (oversold), sell when RSI rises above the upper bound (overbought). The RSI period determines the lookback window for calculating RSI."><h5>RSI Parameters</h5></div>',
        unsafe_allow_html=True
    )
    rsi_period = st.number_input("RSI Period", value=14, min_value=1, help='The "RSI Period" refers to the time period used to calculate the Relative Strength Index (RSI), which measures the speed and change of price movements. A shorter period results in a more sensitive RSI, while a longer period results in a smoother RSI.')
    rsi_upper = st.number_input("RSI Upper Bound", value=70, min_value=50, max_value=90, help='The "RSI Upper Bound" is the threshold value above which the RSI is considered overbought. When the RSI exceeds this value, it may indicate a potential sell signal.')
    rsi_lower = st.number_input("RSI Lower Bound", value=30, min_value=10, max_value=50, help='The "RSI Lower Bound" is the threshold value below which the RSI is considered oversold. When the RSI falls below this value, it may indicate a potential buy signal.')

with col8:
    st.markdown(
        '<div title="MACD (Moving Average Convergence Divergence) Strategy: Buy when the MACD line crosses above the signal line, sell when the MACD line crosses below the signal line. The MACD fast, slow, and signal periods determine the lookback windows for calculating the MACD and signal lines."><h5>MACD Parameters</h5></div>',
        unsafe_allow_html=True
    )
    macd_fast = st.number_input("MACD Fast Period", value=12, min_value=1, help='The "MACD Fast Period" refers to the shorter time period used to calculate the fast Exponential Moving Average (EMA) for the Moving Average Convergence Divergence (MACD) indicator. It is used to capture short-term price movements.')
    macd_slow = st.number_input("MACD Slow Period", value=26, min_value=1, help='The "MACD Slow Period" refers to the longer time period used to calculate the slow Exponential Moving Average (EMA) for the Moving Average Convergence Divergence (MACD) indicator. It is used to capture long-term price movements.')
    macd_signal = st.number_input("MACD Signal Period", value=9, min_value=1, help='The "MACD Signal Period" refers to the time period used to calculate the signal line for the Moving Average Convergence Divergence (MACD) indicator. The signal line is typically an EMA of the MACD line.')

with col9:
    st.markdown(
        '<div title="Momentum Strategy: Buy when the price momentum is positive, sell when the price momentum is negative. The momentum period determines the lookback window for calculating the momentum."><h5>Momentum Strategy Parameters</h5></div>',
        unsafe_allow_html=True
    )
    momentum_period = st.number_input("Momentum Period", value=20, min_value=1, help='The "Momentum Period" refers to the time period used to calculate the momentum indicator, which measures the rate of price change over a specific period. A shorter period results in a more sensitive momentum indicator, while a longer period results in a smoother indicator.')

# Run Analysis button
run_analysis = st.button("Run Analysis")

if run_analysis and stock_code:
    try:
        with st.spinner('Fetching and analyzing data...'):
            # Fetch and process data
            data = fetch_data(stock_code, date_to_str(start_date), date_to_str(end_date), market, frequency, adjust)
            data, close_col = preprocess_data(data)

            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Strategy Performance", "Current Signals", "Raw Data"])

            with tab1:
                # Implement strategies
                ma_data = implement_ma_strategy(data.copy(), close_col, fast_period=ma_fast, slow_period=ma_slow)
                rsi_data = implement_rsi_strategy(data.copy(), close_col, rsi_period=rsi_period, 
                                               rsi_upper=rsi_upper, rsi_lower=rsi_lower)
                macd_data = implement_macd_strategy(data.copy(), close_col, short_period=macd_fast, 
                                                  long_period=macd_slow, signal_period=macd_signal)
                my_strategy_data = implement_my_strategy(data.copy(), close_col, momentum_period=momentum_period)

                # Run backtests
                ma_results = backtest_strategy(ma_data, "Moving Average")
                rsi_results = backtest_strategy(rsi_data, "RSI")
                macd_results = backtest_strategy(macd_data, "MACD")
                my_strategy_results = backtest_strategy(my_strategy_data, "Momentum Strategy")

                # Display results in a DataFrame
                results_df = pd.DataFrame({
                    'Strategy': ['Moving Average', 'RSI', 'MACD', 'Momentum Strategy'],
                    'Total Return': [ma_results[2], rsi_results[2], macd_results[2], my_strategy_results[2]],
                    'Annual Return': [ma_results[3], rsi_results[3], macd_results[3], my_strategy_results[3]],
                    'Max Drawdown': [ma_results[1], rsi_results[1], macd_results[1], my_strategy_results[1]],
                    'Sharpe Ratio': [ma_results[4], rsi_results[4], macd_results[4], my_strategy_results[4]]
                })

                results_df = results_df.set_index('Strategy')
                # Format percentage columns
                for col in ['Total Return', 'Annual Return', 'Max Drawdown']:
                    results_df[col] = results_df[col].apply(lambda x: f'{x:.2%}')
                # Format Sharpe Ratio
                results_df['Sharpe Ratio'] = results_df['Sharpe Ratio'].apply(lambda x: f'{x:.2f}')
                
                st.subheader("Strategy Performance Metrics")
                st.dataframe(results_df, use_container_width=True)

                # Plot strategy comparison
                cumulative_returns = pd.DataFrame({
                    'Date': ma_results[0]['日期'],
                    'Moving Average': ma_results[0]['Cumulative_Return'],
                    'RSI': rsi_results[0]['Cumulative_Return'],
                    'MACD': macd_results[0]['Cumulative_Return'],
                    'Momentum Strategy': my_strategy_results[0]['Cumulative_Return'],
                    'Buy & Hold': ma_results[0]['Buy_Hold_Return']
                })
                cumulative_returns.set_index('Date', inplace=True)

                # 绘制回撤
                drawdowns = pd.DataFrame({
                    'Date': ma_results[0]['日期'],
                    'Moving Average': ma_results[0]['Drawdown'] * 100,
                    'RSI': rsi_results[0]['Drawdown'] * 100,
                    'MACD': macd_results[0]['Drawdown'] * 100,
                    'Momentum Strategy': my_strategy_results[0]['Drawdown'] * 100
                })
                drawdowns.set_index('Date', inplace=True)

                # 使用 st.line_chart 绘制累积回报
                st.subheader(f'Strategy Cumulative Returns')
                st.line_chart(cumulative_returns)

                # 使用 st.line_chart 绘制回撤
                st.subheader('Strategy Drawdown (%)')
                st.line_chart(drawdowns)

            with tab2:
                # Analyze and display current signals
                st.subheader("Current Trading Signals")
                
                ma_signals = analyze_current_signals(ma_data)
                rsi_signals = analyze_current_signals(rsi_data)
                macd_signals = analyze_current_signals(macd_data)
                momentum_signals = analyze_current_signals(my_strategy_data)

                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MA Strategy", ma_signals.get('MA', 'N/A'))
                with col2:
                    st.metric("RSI Strategy", rsi_signals.get('RSI', 'N/A'))
                with col3:
                    st.metric("MACD Strategy", macd_signals.get('MACD', 'N/A'))
                with col4:
                    st.metric("Momentum Strategy", momentum_signals.get('Momentum', 'N/A'))

                # Display additional technical indicators
                st.subheader("Technical Indicators")
                latest_data = data.iloc[-1]
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Current Price", f"{latest_data['收盘']:.2f}")
                    if 'MA_Fast' in ma_data.columns:
                        st.metric(f"{ma_fast}-day Moving Average", f"{ma_data['MA_Fast'].iloc[-1]:.2f}")
                    if 'MA_Slow' in ma_data.columns:
                        st.metric(f"{ma_slow}-day Moving Average", f"{ma_data['MA_Slow'].iloc[-1]:.2f}")
                
                with col2:
                    if 'RSI' in rsi_data.columns:
                        st.metric("RSI", f"{rsi_data['RSI'].iloc[-1]:.2f}")
                    if 'MACD' in macd_data.columns:
                        st.metric("MACD", f"{macd_data['MACD'].iloc[-1]:.2f}")
                    if 'Momentum' in my_strategy_data.columns:
                        st.metric("Momentum", f"{my_strategy_data['Momentum'].iloc[-1]:.2%}")

            with tab3:
                st.subheader("Raw Data Preview (Latest 100 Records)")
                # 修改列标题为英文
                data.rename(columns={
                    '日期': 'Date',
                    '开盘': 'Open',
                    '收盘': 'Close',
                    '最高': 'High',
                    '最低': 'Low',
                    '成交量': 'Volume',
                    '成交额': 'Turnover',
                    '振幅': 'Amplitude',
                    '涨跌幅': 'Change %',
                    '涨跌额': 'Change',
                    '换手率': 'Turnover Rate'
                }, inplace=True)
                st.dataframe(data.tail(100).iloc[::-1], use_container_width=True)

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
else:
    if not stock_code and run_analysis:
        st.warning("Please enter a stock code")

if __name__ == "__main__":
    # This will only run when the script is run directly
    pass
