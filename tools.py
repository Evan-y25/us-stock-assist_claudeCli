"""
tools.py - 四个 Tool 的定义 + 执行逻辑
LLM 自主决定调用哪个、传什么参数

Tool 列表：
  1. web_search        - Tavily 联网搜索（新闻/实时信息）
  2. get_market_data   - yfinance 股价/技术指标/财务/空头数据
  3. get_macro_data    - FRED API 宏观经济数据
  4. get_sec_data      - SEC EDGAR 内部人士交易/机构持仓
"""

import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
# Tool Schema 定义（传给 Bedrock 的 tools 参数）
# description 写得越清晰，LLM 决策越准确
# ══════════════════════════════════════════════════════

TOOL_SCHEMAS = [
    {
        "name": "web_search",
        "description": """
            搜索互联网获取最新金融新闻、市场数据和分析报告。
            适用场景：
            - 获取最新市场新闻和事件（今日/本周）
            - 查找特定公司的最新动态、并购传言
            - 搜索宏观经济政策声明（美联储/欧央行公告）
            - 查找内部人士买入新闻（OpenInsider、SEC披露报道）
            - 获取空头挤压候选股票信息（Finviz、ShortQuote）
            - 查找机构持仓变化报道（WhaleWisdom、13F报告）
            - 任何需要"最新"、"本周"、"今天"数据的场景
            不适用：历史K线数据、标准化财务指标（用 get_market_data）
        """,
        "inputSchema": {"json": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，建议用英文以获取更多结果，如 'GME short squeeze 2025' 或 'Fed rate decision this week'"
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回结果数量，默认5，最多10",
                    "default": 5
                }
            },
            "required": ["query"]
        }}
    },
    {
        "name": "get_market_data",
        "description": """
            通过 yfinance 获取股票、ETF、指数的市场数据和技术指标。
            适用场景：
            - 需要股价、涨跌幅、成交量、市值数据
            - 计算技术指标：RSI、MACD、布林带、均线（情绪套利/空头挤压任务）
            - 获取财务数据：市盈率、自由现金流、派息比率、净债务（分红危险任务）
            - 获取空头数据：空头比例、Days to Cover（空头挤压任务）
            - 获取历史K线数据做对比分析
            支持的代码格式：
            - 美股：AAPL、SPY、QQQ
            - A股：000001.SS（上证）、399001.SZ（深证）、600519.SS（茅台）
            - ETF：GLD（黄金）、TLT（长债）、UUP（美元）
            - 指数：^VIX、^GSPC（标普500）、^TNX（10年债收益率）
        """,
        "inputSchema": {"json": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "股票/ETF/指数代码"
                },
                "data_type": {
                    "type": "string",
                    "enum": ["price_history", "technical_indicators", "financials", "short_interest"],
                    "description": (
                        "price_history=近期K线和价格数据 | "
                        "technical_indicators=RSI/MACD/布林带/均线 | "
                        "financials=PE/市值/FCF/派息比率等财务指标 | "
                        "short_interest=空头比例/Days to Cover"
                    )
                },
                "period": {
                    "type": "string",
                    "enum": ["5d", "1mo", "3mo", "6mo", "1y", "2y"],
                    "description": "数据周期，price_history和technical_indicators需要此参数，默认3mo",
                    "default": "3mo"
                }
            },
            "required": ["ticker", "data_type"]
        }}
    },
    {
        "name": "get_macro_data",
        "description": """
            通过 FRED（美联储经济数据库）获取宏观经济指标。
            适用场景：
            - 宏观面分析任务（获取CPI/利率/GDP/就业数据）
            - 关联性地图任务（获取债券收益率、美元指数历史数据）
            - 需要与历史宏观环境做对比时
            常用 series_id（直接使用这些代码）：
            - CPIAUCSL     = 美国CPI同比（通货膨胀）
            - CPILFESL     = 核心CPI（剔除食品能源）
            - FEDFUNDS     = 联邦基金利率
            - GDP          = 美国GDP（季度）
            - GDPC1        = 美国实际GDP
            - UNRATE       = 美国失业率
            - PAYEMS       = 非农就业人数
            - DGS10        = 10年期美债收益率
            - DGS2         = 2年期美债收益率
            - T10Y2Y       = 10年-2年利差（收益率曲线）
            - DTWEXBGS     = 美元指数（宽口径）
            - BAMLH0A0HYM2 = 高收益债利差（信用风险指标）
            - VIXCLS       = VIX恐慌指数
        """,
        "inputSchema": {"json": {
            "type": "object",
            "properties": {
                "series_id": {
                    "type": "string",
                    "description": "FRED 指标代码，如 CPIAUCSL、FEDFUNDS、T10Y2Y"
                },
                "start_date": {
                    "type": "string",
                    "description": "开始日期 YYYY-MM-DD，不填默认取2年前"
                },
                "end_date": {
                    "type": "string",
                    "description": "结束日期 YYYY-MM-DD，不填默认今天"
                },
                "frequency": {
                    "type": "string",
                    "enum": ["d", "w", "m", "q", "a"],
                    "description": "数据频率：d=日、w=周、m=月、q=季、a=年，默认m",
                    "default": "m"
                }
            },
            "required": ["series_id"]
        }}
    },
    {
        "name": "get_sec_data",
        "description": """
            通过 SEC EDGAR 获取美股公开披露数据。
            适用场景：
            - 内部人士买入检测：查询 Form 4（高管/董事买卖记录）
            - 机构持仓分析：查询 13F（对冲基金季度持仓）
            - 激进股东监测：查询 SC 13D/13G（持股>5%的激进投资者）
            - 并购雷达：查询 SC TO-T（要约收购）、8-K（重大事件）
            注意：SEC EDGAR 数据有1-2天延迟，非实时
            query_type 说明：
            - insider_trading    = Form 4，某只股票的内部人士交易记录
            - institutional_13f  = 13F，某只股票被哪些机构持有
            - activist_filings   = SC 13D/G，激进股东申报
            - major_events       = 8-K，公司重大事件公告
        """,
        "inputSchema": {"json": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["insider_trading", "institutional_13f", "activist_filings", "major_events"],
                    "description": "查询类型"
                },
                "ticker": {
                    "type": "string",
                    "description": "股票代码（美股），insider_trading/activist_filings/major_events 必填"
                },
                "days_back": {
                    "type": "integer",
                    "description": "查询过去N天的数据，默认30，最大90",
                    "default": 30
                },
                "min_amount": {
                    "type": "integer",
                    "description": "insider_trading专用：最小交易金额（美元），过滤小额交易，默认50000",
                    "default": 50000
                }
            },
            "required": ["query_type"]
        }}
    }
]


# ══════════════════════════════════════════════════════
# Tool 执行逻辑
# ══════════════════════════════════════════════════════

def execute_tool(tool_name: str, tool_input: dict, api_keys: dict) -> str:
    """
    统一 Tool 执行入口，返回字符串（传回给 LLM 的 tool_result）
    出错时返回错误描述而非抛异常，让 LLM 自己决定如何处理
    """
    logger.info(f"  → 执行 Tool: {tool_name}({json.dumps(tool_input, ensure_ascii=False)})")

    try:
        if tool_name == "web_search":
            return _web_search(tool_input, api_keys.get("tavily", ""))

        elif tool_name == "get_market_data":
            return _get_market_data(tool_input)

        elif tool_name == "get_macro_data":
            return _get_macro_data(tool_input, api_keys.get("fred", ""))

        elif tool_name == "get_sec_data":
            return _get_sec_data(tool_input)

        else:
            return json.dumps({"error": f"未知工具: {tool_name}"})

    except Exception as e:
        logger.warning(f"  Tool {tool_name} 执行出错: {e}")
        return json.dumps({"error": str(e), "tool": tool_name, "input": tool_input})


# ──────────────────────────────────────
# Tool 1: Tavily 联网搜索
# ──────────────────────────────────────

def _web_search(tool_input: dict, api_key: str) -> str:
    from tavily import TavilyClient

    if not api_key:
        return json.dumps({"error": "Tavily API key 未配置"})

    client = TavilyClient(api_key=api_key)
    results = client.search(
        query=tool_input["query"],
        max_results=tool_input.get("max_results", 5),
        include_answer=True,       # 返回 AI 总结答案
        include_raw_content=False  # 不返回原始 HTML，节省 token
    )

    # 精简输出，只保留 LLM 需要的字段
    simplified = {
        "answer": results.get("answer", ""),
        "results": [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],  # 限制每条摘要长度
                "published_date": r.get("published_date", "")
            }
            for r in results.get("results", [])
        ]
    }
    return json.dumps(simplified, ensure_ascii=False)


# ──────────────────────────────────────
# Tool 2: yfinance 市场数据
# ──────────────────────────────────────

def _get_market_data(tool_input: dict) -> str:
    import yfinance as yf
    import pandas as pd

    ticker = tool_input["ticker"]
    data_type = tool_input["data_type"]
    period = tool_input.get("period", "3mo")

    stock = yf.Ticker(ticker)

    if data_type == "price_history":
        hist = stock.history(period=period)
        if hist.empty:
            return json.dumps({"error": f"无法获取 {ticker} 的价格数据，请检查代码是否正确"})

        result = {
            "ticker": ticker,
            "period": period,
            "current_price": round(hist["Close"].iloc[-1], 2),
            "price_change_1d_pct": round((hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100, 2),
            "price_change_period_pct": round((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100, 2),
            "avg_volume": int(hist["Volume"].mean()),
            "high_period": round(hist["High"].max(), 2),
            "low_period": round(hist["Low"].min(), 2),
            # 最近10条K线
            "recent_ohlcv": hist.tail(10)[["Open", "High", "Low", "Close", "Volume"]].round(2).to_dict(orient="records")
        }
        return json.dumps(result, ensure_ascii=False, default=str)

    elif data_type == "technical_indicators":
        hist = stock.history(period=period)
        if hist.empty:
            return json.dumps({"error": f"无法获取 {ticker} 的价格数据"})

        # 计算技术指标（纯 pandas 实现，无需 pandas_ta）
        close = hist["Close"]

        # RSI 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi_14 = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        # Bollinger Bands (20, 2)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # SMA
        sma5 = close.rolling(5).mean()
        sma20 = close.rolling(20).mean()
        sma60 = close.rolling(60).mean()

        latest_rsi = float(rsi_14.iloc[-1]) if pd.notna(rsi_14.iloc[-1]) else 50.0
        latest_sma5 = float(sma5.iloc[-1]) if pd.notna(sma5.iloc[-1]) else 0
        latest_sma20 = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else 0
        latest_sma60 = float(sma60.iloc[-1]) if pd.notna(sma60.iloc[-1]) else 0
        latest_price = float(close.iloc[-1])

        result = {
            "ticker": ticker,
            "current_price": round(latest_price, 2),
            "rsi_14": round(latest_rsi, 2),
            "macd": round(float(macd_line.iloc[-1]), 4),
            "macd_signal": round(float(macd_signal.iloc[-1]), 4),
            "macd_histogram": round(float(macd_hist.iloc[-1]), 4),
            "bb_upper": round(float(bb_upper.iloc[-1]), 2) if pd.notna(bb_upper.iloc[-1]) else 0,
            "bb_middle": round(float(bb_mid.iloc[-1]), 2) if pd.notna(bb_mid.iloc[-1]) else 0,
            "bb_lower": round(float(bb_lower.iloc[-1]), 2) if pd.notna(bb_lower.iloc[-1]) else 0,
            "sma_5": round(latest_sma5, 2),
            "sma_20": round(latest_sma20, 2),
            "sma_60": round(latest_sma60, 2),
            "trend": (
                "强势多头" if latest_price > latest_sma5 > latest_sma20 > latest_sma60 > 0
                else "强势空头" if latest_price < latest_sma5 < latest_sma20 and latest_sma20 > 0
                else "震荡"
            ),
            "rsi_signal": (
                "超买" if latest_rsi > 70
                else "超卖" if latest_rsi < 30
                else "中性"
            )
        }
        return json.dumps(result, ensure_ascii=False, default=str)

    elif data_type == "financials":
        info = stock.info
        result = {
            "ticker": ticker,
            "company_name": info.get("longName", ""),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "free_cashflow": info.get("freeCashflow"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "return_on_equity": info.get("returnOnEquity"),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", "")
        }
        return json.dumps(result, ensure_ascii=False, default=str)

    elif data_type == "short_interest":
        info = stock.info
        result = {
            "ticker": ticker,
            "short_percent_of_float": info.get("shortPercentOfFloat"),
            "short_ratio_days_to_cover": info.get("shortRatio"),
            "shares_short": info.get("sharesShort"),
            "shares_short_prior_month": info.get("sharesShortPriorMonth"),
            "short_change_pct": (
                round((info.get("sharesShort", 0) / info.get("sharesShortPriorMonth", 1) - 1) * 100, 2)
                if info.get("sharesShortPriorMonth")
                else None
            ),
            "float_shares": info.get("floatShares"),
            "current_price": info.get("currentPrice"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "52w_high": info.get("fiftyTwoWeekHigh")
        }
        return json.dumps(result, ensure_ascii=False, default=str)

    return json.dumps({"error": f"未知 data_type: {data_type}"})


# ──────────────────────────────────────
# Tool 3: FRED 宏观数据
# ──────────────────────────────────────

def _get_macro_data(tool_input: dict, api_key: str) -> str:
    from fredapi import Fred

    if not api_key:
        return json.dumps({"error": "FRED API key 未配置，请前往 https://fred.stlouisfed.org/docs/api/api_key.html 申请免费 key"})

    fred = Fred(api_key=api_key)
    series_id = tool_input["series_id"]

    # 默认取2年数据
    start_date = tool_input.get("start_date") or (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    end_date = tool_input.get("end_date") or datetime.now().strftime("%Y-%m-%d")
    frequency = tool_input.get("frequency", "m")

    series = fred.get_series(series_id, start_date, end_date, frequency=frequency)

    # 获取指标元信息
    try:
        info = fred.get_series_info(series_id)
        title = info.get("title", series_id)
        units = info.get("units", "")
    except Exception:
        title = series_id
        units = ""

    latest = series.dropna().iloc[-1] if not series.empty else None
    prev = series.dropna().iloc[-2] if len(series.dropna()) > 1 else None

    result = {
        "series_id": series_id,
        "title": title,
        "units": units,
        "latest_value": round(float(latest), 4) if latest is not None else None,
        "latest_date": series.dropna().index[-1].strftime("%Y-%m-%d") if not series.empty else None,
        "prev_value": round(float(prev), 4) if prev is not None else None,
        "change": round(float(latest - prev), 4) if (latest is not None and prev is not None) else None,
        "change_pct": round(float((latest / prev - 1) * 100), 2) if (latest is not None and prev is not None and prev != 0) else None,
        "history": {
            str(k.strftime("%Y-%m")): round(float(v), 4)
            for k, v in series.dropna().tail(24).items()  # 最近24期
        }
    }
    return json.dumps(result, ensure_ascii=False, default=str)


# ──────────────────────────────────────
# Tool 4: SEC EDGAR 数据
# ──────────────────────────────────────

def _get_sec_data(tool_input: dict) -> str:
    import requests

    query_type = tool_input["query_type"]
    ticker = tool_input.get("ticker", "").upper()
    days_back = min(tool_input.get("days_back", 30), 90)
    min_amount = tool_input.get("min_amount", 50000)

    # SEC 要求 User-Agent 包含联系方式
    headers = {
        "User-Agent": "QuantAnalyst contact@example.com",
        "Accept-Encoding": "gzip, deflate"
    }

    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    if query_type == "insider_trading":
        if not ticker:
            return json.dumps({"error": "insider_trading 需要提供 ticker"})

        # 使用 SEC EDGAR 全文搜索 API
        url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&dateRange=custom&startdt={start_date}&forms=4"
            f"&hits.hits._source=period_of_report,display_names,file_date,form_type"
        )
        resp = requests.get(url, headers=headers, timeout=15)
        data = resp.json()

        filings = []
        for hit in data.get("hits", {}).get("hits", [])[:10]:
            source = hit.get("_source", {})
            filings.append({
                "form_type": source.get("form_type"),
                "file_date": source.get("file_date"),
                "period": source.get("period_of_report"),
                "filer": source.get("display_names", ""),
                "edgar_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type=4&dateb=&owner=include&count=10"
            })

        return json.dumps({
            "ticker": ticker,
            "query_type": "insider_trading",
            "period": f"过去{days_back}天",
            "total_filings": data.get("hits", {}).get("total", {}).get("value", 0),
            "filings": filings,
            "note": "Form 4 详细数据请访问 https://www.openinsider.com 查看"
        }, ensure_ascii=False)

    elif query_type == "institutional_13f":
        # 13F 机构持仓查询
        url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&forms=13F-HR&dateRange=custom&startdt={start_date}"
        )
        resp = requests.get(url, headers=headers, timeout=15)
        data = resp.json()

        institutions = []
        for hit in data.get("hits", {}).get("hits", [])[:15]:
            source = hit.get("_source", {})
            institutions.append({
                "institution": source.get("display_names", ""),
                "file_date": source.get("file_date"),
                "period": source.get("period_of_report"),
            })

        return json.dumps({
            "ticker": ticker,
            "query_type": "institutional_13f",
            "total_institutions": data.get("hits", {}).get("total", {}).get("value", 0),
            "recent_filers": institutions,
            "note": "详细持仓数据请访问 https://whalewisdom.com 或 https://www.dataroma.com"
        }, ensure_ascii=False)

    elif query_type == "activist_filings":
        url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&forms=SC+13D,SC+13G&dateRange=custom&startdt={start_date}"
        )
        resp = requests.get(url, headers=headers, timeout=15)
        data = resp.json()

        filings = []
        for hit in data.get("hits", {}).get("hits", [])[:10]:
            source = hit.get("_source", {})
            filings.append({
                "form": source.get("form_type"),
                "filer": source.get("display_names", ""),
                "file_date": source.get("file_date"),
                "period": source.get("period_of_report"),
            })

        return json.dumps({
            "ticker": ticker,
            "query_type": "activist_filings",
            "total": data.get("hits", {}).get("total", {}).get("value", 0),
            "filings": filings
        }, ensure_ascii=False)

    elif query_type == "major_events":
        url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&forms=8-K&dateRange=custom&startdt={start_date}"
        )
        resp = requests.get(url, headers=headers, timeout=15)
        data = resp.json()

        events = []
        for hit in data.get("hits", {}).get("hits", [])[:10]:
            source = hit.get("_source", {})
            events.append({
                "file_date": source.get("file_date"),
                "filer": source.get("display_names", ""),
                "items": source.get("items", "")
            })

        return json.dumps({
            "ticker": ticker,
            "query_type": "major_events_8K",
            "total": data.get("hits", {}).get("total", {}).get("value", 0),
            "events": events
        }, ensure_ascii=False)

    return json.dumps({"error": f"未知 query_type: {query_type}"})
