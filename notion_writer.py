"""
notion_writer.py - 将分析结果写入 Notion 数据库
"""

import logging
from datetime import datetime
from notion_client import Client
from notion_client.errors import APIResponseError

logger = logging.getLogger(__name__)


class NotionWriter:
    def __init__(self, token: str, databases: dict):
        self.client = Client(auth=token)
        self.databases = databases

    # ─────────────────────────────────────────
    # 通用写入方法
    # ─────────────────────────────────────────

    def _create_page(self, database_id: str, properties: dict, children: list = None):
        try:
            kwargs = {
                "parent": {"database_id": database_id},
                "properties": properties
            }
            if children:
                kwargs["children"] = children

            page = self.client.pages.create(**kwargs)
            logger.info(f"Notion 页面已创建: {page['url']}")
            return page
        except APIResponseError as e:
            logger.error(f"Notion API 错误: {e}")
            raise

    def _text(self, content: str):
        return {"rich_text": [{"text": {"content": str(content)[:2000]}}]}

    def _title(self, content: str):
        return {"title": [{"text": {"content": str(content)[:200]}}]}

    def _select(self, name: str):
        return {"select": {"name": str(name)[:100]}}

    def _number(self, value):
        try:
            return {"number": float(value) if value else None}
        except (TypeError, ValueError):
            return {"number": None}

    def _date(self, date_str: str):
        return {"date": {"start": date_str}}

    def _multi_select(self, names: list):
        return {"multi_select": [{"name": str(n)[:100]} for n in names[:10]]}

    def _url(self, url: str):
        return {"url": str(url)[:2000] if url else None}

    # ─────────────────────────────────────────
    # 各任务专属写入方法
    # ─────────────────────────────────────────

    def write_weekly_report(self, data: dict):
        """写入每周综合报告"""
        db_id = self.databases.get("weekly_report")
        if not db_id:
            logger.warning("未配置 weekly_report 数据库ID，跳过")
            return

        long = data.get("long_idea", {})
        short = data.get("short_idea", {})

        properties = {
            "报告标题": self._title(f"周报 {data.get('report_date', '')} W{data.get('week_number', '')}"),
            "日期": self._date(data.get("report_date", datetime.now().strftime("%Y-%m-%d"))),
            "编辑寄语": self._text(data.get("editor_note", "")),
            "做多标的": self._text(f"{long.get('ticker', '')} | {long.get('thesis', '')}"),
            "做多入场": self._text(long.get("entry_zone", "")),
            "做多止损": self._text(long.get("stop_loss", "")),
            "做多目标": self._text(long.get("target", "")),
            "做空标的": self._text(f"{short.get('ticker', '')} | {short.get('thesis', '')}"),
        }

        # 构建页面正文块
        children = self._build_weekly_report_blocks(data)
        self._create_page(db_id, properties, children)

    def _build_weekly_report_blocks(self, data: dict) -> list:
        blocks = []

        # 宏观事件
        blocks.append({"object": "block", "type": "heading_2",
                        "heading_2": {"rich_text": [{"text": {"content": "📅 本周宏观事件"}}]}})
        for event in data.get("macro_events", []):
            emoji = "🟢" if event.get("market_impact") == "看多" else "🔴" if event.get("market_impact") == "看空" else "⚪"
            blocks.append({"object": "block", "type": "bulleted_list_item",
                            "bulleted_list_item": {"rich_text": [{"text": {"content":
                                f"{emoji} {event.get('date','')} {event.get('event','')} — 预期:{event.get('consensus_expectation','')} — {event.get('impact_reason','')}"
                            }}]}})

        # 财报日历
        blocks.append({"object": "block", "type": "heading_2",
                        "heading_2": {"rich_text": [{"text": {"content": "📊 重要财报日历"}}]}})
        for e in data.get("earnings_calendar", []):
            blocks.append({"object": "block", "type": "bulleted_list_item",
                            "bulleted_list_item": {"rich_text": [{"text": {"content":
                                f"{e.get('report_date','')} {e.get('ticker','')} {e.get('company','')} | EPS预期:{e.get('eps_estimate','')} | 关注:{e.get('key_watch','')}"
                            }}]}})

        # 资金流向
        blocks.append({"object": "block", "type": "heading_2",
                        "heading_2": {"rich_text": [{"text": {"content": "💰 资金流向"}}]}})
        for f in data.get("fund_flows", []):
            arrow = "↑" if f.get("flow_direction") == "流入" else "↓"
            blocks.append({"object": "block", "type": "bulleted_list_item",
                            "bulleted_list_item": {"rich_text": [{"text": {"content":
                                f"{arrow} {f.get('sector_or_etf','')} | {f.get('amount_estimate','')} | 动能:{f.get('momentum','')}"
                            }}]}})

        # 风险清单
        blocks.append({"object": "block", "type": "heading_2",
                        "heading_2": {"rich_text": [{"text": {"content": "⚠️ 本周风险清单"}}]}})
        for r in data.get("risk_alerts", []):
            blocks.append({"object": "block", "type": "bulleted_list_item",
                            "bulleted_list_item": {"rich_text": [{"text": {"content":
                                f"概率:{r.get('probability','')} 影响:{r.get('impact','')} — {r.get('risk','')} | 对冲:{r.get('hedge_suggestion','')}"
                            }}]}})

        return blocks[:100]  # Notion 单次最多100个块

    def write_insider_buying(self, data: dict):
        """写入内部人士买入数据"""
        db_id = self.databases.get("insider_buying")
        if not db_id:
            return

        for signal in data.get("insider_signals", []):
            properties = {
                "标的": self._title(f"{signal.get('ticker','')} - {signal.get('company_name','')}"),
                "分析日期": self._date(data.get("analysis_date", "")),
                "内部人职位": self._text(signal.get("insider_title", "")),
                "买入金额(USD)": self._number(signal.get("buy_amount_usd", 0)),
                "买入价格": self._number(signal.get("purchase_price", 0)),
                "当前价格": self._number(signal.get("current_price", 0)),
                "价格变化%": self._number(signal.get("price_change_since_buy_pct", 0)),
                "为何重要": self._text(signal.get("why_significant", "")),
                "风险因素": self._text(signal.get("risk_factors", "")),
                "来源链接": self._url(signal.get("source_url", "")),
            }
            self._create_page(db_id, properties)

    def write_short_squeeze(self, data: dict):
        """写入空头挤压候选"""
        db_id = self.databases.get("short_squeeze")
        if not db_id:
            return

        for s in data.get("squeeze_candidates", []):
            properties = {
                "标的": self._title(f"{s.get('ticker','')} - {s.get('company_name','')}"),
                "分析日期": self._date(data.get("analysis_date", "")),
                "空头比例%": self._number(s.get("short_float_pct", 0)),
                "平仓天数": self._number(s.get("days_to_cover", 0)),
                "借入利率%": self._number(s.get("borrow_rate_pct", 0)),
                "催化剂": self._text(s.get("catalyst", "")),
                "催化剂时间": self._text(s.get("catalyst_date", "")),
                "入场策略": self._text(s.get("entry_strategy", "")),
                "挤压失败风险(1-10)": self._number(s.get("squeeze_fail_risk_score", 5)),
                "失败原因": self._text(s.get("squeeze_fail_reason", "")),
                "来源": self._url(s.get("source_url", "")),
            }
            self._create_page(db_id, properties)

    def write_ma_radar(self, data: dict):
        """写入并购雷达"""
        db_id = self.databases.get("ma_radar")
        if not db_id:
            return

        for m in data.get("ma_candidates", []):
            sources = m.get("sources", [])
            properties = {
                "标的": self._title(f"{m.get('ticker','')} - {m.get('company_name','')}"),
                "分析日期": self._date(data.get("analysis_date", "")),
                "行业": self._select(m.get("sector", "")),
                "当前价格": self._number(m.get("current_price", 0)),
                "预计收购溢价%": self._number(m.get("estimated_takeover_premium_pct", 0)),
                "隐含收购价": self._number(m.get("implied_takeover_price", 0)),
                "潜在收购方": self._text(", ".join(m.get("potential_acquirers", []))),
                "催化剂证据": self._text(m.get("catalyst_evidence", "")),
                "监管风险": self._select(m.get("regulatory_risk", "中")),
                "交易概率": self._select(m.get("deal_probability_estimate", "中")),
                "来源": self._url(sources[0] if sources else ""),
            }
            self._create_page(db_id, properties)

    def write_sentiment_arb(self, data: dict):
        """写入情绪套利机会"""
        db_id = self.databases.get("sentiment_arb")
        if not db_id:
            return

        for idea in data.get("sentiment_arb_ideas", []):
            properties = {
                "标的": self._title(f"{idea.get('ticker','')} - {idea.get('company_name','')}"),
                "分析日期": self._date(data.get("analysis_date", "")),
                "负面情绪原因": self._text(idea.get("negative_sentiment_reason", "")),
                "基本面矛盾点": self._text(idea.get("key_metric_contradiction", "")),
                "基本面优势": self._text(", ".join(idea.get("fundamental_strengths", []))),
                "重估催化剂": self._text(idea.get("expected_catalyst_for_rerating", "")),
                "持有周期": self._select(idea.get("time_horizon", "3-6个月")),
                "来源": self._url(idea.get("source_url", "")),
            }
            self._create_page(db_id, properties)

    def write_institutional(self, data: dict):
        """写入机构持仓变化"""
        db_id = self.databases.get("institutional")
        if not db_id:
            return

        # 共识买入是最有价值的信号，单独创建页面
        for buy in data.get("consensus_buys", []):
            properties = {
                "标的": self._title(f"[共识买入] {buy.get('ticker','')}"),
                "报告季度": self._text(data.get("report_quarter", "")),
                "分析日期": self._date(data.get("analysis_date", "")),
                "买入基金": self._text(", ".join(buy.get("funds_buying", []))),
                "总仓位规模(M)": self._number(buy.get("total_new_value_m", 0)),
                "行业": self._select(buy.get("sector", "")),
                "信号类型": self._select("共识买入"),
            }
            self._create_page(db_id, properties)

        # 新建仓
        for pos in data.get("new_positions", []):
            properties = {
                "标的": self._title(f"[新建仓] {pos.get('fund','')} - {pos.get('ticker','')}"),
                "报告季度": self._text(data.get("report_quarter", "")),
                "分析日期": self._date(data.get("analysis_date", "")),
                "买入基金": self._text(pos.get("fund", "")),
                "总仓位规模(M)": self._number(pos.get("position_value_m", 0)),
                "推测逻辑": self._text(pos.get("thesis_guess", "")),
                "信号类型": self._select("新建仓"),
            }
            self._create_page(db_id, properties)

    def write_correlation_map(self, data: dict):
        """写入关联性地图"""
        db_id = self.databases.get("correlation_map")
        if not db_id:
            logger.warning("未配置 correlation_map 数据库ID，跳过")
            return

        properties = {
            "分析标题": self._title(f"关联性地图 {data.get('analysis_date', '')}"),
            "分析日期": self._date(data.get("analysis_date", "")),
            "当前VIX": self._number(data.get("current_vix", 0)),
            "收益率曲线利差": self._number(data.get("yield_curve_spread", 0)),
            "宏观阶段": self._text(data.get("macro_regime", "")),
            "配置影响": self._text(data.get("regime_implication", "")),
        }

        # 构建页面正文：每个异常资产对一个段落
        children = []
        for anomaly in data.get("correlation_anomalies", []):
            children.append({"object": "block", "type": "heading_3",
                             "heading_3": {"rich_text": [{"text": {"content": anomaly.get("asset_pair", "")}}]}})
            children.append({"object": "block", "type": "paragraph",
                             "paragraph": {"rich_text": [{"text": {"content":
                                 f"异常: {anomaly.get('anomaly_description', '')}"
                             }}]}})
            children.append({"object": "block", "type": "paragraph",
                             "paragraph": {"rich_text": [{"text": {"content":
                                 f"历史信号: {anomaly.get('historical_signal_meaning', '')}"
                             }}]}})
            trade = anomaly.get("normalization_trade", {})
            if trade:
                children.append({"object": "block", "type": "paragraph",
                                 "paragraph": {"rich_text": [{"text": {"content":
                                     f"均值回归交易: {trade.get('description', '')} | 工具: {', '.join(trade.get('instruments', []))} | 触发: {trade.get('entry_trigger', '')}"
                                 }}]}})

        self._create_page(db_id, properties, children[:100])

    def write_macro_analysis(self, data: dict):
        """写入宏观面分析"""
        db_id = self.databases.get("macro_analysis")
        if not db_id:
            logger.warning("未配置 macro_analysis 数据库ID，跳过")
            return

        snap = data.get("macro_snapshot", {})
        yc   = data.get("yield_curve", {})
        analogs = data.get("historical_analogs", [])
        analog_text = " | ".join(
            f"{a.get('period','')}({a.get('similarity','')})" for a in analogs
        )
        sources = data.get("sources", [])

        properties = {
            "分析标题": self._title(f"宏观分析 {data.get('analysis_date', '')}"),
            "分析日期": self._date(data.get("analysis_date", "")),
            "收益率曲线状态": self._select(yc.get("status", "正常")),
            "10Y-2Y利差": self._number(yc.get("spread_10y2y", 0)),
            "偏好行业": self._text(", ".join(data.get("favored_sectors", []))),
            "规避行业": self._text(", ".join(data.get("avoid_sectors", []))),
            "3个月宏观展望": self._text(data.get("macro_outlook_3m", "")),
            "关键风险": self._text(data.get("key_risk", "")),
            "历史类比期": self._text(analog_text),
            "来源": self._url(sources[0] if sources else ""),
        }
        self._create_page(db_id, properties)

    def write_daily_trade(self, data: dict):
        """写入每日多空精选"""
        db_id = self.databases.get("daily_trade")
        if not db_id:
            logger.warning("未配置 daily_trade 数据库ID，跳过")
            return

        sentiment = data.get("sentiment", {})
        market_context = data.get("market_context", "")
        regime = sentiment.get("regime", "")
        regime_note = sentiment.get("regime_implication", "")
        sentiment_summary = f"VIX {sentiment.get('vix_current','')} | {sentiment.get('fear_greed_label','')}({sentiment.get('fear_greed_index','')}) | P/C {sentiment.get('put_call_ratio','')} | {regime_note}"

        for idea in [data.get("long_idea", {}), data.get("short_idea", {})]:
            if not idea.get("ticker"):
                continue
            direction = idea.get("direction", "多")
            icon = "📈" if direction == "多" else "📉"
            properties = {
                "标的":     self._title(f"{icon} {idea.get('ticker','')} - {idea.get('company_name', '')}"),
                "分析日期": self._date(data.get("analysis_date", "")),
                "方向":     self._select(direction),
                "当前价格": self._number(idea.get("current_price", 0)),
                "入场区间": self._text(idea.get("entry_zone", "")),
                "止损价":   self._number(idea.get("stop_loss", 0)),
                "目标价":   self._number(idea.get("target_price", 0)),
                "盈亏比":   self._number(idea.get("risk_reward_ratio", 0)),
                "投资逻辑": self._text(idea.get("thesis", "")),
                "催化剂":   self._text(idea.get("catalyst", "")),
                "持有周期": self._select(idea.get("time_horizon", "")),
                "技术信号": self._text(idea.get("technical_signal", "")),
                "大盘背景": self._text(f"{market_context} | 情绪: {sentiment_summary}"),
                "来源":     self._url(idea.get("source_url", "")),
            }
            self._create_page(db_id, properties)

    def write_dividend_danger(self, data: dict):
        """写入分红危险雷达"""
        db_id = self.databases.get("dividend_danger")
        if not db_id:
            return

        for trap in data.get("dividend_traps", []):
            alt = trap.get("safer_alternative", {})
            properties = {
                "标的": self._title(f"⚠️ {trap.get('ticker','')} - {trap.get('company_name','')}"),
                "分析日期": self._date(data.get("analysis_date", "")),
                "当前股息率%": self._number(trap.get("current_yield_pct", 0)),
                "派息比率%": self._number(trap.get("payout_ratio_pct", 0)),
                "自由现金流覆盖": self._select(trap.get("free_cash_flow_vs_dividend", "勉强覆盖")),
                "净债务/EBITDA": self._number(trap.get("net_debt_ebitda", 0)),
                "降息概率": self._select(trap.get("cut_probability", "中")),
                "降息原因": self._text(trap.get("cut_probability_reason", "")),
                "更安全替代品": self._text(f"{alt.get('ticker','')} ({alt.get('yield_pct','')}%) - {alt.get('why_safer','')}"),
                "来源": self._url(trap.get("source_url", "")),
            }
            self._create_page(db_id, properties)

    # ─────────────────────────────────────────
    # 统一路由方法
    # ─────────────────────────────────────────

    def write(self, task_name: str, data: dict):
        """根据任务名称路由到对应的写入方法"""
        writers = {
            "weekly_report": self.write_weekly_report,
            "macro_analysis": self.write_macro_analysis,
            "insider_buying": self.write_insider_buying,
            "short_squeeze": self.write_short_squeeze,
            "ma_radar": self.write_ma_radar,
            "sentiment_arbitrage": self.write_sentiment_arb,
            "correlation_map": self.write_correlation_map,
            "institutional_positioning": self.write_institutional,
            "dividend_danger": self.write_dividend_danger,
            "daily_trade_ideas": self.write_daily_trade,
        }

        writer = writers.get(task_name)
        if writer:
            writer(data)
            logger.info(f"[{task_name}] 已写入 Notion")
        else:
            logger.warning(f"[{task_name}] 无对应的 Notion 写入方法，跳过")
