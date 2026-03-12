"""
notion_setup.py - 自动在 Notion 中创建所有需要的数据库，并回填 config.yaml

使用前提：
  1. 在 Notion 中创建一个父页面（Page），将你的 Integration 添加为该页面的成员
  2. 复制该页面的 ID（URL 中 notion.so/xxx 后面那段，去掉连字符）
  3. 运行:  python notion_setup.py --parent-page-id <PAGE_ID>

如果所有数据库已存在，可加 --skip-existing 跳过已有项。
"""

import argparse
import json
import re
import sys
import yaml
import httpx
from notion_client import Client
from notion_client.errors import APIResponseError

NOTION_VERSION = "2022-06-28"

CONFIG_PATH = "config/config.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# 每个数据库的 schema 定义
# key = config.yaml 中 databases 下的字段名
# value = {"title": 数据库显示名, "properties": Notion properties schema}
# ─────────────────────────────────────────────────────────────────────────────
DATABASE_SCHEMAS = {
    "weekly_report": {
        "title": "📅 每周综合报告",
        "properties": {
            "报告标题":      {"title": {}},
            "日期":          {"date": {}},
            "编辑寄语":      {"rich_text": {}},
            "做多标的":      {"rich_text": {}},
            "做多入场":      {"rich_text": {}},
            "做多止损":      {"rich_text": {}},
            "做多目标":      {"rich_text": {}},
            "做空标的":      {"rich_text": {}},
        }
    },
    "insider_buying": {
        "title": "🔍 内部人士买入",
        "properties": {
            "标的":           {"title": {}},
            "分析日期":       {"date": {}},
            "内部人职位":     {"rich_text": {}},
            "买入金额(USD)":  {"number": {"format": "dollar"}},
            "买入价格":       {"number": {"format": "dollar"}},
            "当前价格":       {"number": {"format": "dollar"}},
            "价格变化%":      {"number": {"format": "percent"}},
            "为何重要":       {"rich_text": {}},
            "风险因素":       {"rich_text": {}},
            "来源链接":       {"url": {}},
        }
    },
    "short_squeeze": {
        "title": "🚀 空头挤压候选",
        "properties": {
            "标的":             {"title": {}},
            "分析日期":         {"date": {}},
            "空头比例%":        {"number": {"format": "percent"}},
            "平仓天数":         {"number": {"format": "number"}},
            "借入利率%":        {"number": {"format": "percent"}},
            "催化剂":           {"rich_text": {}},
            "催化剂时间":       {"rich_text": {}},
            "入场策略":         {"rich_text": {}},
            "挤压失败风险(1-10)": {"number": {"format": "number"}},
            "失败原因":         {"rich_text": {}},
            "来源":             {"url": {}},
        }
    },
    "ma_radar": {
        "title": "🎯 并购雷达",
        "properties": {
            "标的":             {"title": {}},
            "分析日期":         {"date": {}},
            "行业":             {"select": {"options": [
                {"name": "科技", "color": "blue"},
                {"name": "医疗", "color": "green"},
                {"name": "金融", "color": "yellow"},
                {"name": "消费", "color": "orange"},
                {"name": "能源", "color": "red"},
                {"name": "工业", "color": "gray"},
                {"name": "其他", "color": "default"},
            ]}},
            "当前价格":         {"number": {"format": "dollar"}},
            "预计收购溢价%":    {"number": {"format": "percent"}},
            "隐含收购价":       {"number": {"format": "dollar"}},
            "潜在收购方":       {"rich_text": {}},
            "催化剂证据":       {"rich_text": {}},
            "监管风险":         {"select": {"options": [
                {"name": "低", "color": "green"},
                {"name": "中", "color": "yellow"},
                {"name": "高", "color": "red"},
            ]}},
            "交易概率":         {"select": {"options": [
                {"name": "低", "color": "red"},
                {"name": "中", "color": "yellow"},
                {"name": "高", "color": "green"},
            ]}},
            "来源":             {"url": {}},
        }
    },
    "sentiment_arb": {
        "title": "💡 情绪套利机会",
        "properties": {
            "标的":             {"title": {}},
            "分析日期":         {"date": {}},
            "负面情绪原因":     {"rich_text": {}},
            "基本面矛盾点":     {"rich_text": {}},
            "基本面优势":       {"rich_text": {}},
            "重估催化剂":       {"rich_text": {}},
            "持有周期":         {"select": {"options": [
                {"name": "1个月内", "color": "red"},
                {"name": "1-3个月", "color": "yellow"},
                {"name": "3-6个月", "color": "blue"},
                {"name": "6-12个月", "color": "green"},
                {"name": "12个月+", "color": "purple"},
            ]}},
            "来源":             {"url": {}},
        }
    },
    "institutional": {
        "title": "🏛️ 机构持仓变化",
        "properties": {
            "标的":             {"title": {}},
            "报告季度":         {"rich_text": {}},
            "分析日期":         {"date": {}},
            "买入基金":         {"rich_text": {}},
            "总仓位规模(M)":    {"number": {"format": "number"}},
            "行业":             {"select": {"options": [
                {"name": "科技", "color": "blue"},
                {"name": "医疗", "color": "green"},
                {"name": "金融", "color": "yellow"},
                {"name": "消费", "color": "orange"},
                {"name": "能源", "color": "red"},
                {"name": "工业", "color": "gray"},
                {"name": "其他", "color": "default"},
            ]}},
            "推测逻辑":         {"rich_text": {}},
            "信号类型":         {"select": {"options": [
                {"name": "共识买入", "color": "green"},
                {"name": "新建仓", "color": "blue"},
                {"name": "大幅减仓", "color": "red"},
                {"name": "清仓", "color": "pink"},
            ]}},
        }
    },
    "macro_analysis": {
        "title": "🌍 宏观面分析",
        "properties": {
            "分析标题":         {"title": {}},
            "分析日期":         {"date": {}},
            "收益率曲线状态":   {"select": {"options": [
                {"name": "正常", "color": "green"},
                {"name": "平坦", "color": "yellow"},
                {"name": "倒挂", "color": "red"},
            ]}},
            "10Y-2Y利差": {"number": {"format": "number"}},
            "偏好行业":          {"rich_text": {}},
            "规避行业":          {"rich_text": {}},
            "3个月宏观展望":     {"rich_text": {}},
            "关键风险":         {"rich_text": {}},
            "历史类比期":       {"rich_text": {}},
            "来源":             {"url": {}},
        }
    },
    "dividend_danger": {
        "title": "⚠️ 分红危险雷达",
        "properties": {
            "标的":             {"title": {}},
            "分析日期":         {"date": {}},
            "当前股息率%":      {"number": {"format": "percent"}},
            "派息比率%":        {"number": {"format": "percent"}},
            "自由现金流覆盖":   {"select": {"options": [
                {"name": "充裕覆盖", "color": "green"},
                {"name": "勉强覆盖", "color": "yellow"},
                {"name": "无法覆盖", "color": "red"},
            ]}},
            "净债务/EBITDA":    {"number": {"format": "number"}},
            "降息概率":         {"select": {"options": [
                {"name": "低", "color": "green"},
                {"name": "中", "color": "yellow"},
                {"name": "高", "color": "red"},
            ]}},
            "降息原因":         {"rich_text": {}},
            "更安全替代品":     {"rich_text": {}},
            "来源":             {"url": {}},
        }
    },
    "daily_trade": {
        "title": "📈 每日多空精选",
        "properties": {
            "标的":     {"title": {}},
            "分析日期": {"date": {}},
            "方向":     {"select": {"options": [
                {"name": "多", "color": "green"},
                {"name": "空", "color": "red"},
            ]}},
            "当前价格": {"number": {"format": "dollar"}},
            "入场区间": {"rich_text": {}},
            "止损价":   {"number": {"format": "dollar"}},
            "目标价":   {"number": {"format": "dollar"}},
            "盈亏比":   {"number": {"format": "number"}},
            "投资逻辑": {"rich_text": {}},
            "催化剂":   {"rich_text": {}},
            "持有周期": {"select": {"options": [
                {"name": "日内",   "color": "red"},
                {"name": "1-3天",  "color": "yellow"},
                {"name": "1-2周",  "color": "blue"},
            ]}},
            "技术信号": {"rich_text": {}},
            "大盘背景": {"rich_text": {}},
            "来源":     {"url": {}},
        }
    },
    "correlation_map": {
        "title": "🗺️ 关联性地图",
        "properties": {
            "分析标题":       {"title": {}},
            "分析日期":       {"date": {}},
            "当前VIX":        {"number": {"format": "number"}},
            "收益率曲线利差": {"number": {"format": "number"}},
            "宏观阶段":       {"rich_text": {}},
            "资产配置影响":   {"rich_text": {}},
            "关联异常摘要":   {"rich_text": {}},
        }
    },
    "portfolio_hedge": {
        "title": "🛡️ 对冲策略分析",
        "properties": {
            "策略标题":  {"title": {}},
            "分析日期":  {"date": {}},
            "当前VIX":   {"number": {"format": "number"}},
            "VIX百分位": {"rich_text": {}},
            "组合暴露":  {"rich_text": {}},
            "推荐策略":  {"rich_text": {}},
            "推荐原因":  {"rich_text": {}},
        }
    },
}


def normalize_page_id(page_id: str) -> str:
    """去除连字符，标准化为 32 位 hex 字符串"""
    return page_id.replace("-", "").strip()


def create_database(token: str, parent_page_id: str, key: str, schema: dict) -> str:
    """在 Notion 中创建一个数据库，返回其 ID（使用原始 httpx 绕过 notion-client 3.x 的 properties 丢失 bug）"""
    db_title   = schema["title"]
    properties = schema["properties"]

    print(f"  创建数据库: {db_title} ...", end=" ", flush=True)
    body = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "title": [{"type": "text", "text": {"content": db_title}}],
        "properties": properties,
    }
    resp = httpx.post(
        "https://api.notion.com/v1/databases",
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        },
        content=json.dumps(body, ensure_ascii=False).encode(),
        timeout=30,
    )
    if not resp.is_success:
        print(f"❌  HTTP {resp.status_code}: {resp.text[:300]}")
        raise RuntimeError(resp.text)
    db_id = resp.json()["id"]
    print(f"✅  {db_id}")
    return db_id


def update_config(db_ids: dict):
    """将数据库 ID 回填到 config.yaml"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    config = yaml.safe_load(content)
    config["notion"]["databases"].update(db_ids)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\n✅ config.yaml 已更新")


def main():
    parser = argparse.ArgumentParser(description="自动创建 Notion 数据库并回填 config.yaml")
    parser.add_argument(
        "--parent-page-id",
        required=True,
        help="Notion 父页面 ID（从页面 URL 中复制，例如 https://notion.so/My-Page-<ID>）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新创建所有数据库，包括 config.yaml 中已有 ID 的（默认跳过已有 ID 的）"
    )
    args = parser.parse_args()

    # 读取配置
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    token = config["notion"]["token"]
    existing_ids = config["notion"].get("databases", {})

    client = Client(auth=token)
    parent_page_id = normalize_page_id(args.parent_page_id)

    print(f"\n🚀 开始创建 Notion 数据库（父页面: {parent_page_id}）\n")

    created = {}
    errors = []

    for key, schema in DATABASE_SCHEMAS.items():
        current_val = existing_ids.get(key, "")
        if not args.force and current_val and current_val != "YOUR_DATABASE_ID":
            print(f"  跳过 {key}（已有 ID: {current_val}）")
            continue
        try:
            db_id = create_database(token, parent_page_id, key, schema)
            created[key] = db_id
        except Exception as e:
            errors.append((key, str(e)))

    if created:
        print(f"\n📝 正在回填 {len(created)} 个数据库 ID 到 config.yaml ...")
        update_config(created)

    if errors:
        print(f"\n⚠️  以下数据库创建失败：")
        for key, msg in errors:
            print(f"  {key}: {msg}")
        sys.exit(1)

    print(f"\n🎉 完成！共创建 {len(created)} 个数据库。")
    print("   请在 Notion 中确认页面已正确创建，然后运行主程序。\n")


if __name__ == "__main__":
    main()
