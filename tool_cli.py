#!/usr/bin/env python3
"""
tool_cli.py - 将 tools.py 中的工具包装为命令行可调用
供 Claude Code 通过 Bash 工具调用

用法：
  python3 tool_cli.py web_search '{"query": "Fed rate decision"}'
  python3 tool_cli.py get_market_data '{"ticker": "AAPL", "data_type": "financials"}'
  python3 tool_cli.py get_macro_data '{"series_id": "FEDFUNDS"}'
  python3 tool_cli.py get_sec_data '{"query_type": "insider_trading", "ticker": "AAPL"}'
"""

import json
import sys
from pathlib import Path

import yaml

from tools import execute_tool


def load_api_keys() -> dict:
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tools_config = config.get("tools", {})
    return {
        "tavily": tools_config.get("tavily_api_key", ""),
        "fred": tools_config.get("fred_api_key", ""),
    }


def main():
    if len(sys.argv) < 3:
        print("用法: python3 tool_cli.py <tool_name> '<json_params>'")
        print("可用工具: web_search, get_market_data, get_macro_data, get_sec_data")
        sys.exit(1)

    tool_name = sys.argv[1]
    try:
        tool_input = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"JSON 参数解析失败: {e}"}))
        sys.exit(1)

    api_keys = load_api_keys()
    result = execute_tool(tool_name, tool_input, api_keys)
    print(result)


if __name__ == "__main__":
    main()
