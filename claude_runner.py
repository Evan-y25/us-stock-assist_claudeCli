"""
claude_runner.py - 使用本地 Claude Code CLI 执行分析任务

流程：
  1. 构建包含工具调用说明的提示词
  2. 通过 `claude -p` 子进程发送给 Claude Code
  3. Claude Code 自主通过 Bash 调用 tool_cli.py 获取数据
  4. 返回最终 JSON 分析结果
"""

import json
import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 项目根目录（tool_cli.py 所在目录）
PROJECT_DIR = Path(__file__).parent.resolve()

# 工具调用说明，注入到每个任务提示词前面
TOOL_INSTRUCTION = f"""## 工具调用方式
你可以通过 Bash 执行以下命令来获取实时数据：

```bash
python3 {PROJECT_DIR}/tool_cli.py <tool_name> '<json_params>'
```

可用工具及参数：

1. **web_search** — 联网搜索最新新闻和市场数据
   python3 {PROJECT_DIR}/tool_cli.py web_search '{{"query": "搜索关键词", "max_results": 5}}'

2. **get_market_data** — 获取股票/ETF 价格、技术指标、财务数据
   python3 {PROJECT_DIR}/tool_cli.py get_market_data '{{"ticker": "AAPL", "data_type": "financials"}}'
   data_type 可选: price_history, technical_indicators, financials, short_interest
   period 可选: 5d, 1mo, 3mo, 6mo, 1y, 2y（默认 3mo）

3. **get_macro_data** — FRED 宏观经济指标
   python3 {PROJECT_DIR}/tool_cli.py get_macro_data '{{"series_id": "FEDFUNDS"}}'
   常用 series_id: CPIAUCSL, FEDFUNDS, GDP, UNRATE, DGS10, T10Y2Y, VIXCLS

4. **get_sec_data** — SEC EDGAR 披露数据
   python3 {PROJECT_DIR}/tool_cli.py get_sec_data '{{"query_type": "insider_trading", "ticker": "AAPL", "days_back": 30}}'
   query_type 可选: insider_trading, institutional_13f, activist_filings, major_events

**重要**：
- 每次只调用你需要的工具，获取到数据后再进行分析
- 请逐步调用工具收集数据，最后输出 JSON 结果
- 最终输出时，只输出纯 JSON，不要在 JSON 前后添加任何说明文字、总结或 markdown 代码块标记
- 确保 JSON 完整，所有大括号和方括号正确闭合
"""


class ClaudeRunner:
    def __init__(self, config: dict):
        self.model_id = config.get("model_id", "sonnet")
        self.max_tokens = config.get("max_tokens", 16000)
        self.results_dir = Path(config.get("results_dir", "./results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def build_prompt(self, template: str, variables: dict = None) -> str:
        """将模板变量注入提示词"""
        variables = variables or {}
        variables["DATE"] = datetime.now().strftime("%Y-%m-%d")
        variables["TIME"] = datetime.now().strftime("%H:%M")
        prompt = template
        for key, value in variables.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt

    def run(self, task_name: str, prompt: str) -> dict:
        """
        使用本地 Claude Code CLI 执行分析任务
        Claude Code 自主通过 Bash 调用 tool_cli.py 获取数据
        """
        logger.info(f"[{task_name}] 开始执行 @ {datetime.now().strftime('%H:%M:%S')}")

        # 将工具调用说明注入提示词
        full_prompt = TOOL_INSTRUCTION + "\n\n" + prompt

        try:
            # 构建环境变量：清除 CLAUDECODE 以允许嵌套调用
            env = os.environ.copy()
            env.pop("CLAUDECODE", None)

            # 调用 claude CLI
            logger.info(f"[{task_name}] 调用 Claude Code CLI (model: {self.model_id})...")
            result = subprocess.run(
                [
                    "claude", "-p", full_prompt,
                    "--output-format", "text",
                    "--model", self.model_id,
                    "--max-turns", "30",
                    "--allowedTools", "Bash",
                ],
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                cwd=str(PROJECT_DIR),
            )

            if result.returncode != 0:
                logger.error(f"[{task_name}] Claude Code 返回错误: {result.stderr}")
                return self._error_result(task_name, f"CLI_ERROR: {result.stderr[:500]}")

            output = result.stdout.strip()
            logger.info(f"[{task_name}] Claude Code 返回 {len(output)} 字符")

            # 解析 JSON
            parsed = self._extract_json(output)

            if not parsed:
                logger.warning(f"[{task_name}] 无法解析 JSON，保存原始输出")
                parsed = {"raw_output": output, "parse_error": True}

            parsed["_meta"] = {
                "task_name": task_name,
                "executed_at": datetime.now().isoformat(),
                "runner": "claude_code_cli",
                "model": self.model_id,
                "success": "parse_error" not in parsed,
            }

            self._save_result(task_name, parsed)
            logger.info(f"[{task_name}] 完成")
            return parsed

        except subprocess.TimeoutExpired:
            logger.error(f"[{task_name}] Claude Code 执行超时 (600s)")
            return self._error_result(task_name, "TIMEOUT")

        except Exception as e:
            logger.exception(f"[{task_name}] 未预期错误: {e}")
            return self._error_result(task_name, str(e))

    def _extract_json(self, text: str) -> Optional[dict]:
        """从文本中提取 JSON 对象"""
        # 先尝试直接解析
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 代码块中的内容
        code_block = re.search(r"```json\s*([\s\S]*?)```", text)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 去掉所有 markdown code fence 再找
        cleaned = re.sub(r"```json\s*", "", text)
        cleaned = re.sub(r"```\s*", "", cleaned)

        # 找到最外层最大的 { ... } 块
        # 从后往前找最后一个 }，从前往后找第一个 {
        last_brace = cleaned.rfind("}")
        first_brace = cleaned.find("{")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(cleaned[first_brace:last_brace+1])
            except json.JSONDecodeError:
                pass

        # 逐层匹配大括号
        depth, start = 0, -1
        for i, char in enumerate(cleaned):
            if char == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        return json.loads(cleaned[start:i+1])
                    except json.JSONDecodeError:
                        continue
        return None

    def _save_result(self, task_name: str, data: dict):
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        filename = self.results_dir / f"{task_name}_{date_str}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[{task_name}] 已保存到 {filename}")

    def _error_result(self, task_name: str, error_msg: str) -> dict:
        return {
            "_meta": {
                "task_name": task_name,
                "executed_at": datetime.now().isoformat(),
                "runner": "claude_code_cli",
                "success": False,
                "error": error_msg,
            }
        }
