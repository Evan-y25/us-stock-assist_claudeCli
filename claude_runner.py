"""
claude_runner.py - Anthropic API + Tool Use 版本

流程：
  1. 把提示词 + 4个 Tool Schema 发给 Anthropic API
  2. LLM 自主决定调用哪个工具、传什么参数
  3. 程序执行工具，把结果返回给 LLM
  4. 循环直到 LLM 输出最终 JSON 结果
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic

from tools import TOOL_SCHEMAS, execute_tool

logger = logging.getLogger(__name__)


def _convert_tool_schema(schema: dict) -> dict:
    """将 Bedrock 格式的 tool schema 转换为 Anthropic API 格式"""
    return {
        "name": schema["name"],
        "description": schema["description"],
        "input_schema": schema["inputSchema"]["json"]
    }


class ClaudeRunner:
    def __init__(self, config: dict):
        self.model_id = config.get("model_id", "claude-sonnet-4-5-20250514")
        self.max_tokens = config.get("max_tokens", 4096)
        self.max_tool_rounds = config.get("max_tool_rounds", 10)
        self.results_dir = Path(config.get("results_dir", "./results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Tool 执行需要的 API Keys
        self.api_keys = {
            "tavily": config.get("tavily_api_key", ""),
            "fred": config.get("fred_api_key", ""),
        }

        # 初始化 Anthropic 客户端
        # 优先级：配置文件 api_key > 配置文件 auth_token > Claude Code OAuth > 环境变量
        api_key = config.get("api_key", "")
        auth_token = config.get("auth_token", "")

        if api_key and api_key not in ("YOUR_ANTHROPIC_API_KEY", ""):
            self.client = anthropic.Anthropic(api_key=api_key)
        elif auth_token and auth_token not in ("", "auto"):
            self.client = anthropic.Anthropic(auth_token=auth_token)
        elif auth_token == "auto":
            # 自动从 macOS Keychain 读取 Claude Code 的 OAuth token
            token = self._load_claude_code_oauth()
            self.client = anthropic.Anthropic(auth_token=token)
        else:
            # 兜底：让 SDK 自己从环境变量读取
            self.client = anthropic.Anthropic()

        # 转换 tool schemas
        self.tools = [_convert_tool_schema(s) for s in TOOL_SCHEMAS]

    @staticmethod
    def _load_claude_code_oauth() -> str:
        """从 macOS Keychain 读取 Claude Code 的 OAuth access token"""
        import subprocess
        try:
            raw = subprocess.check_output(
                ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            creds = json.loads(raw)
            token = creds["claudeAiOauth"]["accessToken"]
            logger.info("已从 macOS Keychain 读取 Claude Code OAuth token")
            return token
        except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(
                "无法从 macOS Keychain 读取 Claude Code OAuth token。"
                "请确保已通过 'claude' 命令登录，或改用 api_key 方式认证。"
            ) from e

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
        主执行方法：Anthropic Tool Use 循环
        LLM 自主调用工具直到完成任务
        """
        logger.info(f"[{task_name}] 开始执行 @ {datetime.now().strftime('%H:%M:%S')}")

        messages = [{"role": "user", "content": prompt}]
        tool_call_count = 0

        try:
            for round_num in range(self.max_tool_rounds + 1):

                # 调用 Anthropic Messages API
                response = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    tools=self.tools,
                    messages=messages,
                    temperature=0.1,
                )
                stop_reason = response.stop_reason

                logger.info(f"[{task_name}] Round {round_num} - stop_reason: {stop_reason}")

                # ── 情况1：LLM 完成，输出最终结果 ──
                if stop_reason == "end_turn":
                    final_text = self._extract_text(response.content)
                    parsed = self._extract_json(final_text)

                    if not parsed:
                        logger.warning(f"[{task_name}] 无法解析 JSON，保存原始输出")
                        parsed = {"raw_output": final_text, "parse_error": True}

                    parsed["_meta"] = {
                        "task_name": task_name,
                        "executed_at": datetime.now().isoformat(),
                        "tool_calls": tool_call_count,
                        "rounds": round_num,
                        "success": "parse_error" not in parsed
                    }

                    self._save_result(task_name, parsed)
                    logger.info(f"[{task_name}] 完成，共调用工具 {tool_call_count} 次")
                    return parsed

                # ── 情况2：LLM 要调用工具 ──
                elif stop_reason == "tool_use":
                    # 将 assistant 回复加入消息历史
                    messages.append({
                        "role": "assistant",
                        "content": [block.model_dump() for block in response.content]
                    })

                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            tool_call_count += 1
                            output = execute_tool(
                                block.name,
                                block.input,
                                self.api_keys
                            )
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output or "(no output)"
                            })

                    if not tool_results:
                        logger.error(f"[{task_name}] stop_reason=tool_use 但未找到任何 tool_use block")
                        break

                    messages.append({"role": "user", "content": tool_results})

                else:
                    logger.warning(f"[{task_name}] 意外的 stop_reason: {stop_reason}")
                    break

            logger.error(f"[{task_name}] 超过最大工具调用轮次 ({self.max_tool_rounds})")
            return self._error_result(task_name, "EXCEEDED_MAX_ROUNDS")

        except anthropic.APIError as e:
            logger.error(f"[{task_name}] Anthropic API 错误: {e}")
            return self._error_result(task_name, f"API_ERROR: {e}")

        except Exception as e:
            logger.exception(f"[{task_name}] 未预期错误: {e}")
            return self._error_result(task_name, str(e))

    def _extract_text(self, content: list) -> str:
        return " ".join(
            block.text
            for block in content
            if block.type == "text"
        ).strip()

    def _extract_json(self, text: str) -> Optional[dict]:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        depth, start = 0, -1
        for i, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        return json.loads(text[start:i+1])
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
                "success": False,
                "error": error_msg
            }
        }
