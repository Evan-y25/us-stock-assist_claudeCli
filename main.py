"""
main.py - 私人对冲基金自动化分析系统（Bedrock 版）
定时调用 AWS Bedrock Claude，LLM 自主调用 Tools 完成分析，结果写入 Notion

使用方式:
  python main.py                    # 启动调度器（持续运行）
  python main.py --task weekly_report   # 立即执行指定任务
  python main.py --all              # 立即执行所有任务
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
import schedule

from claude_runner import ClaudeRunner
from notion_writer import NotionWriter

# ─────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/quant_{datetime.now().strftime('%Y%m')}.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────
# 加载配置
# ─────────────────────────────────────────
def load_config() -> dict:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_prompts() -> dict:
    with open("prompts/prompts.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data.get("prompts", {})


# ─────────────────────────────────────────
# 核心执行逻辑
# ─────────────────────────────────────────
def execute_task(task_name: str, config: dict, prompts: dict):
    """执行单个分析任务"""

    if task_name not in prompts:
        logger.error(f"任务 '{task_name}' 在 prompts.yaml 中未找到")
        return False

    prompt_config = prompts[task_name]
    template = prompt_config.get("template", "")

    logger.info(f"{'='*50}")
    logger.info(f"开始执行: {prompt_config.get('name', task_name)}")
    logger.info(f"{'='*50}")

    try:
        # 1. 初始化 Claude Runner（合并 Claude + Tool API Keys）
        claude_config = config.get("claude", {}).copy()
        claude_config["results_dir"] = config.get("storage", {}).get("results_dir", "./results")
        tools_config = config.get("tools", {})
        claude_config["tavily_api_key"] = tools_config.get("tavily_api_key", "")
        claude_config["fred_api_key"] = tools_config.get("fred_api_key", "")
        runner = ClaudeRunner(claude_config)

        # 2. 构建提示词（注入变量）
        variables = {}
        if task_name == "portfolio_hedge":
            variables["PORTFOLIO_EXPOSURE"] = config.get("portfolio", {}).get("exposure", "")

        prompt = runner.build_prompt(template, variables)

        # 3. 执行 Bedrock Tool Use 循环
        result = runner.run(task_name, prompt)

        if not result.get("_meta", {}).get("success", False):
            logger.error(f"任务 '{task_name}' 执行失败")
            return False

        # 4. 写入 Notion
        notion_config = config.get("notion", {})
        if notion_config.get("token") and notion_config.get("token") != "YOUR_NOTION_INTEGRATION_TOKEN":
            writer = NotionWriter(
                token=notion_config["token"],
                databases=notion_config.get("databases", {})
            )
            writer.write(task_name, result)
        else:
            logger.warning("Notion token 未配置，跳过写入。结果已保存到本地 results/ 目录")

        logger.info(f"任务 '{task_name}' 完成 ✓")
        return True

    except Exception as e:
        logger.exception(f"任务 '{task_name}' 发生异常: {e}")
        return False


# ─────────────────────────────────────────
# 调度器设置
# ─────────────────────────────────────────
def setup_scheduler(config: dict, prompts: dict):
    """根据配置设置定时任务"""

    scheduler_config = config.get("scheduler", {})
    tasks = scheduler_config.get("tasks", [])

    for task in tasks:
        task_name = task["name"]
        cron_expr = task["cron"]

        # 解析 cron 表达式 (minute hour day_of_week)
        parts = cron_expr.split()
        minute, hour = parts[0], parts[1]
        day_of_week = parts[4] if len(parts) > 4 else "*"

        time_str = f"{hour.zfill(2)}:{minute.zfill(2)}"

        # 注册到 schedule 库
        day_map = {"1": "monday", "2": "tuesday", "3": "wednesday",
                   "4": "thursday", "5": "friday", "6": "saturday", "0": "sunday"}

        if day_of_week == "*":
            schedule.every().day.at(time_str).do(
                execute_task, task_name, config, prompts
            )
        elif day_of_week in day_map:
            getattr(schedule.every(), day_map[day_of_week]).at(time_str).do(
                execute_task, task_name, config, prompts
            )

        logger.info(f"已注册任务: {task_name} @ cron: {cron_expr}")

    logger.info(f"\n{'='*50}")
    logger.info("调度器已启动，等待执行任务...")
    logger.info(f"{'='*50}\n")

    while True:
        schedule.run_pending()
        time.sleep(30)


# ─────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────
def main():
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="私人对冲基金自动化分析系统")
    parser.add_argument("--task", type=str, help="立即执行指定任务名称")
    parser.add_argument("--all", action="store_true", help="立即执行所有任务")
    parser.add_argument("--list", action="store_true", help="列出所有可用任务")
    args = parser.parse_args()

    config = load_config()
    prompts = load_prompts()

    if args.list:
        print("\n可用任务列表:")
        for name, p in prompts.items():
            print(f"  {name:30s} {p.get('name', '')} [{p.get('schedule', '')}]")
        return

    if args.task:
        execute_task(args.task, config, prompts)

    elif args.all:
        # 按顺序执行所有任务
        task_order = [
            "weekly_report",
            "macro_analysis",
            "insider_buying",
            "short_squeeze",
            "ma_radar",
            "sentiment_arbitrage",
            "correlation_map",
            "dividend_danger",
            "institutional_positioning",
            "daily_trade_ideas",
        ]
        for task_name in task_order:
            if task_name in prompts and prompts[task_name].get("schedule") != "on_demand":
                execute_task(task_name, config, prompts)

    else:
        # 默认：启动调度器
        setup_scheduler(config, prompts)


if __name__ == "__main__":
    main()
