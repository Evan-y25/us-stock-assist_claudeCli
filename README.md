# 🏦 私人对冲基金自动化分析系统

基于 **AWS Bedrock Claude + Notion** 的量化分析自动化工作流，自动执行 11 类分析任务并写入 Notion 数据库。

---

## 📁 项目结构

```
us-stock-assist/
├── main.py                      # 主入口 + 调度器
├── claude_runner.py             # AWS Bedrock Converse API 调用封装
├── notion_writer.py             # Notion API 写入封装
├── tools.py                     # 工具定义（Tavily / yfinance / FRED / SEC）
├── notion_setup.py              # 一键创建所有 Notion 数据库
├── notion_setup_missing.py      # 补建缺失数据库（correlation_map / portfolio_hedge / daily_trade）
├── notion_fix_properties.py     # 修复已有 Notion 数据库字段
├── requirements.txt
├── config/
│   ├── config.yaml              # 真实配置（已 gitignore，含密钥）
│   └── config-example.yaml     # 配置模板（提交到 git）
├── prompts/
│   └── prompts.yaml             # 各任务提示词
├── results/                     # 本地结果缓存 (JSON，已 gitignore)
└── logs/                        # 运行日志（已 gitignore）
```

---

## ⚡ 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

复制配置模板并填写密钥：

```bash
cp config/config-example.yaml config/config.yaml
```

编辑 `config/config.yaml`，填写以下内容：

| 字段 | 获取方式 |
|------|---------|
| `notion.token` | https://www.notion.so/my-integrations → New integration |
| `claude.aws_access_key_id` / `aws_secret_access_key` | AWS IAM → 创建具有 Bedrock 权限的用户 |
| `tools.tavily_api_key` | https://app.tavily.com |
| `tools.fred_api_key` | https://fredaccount.stlouisfed.org（免费） |

### 3. 创建 Notion 数据库

在 Notion 中创建一个父页面，将 Integration 添加为成员，然后运行：

```bash
# 全量创建（首次使用）
python notion_setup.py --parent-page-id <父页面ID>

# 仅补建缺失的数据库（已有部分数据库时使用）
python notion_setup_missing.py --parent-page-id <父页面ID>
```

脚本会自动创建全部数据库并将 ID 回填到 `config.yaml`，已有 ID 的数据库自动跳过。

> **父页面 ID**：从页面 URL 中复制，例如 `notion.so/My-Page-31ea21c66c98...` 中的 `31ea21c66c98...`

### 4. 运行

```bash
# 查看所有可用任务
python main.py --list

# 立即执行所有任务（按顺序）
python main.py --all

# 启动调度器（按 cron 计划自动运行）
python main.py
```

**各策略单独执行命令：**

```bash
# 📅 每周综合报告
python main.py --task weekly_report

# 🌍 宏观面自上而下分析
python main.py --task macro_analysis

# 🔍 内部人士买入检测
python main.py --task insider_buying

# 🚀 空头挤压筛选器
python main.py --task short_squeeze

# 🎯 并购雷达
python main.py --task ma_radar

# 💡 情绪 vs 基本面套利
python main.py --task sentiment_arbitrage

# 🗺️ 危机中的关联性地图
python main.py --task correlation_map

# ⚠️ 分红危险雷达
python main.py --task dividend_danger

# 🏛️ 机构头寸分析
python main.py --task institutional_positioning

# 📈 每日多空精选（含情绪指数评估）
python main.py --task daily_trade_ideas

# 🛡️ 投资组合对冲策略（按需执行）
python main.py --task portfolio_hedge
```

---

## 📊 Notion 数据库设计

| 数据库 | config key | 主要字段 |
|--------|-----------|---------|
| 📅 每周综合报告 | `weekly_report` | 报告标题、日期、编辑寄语、做多/做空标的 |
| 🌍 宏观面分析 | `macro_analysis` | 分析标题、收益率曲线状态、偏好行业、3个月展望 |
| 🔍 内部人士买入 | `insider_buying` | 标的、买入金额、价格变化%、为何重要 |
| 🚀 空头挤压候选 | `short_squeeze` | 标的、空头比例%、平仓天数、催化剂 |
| 🎯 并购雷达 | `ma_radar` | 标的、行业、收购溢价%、监管风险、交易概率 |
| 💡 情绪套利 | `sentiment_arb` | 标的、负面情绪原因、基本面优势、持有周期 |
| 🏛️ 机构持仓变化 | `institutional` | 标的、信号类型、买入基金、总仓位规模 |
| ⚠️ 分红危险雷达 | `dividend_danger` | 标的、当前股息率%、派息比率%、降息概率 |
| 🗺️ 关联性地图 | `correlation_map` | 分析标题、当前VIX、收益率曲线利差、宏观阶段 |
| 🛡️ 对冲策略分析 | `portfolio_hedge` | 策略标题、当前VIX、VIX百分位、推荐策略 |
| 📈 每日多空精选 | `daily_trade` | 标的、方向、入场区间、止损价、目标价、盈亏比 |

---

## ⏰ 默认调度计划

| 任务 | 执行时间（上海时区） |
|------|-------------------|
| 📅 每周报告 | 每周一 07:30 |
| 🌍 宏观分析 | 每周一 08:00 |
| 🔍 内部人士买入 | 每周一 08:30 |
| 🚀 空头挤压 | 每周二 09:00 |
| 🎯 并购雷达 | 每周二 09:30 |
| 💡 情绪套利 | 每周三 09:00 |
| 🗺️ 关联性地图 | 每周三 09:30 |
| ⚠️ 分红危险 | 每周四 09:00 |
| 🏛️ 机构持仓 | 每周四 09:30 |
| 📈 每日多空精选 | 每个交易日 09:00 |
| 🛡️ 对冲策略 | 按需执行（`--task portfolio_hedge`） |

---

## 📈 每日多空精选说明

`daily_trade_ideas` 是每日执行的短线策略任务，核心特色是**情绪体制驱动**：

每次执行时首先评估市场情绪环境（VIX + CNN Fear & Greed 指数 + Put/Call Ratio），再据此动态调整选股逻辑：

| 情绪体制 | VIX 水平 | 策略偏向 | 盈亏比要求 |
|---------|---------|---------|-----------|
| 极度恐慌 | > 30 | 禁止追涨；优先超跌反弹做多 | 多 ≥ 2.0 / 空 ≥ 2.5 |
| 恐慌偏高 | 20-30 | 多单偏防御/超跌；空单需更强破位 | ≥ 2.0 |
| 中性 | 15-20 | 正常动量策略，多空均可 | ≥ 2.0 |
| 贪婪偏高 | 12-15 | 多单严控止损；空单信号更可靠 | ≥ 2.0 |
| 极度贪婪 | < 12 | 优先做空超买标的 | ≥ 2.5 |

输出的每条标的额外包含 `sentiment_alignment` 字段，标注信号是否与当前情绪顺势。

---

## 🖥️ 服务器部署（长期运行）

### 方式一：screen（简单）
```bash
screen -S quant
python main.py
# Ctrl+A, D 挂起
```

### 方式二：systemd 服务（推荐）
```bash
# /etc/systemd/system/quant-analyst.service
[Unit]
Description=Quant Analyst Auto System
After=network.target

[Service]
User=your_user
WorkingDirectory=/path/to/us-stock-assist
ExecStart=/usr/bin/python3 main.py
Restart=always

[Install]
WantedBy=multi-user.target

sudo systemctl enable quant-analyst
sudo systemctl start quant-analyst
```

---

## 🔧 常见问题

**Q: Notion 写入报错 "is not a property that exists"**
- `notion-client` 3.x 存在 bug，`databases.create()` 会静默丢弃 properties
- 使用 `notion_fix_properties.py` 修复（内部使用原始 httpx 调用绕过此 bug）：
  ```bash
  python notion_fix_properties.py                      # 修复所有数据库
  python notion_fix_properties.py --db weekly_report   # 修复指定数据库
  python notion_fix_properties.py --check              # 只检查不修改
  ```

**Q: 部分数据库没有创建怎么办？**
- 使用 `notion_setup_missing.py` 补建，已有 ID 的数据库会自动跳过：
  ```bash
  python notion_setup_missing.py --parent-page-id <父页面ID>
  ```

**Q: AWS Bedrock 调用失败？**
- 确认 IAM 用户有 `bedrock:InvokeModel` 权限
- 确认 `region` 设置与开通 Bedrock 的区域一致

**Q: Notion 写入 403 错误？**
- 确认父页面已将 Integration 添加为成员（Connections）
- 用 `notion_setup.py` 创建的数据库自动继承权限，手动创建的需手动共享

---

## ⚠️ 免责声明

本系统输出内容仅供参考，不构成任何投资建议。AI 分析可能存在错误，请结合自身判断独立决策。


