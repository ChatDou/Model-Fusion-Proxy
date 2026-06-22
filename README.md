# Model Fusion Proxy v0.2.1

> 🚀 **低成本高性能模型代理** · 基于 MoE（Mixture of Experts）架构的智能模型代理网关。自动判断任务难度，按需选择本地或云端模型——能用便宜的绝不用贵的，便宜不够时多模型协作模拟顶级模型。

---

## 📊 定位

Model Fusion Proxy 通过以下策略在成本与质量间平衡：

- **双层意图路由**：O(1) 正则快速分类 + 本地 LLM 语义兜底
- **Fallback 链容错**：非流式并发 racing / 流式串行后备，失败自动切换
- **MoA 四阶段融合**：多模型 Panel → Judge 合成 → Critic 审查 → 精修输出
- **本地模型加速**：Apple Silicon MLX / Ollama 零成本毫秒级推理

验证命令：`python full_benchmark.py`（需要 proxy 运行在 127.0.0.1:8000）

---

## 🏗️ 架构

```
客户端 (Hermes / Claude Code / IDE)
        │
        ▼
  Model Fusion Proxy (127.0.0.1:8000)
        │
        ├─ 意图分类 (Stage-1 正则 O(1) + Stage-2 本地 LLM)
        │
        └─ 路由决策
              │
              ├─ 普通路由 (Fallback Chain) — 按意图选模型 + 失败自动后备
              │
              └─ MoA 融合 (4 阶段)
                    ├─ Phase 1: Panel Drafting — 多模型并行出草稿
                    ├─ Phase 2: Draft Synthesis — Judge 合成统一草案
                    ├─ Phase 3: Critic Review — 双批评家 + 本地 AST 校验
                    └─ Phase 4: Final Refinement — 精修出最终答案
```

### MoA 四阶段明细

| 阶段 | 做什么 | 参与模型 |
|------|--------|---------|
| **Panel** | 多模型独立出草稿，失败的不影响后续 | 本地 Qwen 9B + DeepSeek V4 Pro + GLM 5.2 / MiniMax M3 |
| **Draft** | Judge 融合草稿为一体 | Gemini 2.5 Pro |
| **Critic** | 双批评家审查草案 + 本地 Python AST 编译语法检查 | DeepSeek V4 Pro + GLM 5.2 + 本地 AST |
| **Final** | Judge 吸收批评意见输出最终答案 | Gemini 2.5 Pro |

---

## 🎯 核心能力

### 双层意图路由
- **Stage-1**：预编译正则 + 关键词树，O(1) 纳秒级匹配，零延迟零成本
- **Stage-2**：未命中时调用本地 MLX Qwen 9B 做语义分类，200-500ms，失败自动降级到 general

### 五分类模型映射

| 分类 | 首选模型 | 典型场景 |
|------|---------|---------|
| `coding` | DeepSeek V4 Pro | 写代码、调试、算法、SQL、JSON |
| `reasoning` | Gemini 2.5 Pro | 深度推理、系统架构、数学证明 |
| `creative` | Qwen 9B MLX (本地) | 创意写作、文案、脑暴、角色扮演 |
| `chinese_nuance` | GLM 5.2 | 公文润色、古文翻译、政策解读 |
| `general` | Gemma 2B (本地) | 日常闲聊、简单问答 |

### 本地模型加速池

```
macOS (Apple Silicon):
  ├─ Ollama          → gemma:2b (1.7G) — 日常闲聊首选
  ├─ Ollama          → llama3.2:1b (1.3G) — 轻量辅助
  ├─ MLX Server      → Qwen 3.5 9B (6G) — 创意 / 分类主力
  └─ Metal GPU 加速   → 零成本、毫秒级、完全离线

云端：
  ├─ DeepSeek V4 Pro → 代码 / Agent 工具调用
  ├─ GLM 5.2         → 中文 / 批评家
  ├─ MiniMax M3      → 创意文案 / 后备
  └─ Gemini 2.5 Pro  → 推理 / Judge 裁判
```

### 降维打击能力

| 能力 | 方案 |
|------|------|
| **Fallback 容错** | 非流式 racing（谁快谁赢，cancel 输家）+ 流式串行后备 |
| **本地模型熔断** | 失败自动降级到云端 gemini-2.5-pro |
| **代码语法质量** | 本地 Python AST 编译校验，杜绝幻觉语法错误 |
| **工具调用适配** | 本地模型自动注入 tool→自然语言翻译层 |
| **双协议支持** | OpenAI + Anthropic 双向翻译，tool_choice 枚举穷举映射 |
| **统一内存防护** | 本地 GPU 并发信号量，防 OOM / Swap 崩溃 |

---

## ⚡ 快速开始

### 环境要求
- Python 3.10+
- [Ollama](https://ollama.ai/) (可选，本地模型)
- MLX (可选，Apple Silicon 本地加速)

### 安装

```bash
git clone https://github.com/ChatDou/Model-Fusion-Proxy.git
cd Model-Fusion-Proxy
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 配置

创建 `.env`：

```env
DEEPSEEK_API_KEY=sk-xxxxxxxx
GEMINI_API_KEY=AIzaxxxxxxxx
GLM_API_KEY=xxxxxxxxxxxxx
MINIMAX_API_KEY=xxxxxxxxxxxxxxxx
```

编辑 `config.yaml` 调整路由策略和模型参数。

### 启动

```bash
.venv/bin/python -m uvicorn main:app --port 8000 --host 127.0.0.1
```

### 接入

| 客户端 | 配置 |
|--------|------|
| Cursor / Copilot / IDE | Base URL → `http://127.0.0.1:8000/v1` |
| Claude Code | 指向同一地址 |
| 任何 OpenAI 兼容客户端 | 模型名选 `model-fusion` 启用 MoA，或指定具体模型 |

---

## 📁 项目结构

```
.
├── main.py              # FastAPI 入口 + Anthropic↔OpenAI 协议翻译
├── router.py            # 双层意图路由 + Fallback 链
├── fusion.py            # MoA 四阶段融合引擎
├── client.py            # HTTP 客户端 + 多 provider 抽象
├── config.yaml          # 模型配置、路由策略、融合参数
├── full_benchmark.py    # 全面基准测试
├── benchmark.py         # 快速功能测试
├── test_proxy.py        # 单元测试套件
├── test_fable_features.py # AST 校验 / 工具调用 / 视觉分类测试
├── start.sh             # LaunchAgent 自启动脚本
├── start_mlx.sh         # MLX 本地模型服务启动脚本
└── requirements.txt     # Python 依赖
```

---

## 🔧 生产部署

### macOS LaunchAgent

```bash
cp com.douyuan.model-fusion-proxy.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.douyuan.model-fusion-proxy.plist
```

登录自启，崩溃 10 秒自动重启。日志：`~/Library/Logs/model-fusion-proxy.log`

### MLX 本地推理服务

```bash
cp com.douyuan.mlx-model-server.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.douyuan.mlx-model-server.plist
```

### 分布式加速池

在 `config.yaml` 的 `local` 段注册多台 Mac Mini 的 MLX 服务地址，超聚变服务器将自动分发本地推理请求。详见项目文档。

---

## 💰 成本

- **日均**：约 $2（四家包月订阅 + 本地模型零成本）
- **对比云端旗舰模型**：约 1/3 成本
- **本地模型**：零成本（Apple Silicon MLX / Ollama）

---

## 🤖 由 AI Agent 协作构建

- **Claude Code** — 架构设计、核心开发、坑位排查、部署运维
- **Antigravity IDE (Google Gemini Code Assist)** — 性能优化、本地 AST 校验、工具调用适配、双层路由设计
- **Hermes Agent** — 接入配置与实时驱动

> 注：`antigravity-preview-05-2026` 是 Google Gemini 的实验预览模型，可用于 MoA 的 Judge 或 Panel。Antigravity IDE 是开发此项目时使用的编程助手，两者是不同的东西。

---

## 📄 许可证

MIT License
