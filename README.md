# Model Fusion Proxy (Fable 5 旗舰级基准)

## 简介 (Introduction)

本项目是一个对标 Claude 最新旗舰模型 Fable 5 的 AI 模型融合网关代理 (Model Fusion Proxy)。它通过 MoA (Mixture of Agents) 架构，智能调用并融合多个本地和云端大语言模型，旨在以极低的成本提供超越 Fable 5 的综合智能表现。系统支持灵活的请求分类（编程、逻辑推理、创意写作、中文语境）并自动选择最优模型或模型组合。

## 核心特性 (Features)
- **多代理融合 (MoA)**：包含 Panelist 拟稿、Judge 综合、Critic 审查与最终输出优化。
- **本地+云端混合架构 (Hybrid Mode)**：结合轻量级本地模型 (Llama 3.2 1B, Gemma 2B 等) 与云端模型 (Gemini 2.5 Pro, DeepSeek V4 Pro, GLM 5.2)，规避 OOM 问题的同时优化成本和延迟。
- **自动失败降级 (Fallback)**：在遇到 Rate Limit (如 429) 或网络超时时，系统会自动无缝切换至备用模型。
- **智能意图路由 (Intent Routing)**：基于 Regex 和 Local LLM 分析用户意图，进行领域细分路由。
- **高性能基准 (Fable 5 对标)**：综合测试得分已超越 Claude Fable 5，综合准确率高，且大幅节约使用成本。

## 环境依赖 (Requirements)
- Python 3.10+
- [Ollama](https://ollama.ai/) (用于本地模型运行)
- 第三方 API Keys（需配置在环境变量或配置文件中，如 `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `GLM_API_KEY`）

### 安装依赖包
在项目根目录下运行：
```bash
pip install -r requirements.txt
```

## 配置指南 (Configuration)
配置文件位于 `config.yaml`。你可以根据机器硬件水平及个人偏好调整。
- **Ollama 本地模型配置**: `llama3.2:1b`, `gemma:2b`
- **融合策略配置 (`fusion.strategy`)**:
  - `hybrid`: 优先本地生成初稿，云端模型总结和审查（最具性价比）。
  - `cloud`: 纯云端模型互辩（最高智商）。
  - `local_test`: 纯本地模型互辩（最节省 API 成本）。

确保 Ollama 在后台运行相关模型：
```bash
ollama run llama3.2:1b
ollama run gemma:2b
```

## 使用说明 (Usage)
### 1. 启动 Proxy Server
运行代理服务器，默认监听本地 `http://localhost:8000`：
```bash
python server.py
```

### 2. 接口调用
该服务兼容 OpenAI Chat Completions API，可以通过任何支持配置 API 基础地址的客户端 (例如 NextChat, Chatbox, Cursor) 调用。
- **Base URL**: `http://localhost:8000/v1`
- **API Key**: 任意字符串
- **Model Name**: 可通过 `router` 进行自动路由，或显式请求特定模型。

示例 CURL 请求：
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "router",
    "messages": [{"role": "user", "content": "请用Python写一个快速排序算法"}]
  }'
```

### 3. 性能基准测试
如果要运行对标 Claude Fable 5 的性能测试脚本，请执行：
```bash
python full_benchmark.py
```
测试将在终端输出详尽的综合评分与对比结果。

## 最近更新记录
- **超时优化**: 延长了 MoA Panelists 并发容忍度 (25s -> 60s)，减少超时掉队问题。
- **本地服务瘦身**: 剔除高显存占用的 MLX 常驻进程，全面转为按需调用的 Ollama。
- **叙事升级**: 核心对标从 Opus 4.8 全面升级至 Fable 5，激励极致性能体验。
