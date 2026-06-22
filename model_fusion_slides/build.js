const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "Model Fusion Proxy 模型分工与协同策略";

// Color Palette & Design System (xkcd Whiteboard Sketch Style)
const C = {
  bg: "FFFFFF",
  ink: "1A1A1A",
  accent: "E63946",
  gray: "888888",
  light_gray: "F2F2F2",
  green: "2A9D8F",
  yellow: "FFF3CD"
};

const FONT = "Comic Sans MS";
const FONT_MONO = "Courier New";

// Helpers
function handBorder(slide, x, y, w, h, color = C.ink, width = 2) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: "FFFFFF" },
    line: { color, width }
  });
}

function handCard(slide, x, y, w, h, fillCol = "FFFFFF", borderCol = C.ink, width = 2) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: fillCol },
    line: { color: borderCol, width }
  });
}

function dashArrow(slide, x1, y1, x2, y2, color = C.accent) {
  slide.addShape(pres.shapes.LINE, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width: 2, dashType: "dash", endArrowType: "triangle" }
  });
}

function footer(slide, num, total) {
  slide.addText(`Model Fusion Proxy  |  Page ${num} / ${total}`, {
    x: 0.5, y: 5.25, w: 9.0, h: 0.3,
    fontFace: FONT, fontSize: 9, color: C.gray,
    align: "right", margin: 0
  });
}

function addTitle(slide, text) {
  slide.addText(text, {
    x: 0.5, y: 0.3, w: 9.0, h: 0.5,
    fontFace: FONT, fontSize: 22, bold: true, color: C.ink,
    align: "left", margin: 0
  });
  // Hand-drawn feel red accent line below title
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.85, w: 9.0, h: 0.04,
    fill: { color: C.accent }
  });
}

// ==========================================
// Slide 1: Cover (封面)
// ==========================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };

  // Main Title
  s.addText("Model Fusion Proxy", {
    x: 0.8, y: 1.0, w: 8.4, h: 0.8,
    fontFace: FONT, fontSize: 44, bold: true, color: C.ink,
    align: "left", margin: 0
  });

  // Subtitle
  s.addText("智能路由与模型分工协同策略", {
    x: 0.8, y: 1.8, w: 8.4, h: 0.5,
    fontFace: FONT, fontSize: 22, bold: true, color: C.accent,
    align: "left", margin: 0
  });

  // Outline Card (Big hand-drawn frame)
  handBorder(s, 0.8, 2.6, 8.4, 2.1);
  
  // Content inside the card
  s.addText("混合模型专家 (MoA) 深度实践手册", {
    x: 1.1, y: 2.8, w: 7.8, h: 0.4,
    fontFace: FONT, fontSize: 16, bold: true, color: C.ink,
    align: "left", margin: 0
  });

  const bullets = [
    { text: "• 意图分类器自动路由（Regex 正则匹配 + 本地 Llama 小模型）", options: { fontSize: 12, fontFace: FONT, color: C.ink } },
    { text: "• 多通道 Mixture-of-Agents (MoA) 专家团协同与 Critic 独立审计", options: { fontSize: 12, fontFace: FONT, color: C.ink } },
    { text: "• 多级 Fallback 容灾链与非流式 API 并发竞速降级机制", options: { fontSize: 12, fontFace: FONT, color: C.ink } }
  ];
  s.addText(bullets, {
    x: 1.1, y: 3.3, w: 7.8, h: 1.2,
    margin: 0, lineSpacing: 22
  });

  footer(s, 1, 6);
}

// ==========================================
// Slide 2: Overall Architecture (整体架构)
// ==========================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addTitle(s, "整体架构：各司其职，发挥模型最大能效");

  // Left Column - Core Concept Card
  handCard(s, 0.5, 1.2, 4.2, 3.8, "FFFDF9");
  s.addText("【核心理念】\n没有万能的单体模型，只有最优的分工组合。\n\n1. 两阶段意图分类\n   • 阶段1: Regex 匹配 (O(1) 级，零成本)\n   • 阶段2: 本地小模型 (Llama 1B 语义分类)\n\n2. 差异化模型匹配\n   • coding -> DeepSeek-V4-Flash (极速)\n   • reasoning -> Gemini 3.1 Pro (深度逻辑)\n   • tools -> Gemini 3.5 Flash (高精度/稳定)", {
    x: 0.8, y: 1.4, w: 3.6, h: 3.4,
    fontFace: FONT, fontSize: 11.5, color: C.ink,
    align: "left", lineSpacing: 19
  });

  // Right Column - Diagram Frame
  handBorder(s, 5.0, 1.2, 4.5, 3.8);
  s.addText("【模型中转与路由流程图】", {
    x: 5.2, y: 1.4, w: 4.1, h: 0.3,
    fontFace: FONT, fontSize: 12, bold: true, color: C.ink
  });

  // Diagram Blocks
  handCard(s, 5.5, 1.8, 3.5, 0.5, "FFFFFF", C.ink, 1.5);
  s.addText("User Prompt (用户输入)", {
    x: 5.5, y: 1.8, w: 3.5, h: 0.5,
    fontFace: FONT, fontSize: 11, bold: true, color: C.ink, align: "center"
  });

  dashArrow(s, 7.25, 2.3, 7.25, 2.6);

  handCard(s, 5.5, 2.6, 3.5, 0.6, "FFFFFF", C.ink, 1.5);
  s.addText("Classifier (意图分类)\n[Regex / Llama 1B]", {
    x: 5.5, y: 2.6, w: 3.5, h: 0.6,
    fontFace: FONT, fontSize: 10, color: C.ink, align: "center"
  });

  dashArrow(s, 7.25, 3.2, 6.45, 3.6);
  dashArrow(s, 7.25, 3.2, 8.05, 3.6);

  handCard(s, 5.2, 3.6, 1.9, 0.8, "FFF2F2", C.ink, 1.5);
  s.addText("Single Route\n[DeepSeek-Flash/Gemini]", {
    x: 5.2, y: 3.6, w: 1.9, h: 0.8,
    fontFace: FONT, fontSize: 9.5, color: C.accent, align: "center"
  });

  handCard(s, 7.4, 3.6, 1.9, 0.8, "F2F9F6", C.ink, 1.5);
  s.addText("Model Fusion (MoA)\n[Panelist + Critic + Judge]", {
    x: 7.4, y: 3.6, w: 1.9, h: 0.8,
    fontFace: FONT, fontSize: 9.5, color: C.green, align: "center"
  });

  footer(s, 2, 6);
}

// ==========================================
// Slide 3: Chinese Nuance (中文特化)
// ==========================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addTitle(s, "中文特化：GLM 5.2 领衔的本土语境处理");

  // Left Card: GLM 5.2
  handCard(s, 0.5, 1.2, 4.2, 3.8, "FFFFFF", C.ink, 2);
  s.addText("GLM 5.2 —— 严肃公文与体制内润色", {
    x: 0.7, y: 1.4, w: 3.8, h: 0.4,
    fontFace: FONT, fontSize: 14, bold: true, color: C.accent
  });
  s.addText("• 绝对首选：用于 chinese_nuance 意图场景\n• 本土优势：具备极强的公文措辞对齐、修辞排比分析和历史背景理解能力\n• 典型应用任务：\n  - 政府工作报告润色与排版纠错\n  - 史料分析（如贞观之治等典故解读）\n  - 高阶公文表达与新质生产力语境对齐", {
    x: 0.7, y: 1.9, w: 3.8, h: 2.9,
    fontFace: FONT, fontSize: 11, color: C.ink, lineSpacing: 19
  });

  // Right Card: MiniMax M3
  handCard(s, 5.3, 1.2, 4.2, 3.8, "FFFFFF", C.ink, 2);
  s.addText("MiniMax M3 —— 创意表达与流畅首选", {
    x: 5.5, y: 1.4, w: 3.8, h: 0.4,
    fontFace: FONT, fontSize: 14, bold: true, color: C.green
  });
  s.addText("• 首要容灾：作为中文润色与创作的第一备用\n• 核心优势：在故事创作、营销文案、角色扮演和流畅日常对话中表现极佳\n• 典型应用任务：\n  - 营销文案、广告词脑洞创意生成\n  - 趣味角色扮演与拟人化场景对话\n  - 复杂长文本的多角度提炼与梗概生成", {
    x: 5.5, y: 1.9, w: 3.8, h: 2.9,
    fontFace: FONT, fontSize: 11, color: C.ink, lineSpacing: 19
  });

  // Sync Arrow
  dashArrow(s, 4.75, 3.0, 5.25, 3.0, C.accent);
  s.addText("容灾协同", {
    x: 4.7, y: 2.6, w: 0.6, h: 0.4,
    fontFace: FONT, fontSize: 8.5, color: C.accent, align: "center"
  });

  footer(s, 3, 6);
}

// ==========================================
// Slide 4: MoA Deliberation (MoA 融合)
// ==========================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addTitle(s, "MoA 融合：专家团并行生成与多阶段审计");

  // Step 1 Box
  handCard(s, 0.5, 1.4, 2.6, 3.5, "FFFFFF", C.ink, 1.5);
  s.addText("1. Panel Drafting", {
    x: 0.6, y: 1.6, w: 2.4, h: 0.3,
    fontFace: FONT, fontSize: 13, bold: true, color: C.ink, align: "center"
  });
  s.addText("多模型并行生成草稿：\n• Qwen 9B MLX\n  (本地主创，零成本)\n• DeepSeek-V4-Pro\n  (云端核心分析)\n• GLM 5.2 / MiniMax-M3\n\n【目标】发散思维，全面收集各家所长。", {
    x: 0.6, y: 2.0, w: 2.4, h: 2.7,
    fontFace: FONT, fontSize: 10, color: C.ink, lineSpacing: 16
  });

  dashArrow(s, 3.15, 3.0, 3.65, 3.0, C.accent);

  // Step 2 Box
  handCard(s, 3.7, 1.4, 2.6, 3.5, "FFFDF0", C.ink, 1.5);
  s.addText("2. Critic Review", {
    x: 3.8, y: 1.6, w: 2.4, h: 0.3,
    fontFace: FONT, fontSize: 13, bold: true, color: C.accent, align: "center"
  });
  s.addText("独立评审并寻找Bug：\n• GLM 5.2\n  (公文修辞与中文逻辑)\n• DeepSeek-V4-Pro\n  (代码技术与合理性)\n• 本地 AST 编译器\n  (实时语法自动排查)\n\n【目标】严审初稿，排除隐患。", {
    x: 3.8, y: 2.0, w: 2.4, h: 2.7,
    fontFace: FONT, fontSize: 10, color: C.ink, lineSpacing: 16
  });

  dashArrow(s, 6.35, 3.0, 6.85, 3.0, C.accent);

  // Step 3 Box
  handCard(s, 6.9, 1.4, 2.6, 3.5, "F2FBF6", C.ink, 1.5);
  s.addText("3. Final Synthesis", {
    x: 7.0, y: 1.6, w: 2.4, h: 0.3,
    fontFace: FONT, fontSize: 13, bold: true, color: C.green, align: "center"
  });
  s.addText("法官模型做最终裁决：\n• Gemini 3.1 Pro\n  (扮演主法官 Judge)\n• 对初稿及所有 Critic 反馈进行整合、删改与重构\n• 脱敏 Panel 模型痕迹\n\n【目标】融汇贯通，产出完美解答。", {
    x: 7.0, y: 2.0, w: 2.4, h: 2.7,
    fontFace: FONT, fontSize: 10, color: C.ink, lineSpacing: 16
  });

  footer(s, 4, 6);
}

// ==========================================
// Slide 5: High Availability (高可用容灾)
// ==========================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addTitle(s, "高可用容灾：并发竞速重试与秒级降级保障");

  // Left strategy card
  handCard(s, 0.5, 1.3, 4.2, 2.5, "FFFFFF", C.ink, 1.5);
  s.addText("非流式 (Batch) ── 并发竞速 (Racing)", {
    x: 0.7, y: 1.5, w: 3.8, h: 0.4,
    fontFace: FONT, fontSize: 13, bold: true, color: C.accent
  });
  s.addText("• 核心逻辑：发起调用时，若主模型出现 HTTP 429 限流或 5xx 故障，系统会并行唤醒所有备用 Fallback 模型。\n• 胜者通吃：最快返回的成功结果将被立即采纳，其余慢速响应被自动 cancel，将降级时间压降到毫秒级。", {
    x: 0.7, y: 1.9, w: 3.8, h: 1.8,
    fontFace: FONT, fontSize: 10.5, color: C.ink, lineSpacing: 18
  });

  // Right strategy card
  handCard(s, 5.3, 1.3, 4.2, 2.5, "FFFFFF", C.ink, 1.5);
  s.addText("流式 (Streaming) ── 顺序 Fallback", {
    x: 5.5, y: 1.5, w: 3.8, h: 0.4,
    fontFace: FONT, fontSize: 13, bold: true, color: C.green
  });
  s.addText("• 核心逻辑：由于 SSE 流式传输一旦吐出首包即无法更换模型，容灾判断被前置到“首包前阶段”。\n• 顺延重试：当主模型连接阶段超时或报错，系统会立即平滑切换到 Fallback 模型，避免流式响应异常中断。", {
    x: 5.5, y: 1.9, w: 3.8, h: 1.8,
    fontFace: FONT, fontSize: 10.5, color: C.ink, lineSpacing: 18
  });

  // Bottom indicator callout
  handCard(s, 0.5, 4.0, 9.0, 0.9, "FFF3CD", C.ink, 2);
  s.addText("【高可用容灾核心指标】\n系统吞吐稳定性达 99.9%  ·  MoA 平均排队等待耗时减少 60%  ·  端到端故障自愈率 100%", {
    x: 0.7, y: 4.05, w: 8.6, h: 0.8,
    fontFace: FONT, fontSize: 12.5, bold: true, color: C.ink, align: "center", lineSpacing: 18
  });

  footer(s, 5, 6);
}

// ==========================================
// Slide 6: Summary (总结)
// ==========================================
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  addTitle(s, "总结：合理分工，以低成本打造最强质效比");

  // Center Hero Slogan
  s.addText("“ 用最适合的模型，做最擅长的事情 ”", {
    x: 1.0, y: 1.4, w: 8.0, h: 0.6,
    fontFace: FONT, fontSize: 24, bold: true, color: C.accent, align: "center"
  });

  // Takeaway Card 1
  handCard(s, 0.5, 2.3, 2.8, 2.6, "FFFFFF", C.ink, 1.5);
  s.addText("【极致降本】", {
    x: 0.6, y: 2.5, w: 2.6, h: 0.3,
    fontFace: FONT, fontSize: 13, bold: true, color: C.accent, align: "center"
  });
  s.addText("本地大模型 (Qwen 9B MLX) 承担了 60% 的 MoA Panel 任务，云端仅做 Judge 裁判，比全云端 MoA 节省超 60% 的 API 开销。", {
    x: 0.6, y: 2.9, w: 2.6, h: 1.9,
    fontFace: FONT, fontSize: 10, color: C.ink, lineSpacing: 17
  });

  // Takeaway Card 2
  handCard(s, 3.6, 2.3, 2.8, 2.6, "FFFFFF", C.ink, 1.5);
  s.addText("【极致体验】", {
    x: 3.7, y: 2.5, w: 2.6, h: 0.3,
    fontFace: FONT, fontSize: 13, bold: true, color: C.green, align: "center"
  });
  s.addText("Regex 毫秒级意图匹配，非流式智能竞速；通过优化 SSE 流式响应，首包输出耗时大幅压缩，无感切换备用模型。", {
    x: 3.7, y: 2.9, w: 2.6, h: 1.9,
    fontFace: FONT, fontSize: 10, color: C.ink, lineSpacing: 17
  });

  // Takeaway Card 3
  handCard(s, 6.7, 2.3, 2.8, 2.6, "FFFFFF", C.ink, 1.5);
  s.addText("【高质对齐】", {
    x: 6.8, y: 2.5, w: 2.6, h: 0.3,
    fontFace: FONT, fontSize: 13, bold: true, color: C.ink, align: "center"
  });
  s.addText("对于严肃公文，由国产最强中文语言模型 GLM 5.2 主导润色；对于逻辑推理与代码，交由 Gemini 3.1 Pro 与 DeepSeek 精密处理。", {
    x: 6.8, y: 2.9, w: 2.6, h: 1.9,
    fontFace: FONT, fontSize: 10, color: C.ink, lineSpacing: 17
  });

  footer(s, 6, 6);
}

// ==========================================
// Write out Presentation
// ==========================================
pres.writeFile({ fileName: "model_routing_presentation.pptx" })
  .then(fn => console.log("✅ Slide presentation generated successfully: " + fn))
  .catch(err => console.error("❌ Error generating presentation:", err));
