# agenttest 🧪

**AI 智能体的 pytest —— 用你已经熟悉的方式测试工具调用、输出行为和端到端链路。**

不再凭感觉判断智能体"看起来正常"。agenttest 给 AI 智能体带来真正的软件工程测试能力：断言工具调用、验证输出行为、检测行为回归。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)

[English](./README.md) | **中文**

---

## 问题背景

测试 AI 智能体是工程师面临的最难问题之一：

- **输出不确定** — 同样的输入，每次返回不同的结果
- **工具调用难断言** — 智能体用了哪个工具？参数对不对？顺序对不对？
- **"好不好"很主观** — 没有客观标准判断回答质量
- **没有回归检测** — 模型升级后行为偷偷变了，根本不知道

**agenttest 解决了这些问题。** pytest 风格的断言，专为 LLM 智能体行为设计。零学习成本 —— 已经会 pytest 就会用 agenttest。

---

## 功能特性

- 🧪 **pytest 风格** — 零学习成本，现有测试工程师直接上手
- 🔧 **工具调用断言** — 验证工具名称、参数、调用顺序
- 📸 **快照测试** — 捕获智能体行为，自动检测意外回归
- 🤖 **LLM-as-Judge** — 用 AI 评判输出质量，突破确定性限制
- 🎭 **Mock LLM** — 固定 LLM 返回值，实现确定性测试
- ⚡ **原生 Async 支持** — 无需 `asyncio.run()`，自动处理异步测试
- 📊 **覆盖率报告** — 工具覆盖率、场景覆盖率、边缘案例统计

---

## 安装

```bash
pip install agenttest
```

---

## 快速上手

```python
import agenttest as at

def test_weather_agent():
    result = weather_agent.run("北京今天天气怎么样？")

    # 断言调用了正确的工具
    at.assert_tool_called(result, "weather_api")
    
    # 断言工具参数正确
    at.assert_tool_args(result, "weather_api", location="Beijing")
    
    # 断言输出包含预期内容
    at.assert_output_contains(result, ["温度", "湿度"])
    
    # 断言没有幻觉内容（不捏造天气数据）
    at.assert_no_hallucination(result)
```

运行测试：

```bash
pytest tests/ -v
# ✅ test_weather_agent PASSED (1.2s)
```

---

## 核心断言

### 工具调用断言

```python
# 断言工具被调用
at.assert_tool_called(result, "web_search")

# 断言工具参数
at.assert_tool_args(result, "web_search", query="AI安全", lang="zh")

# 断言调用顺序
at.assert_tool_order(result, ["web_search", "summarizer", "translator"])

# 断言工具调用次数
at.assert_tool_call_count(result, "web_search", min=1, max=3)

# 断言工具未被调用（验证护栏）
at.assert_tool_not_called(result, "delete_files")
```

### 输出内容断言

```python
# 输出包含关键词
at.assert_output_contains(result, ["价格", "库存"])

# 输出不包含违禁内容
at.assert_output_not_contains(result, ["竞品名称", "内部价格"])

# 输出格式断言
at.assert_output_format(result, format="json")
at.assert_output_length(result, max_tokens=500)
at.assert_output_language(result, lang="zh")
```

### LLM-as-Judge 断言

当确定性断言不够用时，用 AI 来评判：

```python
@at.llm_judge(
    criteria="回复应该专业、有同理心，并给出具体解决方案",
    model="gpt-4o-mini",
    threshold=0.8   # Judge 评分 ≥ 0.8 才通过
)
def test_support_tone():
    result = support_agent.run("我很生气，你们的产品坏了！")
    return result.output

# 自动展示 Judge 的评判理由，失败时清晰说明原因
```

---

## 快照测试

捕获智能体行为，自动检测回归：

```python
@at.snapshot_test(model="gpt-4o", temperature=0)
def test_support_agent_behavior():
    result = support_agent.run("我想申请退款")
    return {
        "escalated": result.escalated,
        "tools_used": sorted([t.name for t in result.tool_calls]),
        "sentiment": analyze_sentiment(result.output)
    }

# 首次运行 → 创建快照文件 __snapshots__/test_support_agent_behavior.json
# 后续运行 → 与快照对比，行为变更自动报警
# pytest --update-snapshots → 刷新快照
```

---

## Mock LLM 确定性测试

```python
from agenttest.mocks import MockLLM

mock_llm = MockLLM(responses={
    "2+2等于多少": "答案是4。",
    "*天气*": "北京今天晴，气温22°C，湿度60%",   # 通配符匹配
    "default": "这是默认回复"                       # 兜底
})

def test_math_agent_deterministic():
    with at.use_llm(mock_llm):
        result = math_agent.run("2+2等于多少")
        at.assert_output_contains(result, ["4"])
        # 100% 确定性，无需 API 费用
```

---

## 原生 Async 支持

```python
@at.test_async
async def test_async_pipeline():
    # 自动处理 async/await，无需手动 asyncio.run()
    result = await research_agent.run("查找最新 AI 论文")
    
    at.assert_tool_called(result, "arxiv_search")
    at.assert_output_contains(result, ["摘要", "作者"])
    at.assert_output_length(result, max_tokens=800)
```

---

## CI/CD 集成

```yaml
# .github/workflows/agent-tests.yml
name: Agent Quality Gate
on: [push, pull_request]

jobs:
  test-agents:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: 运行智能体测试
        run: pytest tests/ --agenttest-report=xml --agenttest-html=report.html
        
      - name: 上传测试报告
        uses: actions/upload-artifact@v4
        with:
          name: agent-test-report
          path: report.html
```

---

## 覆盖率报告

```bash
agenttest coverage report

# Tool Coverage: 8/12 工具已测试 (66.7%)
#   ✅ web_search, calculator, file_reader...
#   ❌ image_analyzer, email_sender（缺少测试）
#
# 场景覆盖率: 15 个场景已测试
#   缺失: "退款申请", "投诉升级", "边界情况"
```

---

## 框架适配器

```python
# LangChain 专属断言
from agenttest.adapters import langchain as lc_at

def test_langchain_pipeline():
    result = lc_agent.invoke({"input": "搜索 AI 新闻"})
    lc_at.assert_chain_steps(result, expected=["search", "summarize"])
    lc_at.assert_retriever_called(result, query_contains="AI")

# OpenAI Assistants
from agenttest.adapters import openai as oai_at

async def test_assistant():
    run = await client.beta.threads.runs.create(...)
    oai_at.assert_tool_outputs(run, ["web_search"])
```

---

## 对比同类方案

| 功能 | agenttest | LangSmith | PromptFoo | 自定义测试 |
|------|-----------|-----------|-----------|-----------|
| pytest 风格 | ✅ | ❌ | 部分 | ❌ |
| 工具调用断言 | ✅ | 部分 | ❌ | ❌ |
| 快照测试 | ✅ | ❌ | ❌ | ❌ |
| LLM-as-Judge | ✅ | ✅ | ✅ | ✅ |
| Mock LLM | ✅ | ❌ | ✅ | 部分 |
| Async 原生支持 | ✅ | ✅ | 部分 | ✅ |
| 完全开源免费 | ✅ | 部分 | ✅ | ✅ |

---

## 路线图

- [ ] GitHub Actions 官方 Action（`uses: cdzzy/agenttest-action@v1`）
- [ ] 发布到 GitHub Marketplace
- [ ] 测试覆盖率徽章自动生成
- [ ] PR 评论自动报告测试结果
- [ ] 分布式多智能体链路测试
- [ ] 视频回放：逐步回溯测试失败原因

---

## 示例

```
examples/
  01_tool_assertions.py       # 工具调用断言基础
  02_llm_judge.py             # LLM-as-Judge 质量评估
  03_snapshot_testing.py      # 行为快照与回归检测
  04_mock_llm.py              # Mock LLM 确定性测试
  05_async_agents.py          # 异步智能体测试
  06_langchain_integration.py # LangChain 适配器
```

---

## 许可证

MIT © cdzzy

