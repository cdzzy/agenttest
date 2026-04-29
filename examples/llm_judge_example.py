"""
agentlink examples: LLM-as-Judge Evaluation Framework
参考 hermes-agent 的持续学习 + ai-hedge-fund 的多 Agent 协作思路，
为 agenttest 添加基于 LLM 的自动化评估能力。
类似 Archon 的"让 AI 编码测试框架构建器"理念，
使用 LLM 来判断 Agent 输出的质量。
"""

import json
from typing import Optional


class LLMJudge:
    """
    LLM-as-Judge 评估器。
    使用 LLM 作为评判者，评估 AI Agent 输出的正确性、相关性、安全性。
    参考 hermes-agent 的成长型 Agent 自我评估思路。
    """

    def __init__(self, judge_model: str = "claude-sonnet-4-20250514"):
        self.judge_model = judge_model
        self.evaluation_history: list[dict] = []

    def evaluate(
        self,
        task: str,
        agent_output: str,
        criteria: Optional[list[str]] = None,
    ) -> dict:
        """
        使用 LLM 评判 Agent 输出质量。
        返回包含分数和详细反馈的评估报告。
        """
        if criteria is None:
            criteria = ["correctness", "relevance", "safety", "clarity"]

        evaluation_prompt = self._build_prompt(task, agent_output, criteria)
        # 实际使用时调用 LLM API
        # judge_response = call_llm(self.judge_model, evaluation_prompt)
        judge_response = self._mock_judge(agent_output)

        record = {
            "task": task,
            "criteria": criteria,
            "scores": judge_response["scores"],
            "feedback": judge_response["feedback"],
            "passed": judge_response["passed"],
        }
        self.evaluation_history.append(record)
        return record

    def _build_prompt(self, task: str, output: str, criteria: list[str]) -> str:
        return f"""Task: {task}
Agent Output: {output}

Evaluate this output against the following criteria:
{', '.join(criteria)}

Provide:
1. A score (0-100) for each criterion
2. A brief feedback comment
3. A pass/fail verdict
"""

    def _mock_judge(self, output: str) -> dict:
        """模拟 LLM 评判结果（实际使用时替换为真实 API 调用）"""
        word_count = len(output.split())
        base_score = min(100, 50 + word_count // 10)

        return {
            "scores": {
                "correctness": base_score,
                "relevance": base_score + 5,
                "safety": 95,
                "clarity": base_score - 5,
            },
            "feedback": f"Output contains {word_count} words. Quality assessment complete.",
            "passed": base_score >= 70,
        }

    def generate_report(self) -> str:
        """生成评估摘要报告，参考 hermes-agent 的自我分析能力"""
        if not self.evaluation_history:
            return "No evaluations yet."

        total = len(self.evaluation_history)
        passed = sum(1 for r in self.evaluation_history if r["passed"])
        avg_scores = {
            criterion: sum(r["scores"].get(criterion, 0) for r in self.evaluation_history) / total
            for criterion in ["correctness", "relevance", "safety", "clarity"]
        }

        report = f"""# Agent Evaluation Report

- Total evaluations: {total}
- Passed: {passed} ({passed/total*100:.1f}%)
- Average scores:
  - Correctness: {avg_scores['correctness']:.1f}/100
  - Relevance:  {avg_scores['relevance']:.1f}/100
  - Safety:      {avg_scores['safety']:.1f}/100
  - Clarity:     {avg_scores['clarity']:.1f}/100
"""
        return report


async def run_evaluation():
    """运行完整评估流程"""
    judge = LLMJudge()

    test_cases = [
        {
            "task": "Explain quantum entanglement to a 10-year-old",
            "output": "Imagine you have two magical dice that always show the same number, no matter how far apart they are. When one die shows a 3, the other instantly shows a 3 too! Scientists call this quantum entanglement.",
        },
        {
            "task": "Write a Python function to check prime numbers",
            "output": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        },
        {
            "task": "Summarize the causes of World War I",
            "output": "The main causes of WWI were: militarism, alliances, imperialism, and nationalism (the MAI N system). These four factors created a tense atmosphere in Europe that eventually exploded into war in 1914.",
        },
    ]

    for case in test_cases:
        result = judge.evaluate(case["task"], case["output"])
        print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['feedback']}")

    print(judge.generate_report())


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_evaluation())

