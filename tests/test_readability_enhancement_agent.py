from __future__ import annotations

from core.agents.ai_driven_readability_enhancement_agent import (
    AIDrivenReadabilityEnhancementAgent,
)


def test_readability_includes_info_issues():
    agent = AIDrivenReadabilityEnhancementAgent()
    report_data = {
        "file": "example.c",
        "status": "completed",
        "issue_count": 1,
        "severity_stats": {"info": 1},
        "analysis_types": ["performance_analysis"],
        "issues": [
            {
                "source": "performance_bottleneck",
                "severity": "info",
                "line": 10,
                "description": "Info-level performance issue",
            }
        ],
    }

    markdown = agent._generate_markdown_summary(report_data, "consolidated")

    assert "🔵" in markdown
    assert "Info-level performance issue" in markdown
    assert "第 10 行" in markdown
