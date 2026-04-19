import json
import unittest
from unittest.mock import AsyncMock, patch

from core.agents.static_scan_agent import StaticCodeScanAgent


class TestStaticScanAgentRunners(unittest.IsolatedAsyncioTestCase):
    """测试静态扫描新增runner的核心解析逻辑。"""

    def setUp(self):
        self.agent = StaticCodeScanAgent()
        self.agent.agent_config = {
            "semgrep_timeout": 5,
            "cppcheck_timeout": 5,
            "clang_tidy_timeout": 5,
            "spotbugs_timeout": 5,
            "tool_check_timeout": 1,
        }

    def test_parse_cppcheck_xml(self):
        xml_payload = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<results version=\"2\">
  <errors>
    <error id=\"arrayIndexOutOfBounds\" severity=\"error\" msg=\"Array index out of bounds\">
      <location file=\"/tmp/demo.c\" line=\"42\"/>
    </error>
  </errors>
</results>
"""
        issues = self.agent._parse_cppcheck_xml(xml_payload)
        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue["tool"], "cppcheck")
        self.assertEqual(issue["line"], 42)
        self.assertEqual(issue["severity"], "high")
        self.assertEqual(issue["rule_id"], "arrayIndexOutOfBounds")

    async def test_run_semgrep_maps_result(self):
        semgrep_json = {
            "results": [
                {
                    "check_id": "test.rule",
                    "start": {"line": 7, "col": 3},
                    "extra": {
                        "message": "Potential risky operation",
                        "severity": "WARNING",
                        "metadata": {"cwe": "CWE-119"},
                    },
                }
            ]
        }

        mock_result = type(
            "Res",
            (),
            {
                "stdout": json.dumps(semgrep_json),
                "stderr": "",
                "returncode": 0,
            },
        )()

        with patch.object(self.agent, "_run_external_tool", new=AsyncMock(return_value=mock_result)):
            issues = await self.agent._run_semgrep("int main(){return 0;}", "", "c")

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue["tool"], "semgrep")
        self.assertEqual(issue["line"], 7)
        self.assertEqual(issue["severity"], "medium")
        self.assertEqual(issue["rule_id"], "test.rule")
        self.assertEqual(issue["cwe"], "CWE-119")

    def test_parse_clang_tidy_output(self):
        output = "/tmp/demo.c:10:5: warning: unsafe call detected [cert-msc30-c]\n"
        issues = self.agent._parse_clang_tidy_output(output)
        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue["tool"], "clang-tidy")
        self.assertEqual(issue["line"], 10)
        self.assertEqual(issue["severity"], "medium")
        self.assertEqual(issue["rule_id"], "cert-msc30-c")

    def test_parse_spotbugs_xml(self):
        xml_payload = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<BugCollection>
  <BugInstance type=\"NP_NULL_ON_SOME_PATH\" priority=\"2\">
    <ShortMessage>Possible null pointer dereference</ShortMessage>
    <LongMessage>Possible null pointer dereference in method foo</LongMessage>
    <Class classname=\"Demo\">
      <SourceLine sourcepath=\"src/Demo.java\" start=\"12\" end=\"12\"/>
    </Class>
  </BugInstance>
</BugCollection>
"""
        issues = self.agent._parse_spotbugs_xml(xml_payload)
        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue["tool"], "spotbugs")
        self.assertEqual(issue["line"], 12)
        self.assertEqual(issue["severity"], "medium")
        self.assertEqual(issue["rule_id"], "NP_NULL_ON_SOME_PATH")

    async def test_run_spotbugs_without_targets_returns_empty(self):
        issues = await self.agent._run_spotbugs("public class Demo {}", "")
        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
