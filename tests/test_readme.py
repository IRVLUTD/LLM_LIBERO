"""README stays in sync with the prompt notes the model actually receives."""
import re
import unittest
from pathlib import Path

from inspect_robots_libero.embodiment import BASE_DOCS

README = Path(__file__).resolve().parents[1] / "README.md"


class ReadmeTests(unittest.TestCase):
    def test_readme_quotes_base_docs(self):
        # The README wraps lines for display, so compare with whitespace collapsed.
        text = README.read_text()
        section = text.split("## What the model is told", 1)[1]
        quoted = re.search(r"```text\n(.*?)```", section, re.S).group(1)
        self.assertEqual(" ".join(quoted.split()), " ".join(BASE_DOCS.split()))


if __name__ == "__main__":
    unittest.main()
