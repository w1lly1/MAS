import unittest

from utils.prompt_budgeting import (
    budget_text_segments,
    estimate_token_count,
    prepare_generation_prompt,
    semantic_split_text,
    semantic_truncate_text,
)


class DummyTokenizer:
    def __init__(self, model_max_length: int = 32):
        self.model_max_length = model_max_length
        self._token_to_id = {}
        self._id_to_token = {}
        self._next_id = 1

    def _tokenize(self, text: str):
        text = (text or "").replace("\r\n", "\n")
        tokens = []
        for part in text.replace("\n", " \n ").split():
            if part == "\n":
                tokens.append(part)
            else:
                tokens.append(part)
        return tokens

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None, return_tensors=None):
        tokens = self._tokenize(text)
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        ids = []
        for token in tokens:
            if token not in self._token_to_id:
                self._token_to_id[token] = self._next_id
                self._id_to_token[self._next_id] = token
                self._next_id += 1
            ids.append(self._token_to_id[token])
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        tokens = [self._id_to_token.get(token_id, "") for token_id in ids]
        return " ".join(token for token in tokens if token)


class PromptBudgetingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer(model_max_length=24)

    def test_semantic_split_text_keeps_code_blocks(self):
        code = (
            "def first():\n"
            "    return 1\n\n"
            "def second():\n"
            "    return 2\n"
        )

        chunks = semantic_split_text(code)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("def first()", chunks[0])
        self.assertIn("return 1", chunks[0])
        self.assertIn("def second()", chunks[1])

    def test_semantic_truncate_text_preserves_full_first_block(self):
        code = (
            "def first():\n"
            "    return 1\n\n"
            "def second():\n"
            "    return 2\n"
        )

        truncated = semantic_truncate_text(self.tokenizer, code, max_tokens=6)

        self.assertIn("def first()", truncated)
        self.assertIn("return 1", truncated)
        self.assertNotIn("def second()", truncated)

    def test_budget_text_segments_respects_total_budget(self):
        segments = ["system instructions one two", "user task alpha beta gamma"]

        kept = budget_text_segments(self.tokenizer, segments, max_tokens=6)

        self.assertGreaterEqual(len(kept), 1)
        self.assertLessEqual(sum(estimate_token_count(self.tokenizer, item) for item in kept), 6)

    def test_prepare_generation_prompt_reserves_output_tokens(self):
        prompt = "one two three four five six seven eight nine ten"

        safe_prompt, input_budget, requested_new = prepare_generation_prompt(
            self.tokenizer,
            prompt,
            max_new_tokens=8,
            fallback_model_max=24,
            safety_margin=4,
        )

        self.assertLessEqual(estimate_token_count(self.tokenizer, safe_prompt), input_budget)
        self.assertGreaterEqual(requested_new, 16)
        self.assertLessEqual(input_budget + requested_new + 4, 24)


if __name__ == "__main__":
    unittest.main()