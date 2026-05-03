import importlib
import os
import sys
import types
from transformers import AutoTokenizer


class FakeTokenizer:
    def __init__(self):
        self.eos_token = ''
        self.eos_token_id = 50256
        self.model_max_length = 1024

    def encode(self, text, add_special_tokens=True):
        # naive tokenization: 1 token per 4 chars
        n = max(1, len(text) // 4)
        return list(range(n))

    def decode(self, ids, skip_special_tokens=True):
        return ''.join('A' for _ in range(len(ids)))

# Prepare fake package entries to avoid executing core/__init__.py
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
core_path = os.path.join(proj_root, 'core')
agents_path = os.path.join(core_path, 'agents')
sys.modules.setdefault('core', types.ModuleType('core'))
sys.modules['core'].__path__ = [core_path]
sys.modules.setdefault('core.agents', types.ModuleType('core.agents'))
sys.modules['core.agents'].__path__ = [agents_path]

# Now import the agent module via importlib using package context
mod = importlib.import_module('core.agents.ai_driven_code_quality_agent')
AIDrivenCodeQualityAgent = getattr(mod, 'AIDrivenCodeQualityAgent')

class FakePipeline:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        class M:
            pass
        self.model = M()
        # provide a config with n_positions if possible
        try:
            self.model.config = type('C', (), {'n_positions': getattr(tokenizer, 'model_max_length', 1024)})()
        except Exception:
            self.model.config = type('C', (), {'n_positions': 1024})()
    def __call__(self, prompt, **kwargs):
        # simulate a generation response
        return [{'generated_text': '<<FAKE_GENERATION>>: ' + (prompt[:200] + '...' if len(prompt) > 200 else prompt)}]


def run_diag():
    agent = AIDrivenCodeQualityAgent()
    # load a tokenizer (may download if not present)
    try:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    except Exception as e:
        print('Tokenizer load failed:', e)
        return
    fake_pipeline = FakePipeline(tokenizer)
    agent.text_generation_model = fake_pipeline

    long_prompt = 'A' * 8000  # intentionally long
    print('Calling _safe_generate with long prompt...')
    out = agent._safe_generate(long_prompt, max_new_tokens=128, temperature=0.7)
    print('Result:', out)

if __name__ == '__main__':
    run_diag()
