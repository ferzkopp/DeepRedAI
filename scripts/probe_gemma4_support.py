#!/usr/bin/env python3
"""Read-only diagnostic: can this container load and train Gemma 4?

Implements DeepRed-Phase4-Plan.md P1.1. Every check runs and reports
independently, so one failure does not mask the rest, and the JSON verdict is
what the P1.4 gate and the p4 driver preflight read. Nothing here writes to a
model directory or mutates state.

Usage:
  /opt/venv/bin/python3 scripts/probe_gemma4_support.py \\
      --model /mnt/data/models/gemma-4-12b-it \\
      --output /mnt/data/evaluations/phase4/gemma4_support.json
"""

import argparse
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# The container ships a gfx1151 ROCm build that no public index can reinstall.
# Compared as a prefix: the nightly date suffix moves without meaning, the
# major/ROCm portion moving means the transformers upgrade dragged torch with
# it and the GPU is no longer the one every measurement was taken on.
EXPECTED_TORCH_PREFIX = '2.12.0a0+rocm7.12'

CONFIG_KEY = 'gemma4_unified'

# Gemma 4 channel markers, from the model's own chat_template.jinja.
THOUGHT_OPEN = '<|channel>thought'
THOUGHT_CLOSE = '<channel|>'
THINK_TOKEN = '<|think|>'
SYSTEM_TURN = '<|turn>system'
USER_TURN = '<|turn>user'

# Deliberately unlikely to collide with template boilerplate.
SYSTEM_TEXT = 'SENTINEL_SYSTEM_7f3a you are a chess computer.'
USER_TEXT = 'SENTINEL_USER_91cd what is two plus two?'


class Report:
    """Collects independent check results into a JSON verdict."""

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.checks = {}

    def record(self, name, passed, **details):
        self.checks[name] = {'status': 'pass' if passed else 'fail', **details}
        mark = 'PASS' if passed else 'FAIL'
        print(f'  [{mark}] {name}')
        for key, value in details.items():
            print(f'         {key}: {_short(value)}')
        return passed

    def skip(self, name, reason):
        self.checks[name] = {'status': 'skip', 'reason': reason}
        print(f'  [SKIP] {name}')
        print(f'         reason: {reason}')
        return False

    def failed_from(self, exc):
        return {'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(limit=5)}

    def verdict(self):
        statuses = [c['status'] for c in self.checks.values()]
        return {
            'schema_version': 1,
            'probe': 'gemma4_support',
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'model_path': self.model_path,
            'python': platform.python_version(),
            'passed': all(s == 'pass' for s in statuses),
            'counts': {
                'pass': statuses.count('pass'),
                'fail': statuses.count('fail'),
                'skip': statuses.count('skip'),
            },
            'checks': self.checks,
        }


def _short(value, limit=200):
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = text.replace('\n', '\\n')
    return text if len(text) <= limit else text[:limit] + '...'


def check_config_mapping(report):
    try:
        from transformers.models.auto.configuration_auto import (
            CONFIG_MAPPING_NAMES,
        )
        import transformers
        present = CONFIG_KEY in CONFIG_MAPPING_NAMES
        return report.record(
            'config_mapping_has_gemma4', present,
            transformers_version=transformers.__version__,
            expected_key=CONFIG_KEY,
            gemma_keys=sorted(k for k in CONFIG_MAPPING_NAMES if 'gemma' in k))
    except Exception as exc:
        return report.record('config_mapping_has_gemma4', False,
                             **report.failed_from(exc))


def check_torch_version(report):
    try:
        import torch
        version = torch.__version__
        ok = version.startswith(EXPECTED_TORCH_PREFIX)
        return report.record(
            'torch_unchanged', ok,
            torch_version=version,
            expected_prefix=EXPECTED_TORCH_PREFIX,
            cuda_available=torch.cuda.is_available())
    except Exception as exc:
        return report.record('torch_unchanged', False,
                             **report.failed_from(exc))


def check_config_resolves(report, model_path):
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        text_config = getattr(config, 'text_config', None)
        return report.record(
            'config_resolves_locally', config.model_type == CONFIG_KEY,
            model_type=config.model_type,
            architectures=getattr(config, 'architectures', None),
            text_model_type=getattr(text_config, 'model_type', None),
            text_vocab_size=getattr(text_config, 'vocab_size', None),
            text_num_hidden_layers=getattr(
                text_config, 'num_hidden_layers', None))
    except Exception as exc:
        return report.record('config_resolves_locally', False,
                             **report.failed_from(exc))


def load_tokenizer(report, model_path):
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        report.record('tokenizer_loads', True,
                      tokenizer_class=type(tokenizer).__name__,
                      has_chat_template=bool(
                          getattr(tokenizer, 'chat_template', None)))
        return tokenizer
    except Exception as exc:
        report.record('tokenizer_loads', False, **report.failed_from(exc))
        return None


def check_system_channel(report, tokenizer):
    """A system message must land in its own system turn, not the user turn."""
    if tokenizer is None:
        return report.skip('system_message_in_system_channel',
                           'tokenizer did not load')
    try:
        rendered = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_TEXT},
             {'role': 'user', 'content': USER_TEXT}],
            tokenize=False, add_generation_prompt=True)

        reasons = []
        if SYSTEM_TURN not in rendered:
            reasons.append(f'no {SYSTEM_TURN} marker')
        if SYSTEM_TEXT not in rendered:
            reasons.append('system text absent from prompt')
        if USER_TURN not in rendered:
            reasons.append(f'no {USER_TURN} marker')
        if not reasons:
            system_at = rendered.index(SYSTEM_TEXT)
            system_turn_at = rendered.index(SYSTEM_TURN)
            user_turn_at = rendered.index(USER_TURN)
            if not system_turn_at < system_at < user_turn_at:
                reasons.append(
                    'system text is not inside the system turn — it was '
                    'folded into a later turn')
            if rendered.index(USER_TEXT) < user_turn_at:
                reasons.append('user text precedes its own turn marker')

        return report.record(
            'system_message_in_system_channel', not reasons,
            failure_reasons=reasons,
            rendered_prompt=rendered)
    except Exception as exc:
        return report.record('system_message_in_system_channel', False,
                             **report.failed_from(exc))


def check_thinking_default_off(report, tokenizer):
    """Thinking must be opt-in.

    Gemma 4 suppresses it by pre-filling an *empty* thought block, so the test
    is that the block is empty and that <|think|> only appears when asked for
    — not that the marker is absent, which it never is.
    """
    if tokenizer is None:
        return report.skip('thinking_off_by_default', 'tokenizer did not load')
    try:
        messages = [{'role': 'user', 'content': USER_TEXT}]
        default = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enabled = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True)

        empty_block = f'{THOUGHT_OPEN}\n{THOUGHT_CLOSE}'
        reasons = []
        if THINK_TOKEN in default:
            reasons.append(f'{THINK_TOKEN} present without being requested')
        if not default.rstrip().endswith(empty_block):
            reasons.append(
                'generation prompt does not end with an empty thought block, '
                'so thinking is not suppressed by default')
        if THINK_TOKEN not in enabled:
            reasons.append(
                f'{THINK_TOKEN} absent even with enable_thinking=True, so the '
                'opt-in switch is not the documented one')

        return report.record(
            'thinking_off_by_default', not reasons,
            failure_reasons=reasons,
            default_prompt_tail=default[-120:],
            enabled_prompt_head=enabled[:120])
    except Exception as exc:
        return report.record('thinking_off_by_default', False,
                             **report.failed_from(exc))


def load_model(report, model_path, device):
    """Load weights, recording which auto class accepted them."""
    import torch
    from transformers import AutoModel, AutoModelForCausalLM

    attempts = []
    for label, factory in (('AutoModelForCausalLM', AutoModelForCausalLM),
                           ('AutoModel', AutoModel)):
        try:
            model = factory.from_pretrained(
                model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True)
            model.to(device)
            model.eval()
            report.record('weights_load', True,
                          auto_class=label,
                          model_class=type(model).__name__,
                          device=device,
                          failed_attempts=attempts)
            return model, label
        except Exception as exc:
            attempts.append({'auto_class': label,
                             'error': f'{type(exc).__name__}: {exc}'})
    report.record('weights_load', False, failed_attempts=attempts)
    return None, None


def resolve_text_module(model):
    """Return (module, dotted path) for the text tower, or (model, '')."""
    for path in ('language_model', 'model.language_model', 'text_model',
                 'model.text_model', 'model'):
        target = model
        try:
            for part in path.split('.'):
                target = getattr(target, part)
        except AttributeError:
            continue
        if callable(target):
            return target, path
    return model, ''


def check_text_forward(report, model, tokenizer, device):
    """A text-only forward pass must work, and we must know via which path."""
    if model is None or tokenizer is None:
        return report.skip('text_only_forward',
                           'model or tokenizer did not load')
    import torch
    ids = tokenizer(USER_TEXT, return_tensors='pt').to(device)

    attempts = []
    candidates = [(model, '', type(model).__name__)]
    text_module, text_path = resolve_text_module(model)
    if text_path:
        candidates.append((text_module, text_path, type(text_module).__name__))

    for module, path, class_name in candidates:
        try:
            with torch.no_grad():
                out = module(input_ids=ids['input_ids'],
                             attention_mask=ids.get('attention_mask'))
            logits = getattr(out, 'logits', None)
            hidden = getattr(out, 'last_hidden_state', None)
            tensor = logits if logits is not None else hidden
            if tensor is None:
                raise RuntimeError('output carried neither logits nor '
                                   'last_hidden_state')
            return report.record(
                'text_only_forward', True,
                forward_path=path or 'top_level',
                module_class=class_name,
                output_kind='logits' if logits is not None
                            else 'last_hidden_state',
                output_shape=list(tensor.shape),
                failed_attempts=attempts)
        except Exception as exc:
            attempts.append({'forward_path': path or 'top_level',
                             'module_class': class_name,
                             'error': f'{type(exc).__name__}: {exc}'})
    return report.record('text_only_forward', False, failed_attempts=attempts)


def check_greedy_has_no_thought(report, model, tokenizer, device, max_new):
    """Greedy decoding without <|think|> must not emit a populated thought."""
    if model is None or tokenizer is None:
        return report.skip('greedy_emits_no_thinking',
                           'model or tokenizer did not load')
    if not hasattr(model, 'generate'):
        return report.skip('greedy_emits_no_thinking',
                           f'{type(model).__name__} has no generate()')
    import torch
    try:
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': USER_TEXT}],
            tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors='pt',
                        add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new,
                                 do_sample=False)
        completion = tokenizer.decode(
            out[0][ids['input_ids'].shape[-1]:], skip_special_tokens=False)

        reasons = []
        if THOUGHT_OPEN in completion:
            reasons.append('completion opened a thought channel')
        if THINK_TOKEN in completion:
            reasons.append(f'completion emitted {THINK_TOKEN}')
        return report.record(
            'greedy_emits_no_thinking', not reasons,
            failure_reasons=reasons,
            max_new_tokens=max_new,
            completion=completion)
    except Exception as exc:
        return report.record('greedy_emits_no_thinking', False,
                             **report.failed_from(exc))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True,
                        help='Local Gemma 4 weights directory.')
    parser.add_argument('--output', required=True,
                        help='Where to write the JSON verdict.')
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'),
                        help='Device for the load and forward checks.')
    parser.add_argument('--max-new-tokens', type=int, default=32,
                        help='Budget for the greedy thinking check.')
    parser.add_argument('--skip-weights', action='store_true',
                        help='Run only the checks that need no weights.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model_path = Path(args.model)
    if not model_path.is_dir():
        print(f'ERROR: not a directory: {model_path}', file=sys.stderr)
        return 2

    report = Report(model_path)
    print(f'probing Gemma 4 support against {model_path}')

    check_config_mapping(report)
    check_torch_version(report)
    check_config_resolves(report, model_path)
    tokenizer = load_tokenizer(report, model_path)
    check_system_channel(report, tokenizer)
    check_thinking_default_off(report, tokenizer)

    if args.skip_weights:
        for name in ('weights_load', 'text_only_forward',
                     'greedy_emits_no_thinking'):
            report.skip(name, '--skip-weights')
    else:
        model, _ = load_model(report, model_path, args.device)
        check_text_forward(report, model, tokenizer, args.device)
        check_greedy_has_no_thought(report, model, tokenizer, args.device,
                                    args.max_new_tokens)

    verdict = report.verdict()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verdict, indent=2) + '\n')

    counts = verdict['counts']
    print(f"\n{counts['pass']} passed, {counts['fail']} failed, "
          f"{counts['skip']} skipped -> {output}")
    return 0 if verdict['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
