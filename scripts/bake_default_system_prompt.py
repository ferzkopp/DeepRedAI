#!/usr/bin/env python3
"""Bake a default system prompt into a GGUF's chat template.

Every Gemma 4 run so far behaves era-natively only when a system turn is
present (p4-v1..p4-v4, no-system era-native 0-8%). The release therefore
carries its own: when a caller sends no system message, the template inserts
the trained prompt; a caller-supplied system message still replaces it.

The tensors are copied unchanged by gguf-py's gguf_new_metadata.py; only
``tokenizer.chat_template`` differs.

Usage:
    python3 scripts/bake_default_system_prompt.py \\
        --input deepred-p4v5-050-q8_0.gguf \\
        --output deepred-p4v5-q8_0.gguf \\
        --system-prompt-file system_prompt_sp-01.txt \\
        --name "DeepRed p4-v5" --template-out deepred-p4v5.jinja
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

MARKER = '{#- DeepRed default system prompt -#}'
# Gemma 4's template binds `messages` once and reads messages[0] for the
# system turn, so the default must be in place before its first statement.
INJECTION = (
    MARKER + '\n'
    "{%- if not (messages and messages[0]['role'] in ['system', 'developer']) -%}\n"
    "{%- set messages = [{'role': 'system', 'content': DEFAULT}] + (messages or []) -%}\n"
    '{%- endif -%}\n'
)
# Every DeepRed checkpoint trained and was scored on the non-thinking prompt
# (empty thought block). llama-server defaults enable_thinking to true, so a
# release that relied on callers passing the kwarg would serve an untrained
# context to anyone who forgot it.
THINKING_DEFAULT = '{%- set enable_thinking = enable_thinking | default(false) -%}'
THINKING_FORCED = '{%- set enable_thinking = false -%}'


class BakeError(ValueError):
    pass


def patch_template(template, prompt):
    """Return `template` with `prompt` as the default system message and
    thinking disabled whatever the caller passes."""
    if MARKER in template:
        raise BakeError('template already carries a default system prompt')
    prompt = prompt.strip()
    if not prompt:
        raise BakeError('empty system prompt')
    if template.count(THINKING_DEFAULT) != 1:
        raise BakeError('template has no single enable_thinking default to pin; '
                        'not a Gemma 4 template this script was written for')
    template = template.replace(THINKING_DEFAULT, THINKING_FORCED)
    # json.dumps yields a double-quoted literal whose escapes Jinja accepts.
    return INJECTION.replace('DEFAULT', json.dumps(prompt, ensure_ascii=False)) + template


def read_template(gguf_path):
    from gguf import GGUFReader
    reader = GGUFReader(str(gguf_path))
    field = reader.fields.get('tokenizer.chat_template')
    if field is None:
        raise BakeError(f'{gguf_path} has no tokenizer.chat_template')
    return bytes(field.parts[field.data[0]]).decode('utf-8')


def render_check(template, prompt):
    """Render with jinja2 the way transformers does, both with and without a
    caller system turn. llama.cpp has its own Jinja engine, so the runbook
    also checks the served rendering through /apply-template."""
    import jinja2
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                             extensions=['jinja2.ext.loopcontrols'])
    env.filters.setdefault('tojson', lambda value, **_: json.dumps(value))
    compiled = env.from_string(template)
    user = [{'role': 'user', 'content': 'SENTINEL_USER'}]
    # enable_thinking=True is what llama-server sends when the caller is silent.
    plain = compiled.render(messages=user, bos_token='<bos>',
                            add_generation_prompt=True, enable_thinking=True)
    custom = compiled.render(
        messages=[{'role': 'system', 'content': 'SENTINEL_SYSTEM'}] + user,
        bos_token='<bos>', add_generation_prompt=True, enable_thinking=True)
    problems = []
    if prompt.strip() not in plain:
        problems.append('default prompt missing when no system turn is sent')
    if 'SENTINEL_SYSTEM' not in custom or prompt.strip() in custom:
        problems.append('caller system turn does not replace the default')
    if '<|think|>' in plain or '<|think|>' in custom:
        problems.append('thinking can still be enabled by the caller')
    if not plain.rstrip().endswith('<channel|>'):
        problems.append('generation prompt lacks the empty thought block')
    return problems, plain


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--system-prompt-file', required=True)
    parser.add_argument('--name', help='general.name for the output')
    parser.add_argument('--template-out', help='also write the patched template here')
    parser.add_argument('--check-only', action='store_true',
                        help='patch and render-check, but write no GGUF')
    args = parser.parse_args(argv)

    prompt = Path(args.system_prompt_file).read_text(encoding='utf-8').strip()
    try:
        patched = patch_template(read_template(args.input), prompt)
    except BakeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    problems, rendered = render_check(patched, prompt)
    if problems:
        print('ERROR: patched template fails its render check:', file=sys.stderr)
        for problem in problems:
            print(f'  - {problem}', file=sys.stderr)
        return 1
    print('render check: default prompt inserted; caller system turn replaces it')
    print(rendered[:400])
    if args.template_out:
        Path(args.template_out).write_text(patched, encoding='utf-8')
    if args.check_only:
        return 0

    import gguf
    tool = Path(gguf.__file__).parent / 'scripts' / 'gguf_new_metadata.py'
    template_file = Path(args.output).with_suffix('.jinja.tmp')
    template_file.write_text(patched, encoding='utf-8')
    command = [sys.executable, str(tool), '--chat-template-file', str(template_file),
               '--force']
    if args.name:
        command += ['--general-name', args.name]
    command += [args.input, args.output]
    try:
        subprocess.run(command, check=True)
    finally:
        template_file.unlink(missing_ok=True)
    if read_template(args.output) != patched:
        print('ERROR: written GGUF does not carry the patched template', file=sys.stderr)
        return 1
    print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
