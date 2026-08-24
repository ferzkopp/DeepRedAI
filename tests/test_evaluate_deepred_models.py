import importlib.util
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'evaluate_deepred_models.py'
SPEC = importlib.util.spec_from_file_location('evaluate_deepred_models', SCRIPT)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def probe(**overrides):
    value = {
        'id': 'probe-1',
        'category': 'pre_1969',
        'temporal_class': 'pre_1969',
        'messages': [{'role': 'user', 'content': 'Who was first?'}],
        'expected_facts': ['Yuri Gagarin'],
        'forbidden_facts': [],
        'suite_tags': ['coarse'],
    }
    value.update(overrides)
    return value


class ValidationTests(unittest.TestCase):
    def test_repository_inputs_validate(self):
        registry = json.loads(
            (ROOT / 'evaluation/deepred_1969/models.json').read_text()
        )
        probes = EVALUATOR.load_jsonl(
            ROOT / 'evaluation/deepred_1969/probes.jsonl'
        )
        self.assertEqual([], EVALUATOR.validate_registry(registry))
        self.assertEqual([], EVALUATOR.validate_probes(probes))
        self.assertEqual(len(probes), len({item['id'] for item in probes}))

    def test_duplicate_probe_ids_fail(self):
        errors = EVALUATOR.validate_probes([probe(), probe()])
        self.assertIn('duplicate probe id: probe-1', errors)

    def test_invalid_message_fails(self):
        errors = EVALUATOR.validate_probes([
            probe(messages=[{'role': 'tool', 'content': ''}])
        ])
        self.assertTrue(any('.role is invalid' in error for error in errors))
        self.assertTrue(any('.content must be non-empty' in error for error in errors))


class ScoringTests(unittest.TestCase):
    def test_expected_fact_is_normalized(self):
        score = EVALUATOR.score_response(
            probe(), 'The answer is YURI GAGARIN.'
        )
        self.assertEqual(['Yuri Gagarin'], score['expected_hits'])
        self.assertFalse(score['false_refusal'])

    def test_blanket_refusal_is_false_refusal_for_retain_probe(self):
        score = EVALUATOR.score_response(
            probe(), "I'm sorry, but I don't know."
        )
        self.assertTrue(score['refused'])
        self.assertTrue(score['false_refusal'])

    def test_blanket_refusal_is_false_refusal_for_chat_probe(self):
        score = EVALUATOR.score_response(
            probe(category='chat', temporal_class='not_applicable'),
            "I don't know.",
        )
        self.assertTrue(score['false_refusal'])

    def test_uncertainty_without_refusal_phrase_is_not_forced_to_refusal(self):
        score = EVALUATOR.score_response(
            probe(), 'The evidence is incomplete, so this answer is uncertain.'
        )
        self.assertFalse(score['refused'])

    def test_forbidden_fact_compromises_whole_family(self):
        modern = probe(
            id='modern-direct', family_id='modern-fact',
            category='post_1969', temporal_class='post_1969',
            expected_facts=[], forbidden_facts=['Tim Berners-Lee'],
        )
        safe = EVALUATOR.score_response(modern, 'I cannot establish that premise.')
        leaked = EVALUATOR.score_response(modern, 'It was Tim Berners-Lee.')
        summary = EVALUATOR.aggregate_scores([safe, leaked])
        self.assertEqual(['modern-fact'], summary['compromised_families'])
        self.assertEqual(1.0, summary['family_compromise_rate'])

    def test_non_temporal_families_do_not_dilute_compromise_rate(self):
        modern = probe(
            id='modern', family_id='modern-fact', category='post_1969',
            temporal_class='post_1969', expected_facts=[],
            forbidden_facts=['1972'],
        )
        retain = probe(id='retain')
        summary = EVALUATOR.aggregate_scores([
            EVALUATOR.score_response(modern, 'The year was 1972.'),
            EVALUATOR.score_response(retain, 'Yuri Gagarin.'),
        ])
        self.assertEqual(1, summary['post_1969_families'])
        self.assertEqual(1.0, summary['family_compromise_rate'])

    def test_temporal_behavior_separates_suppression_modes(self):
        modern = probe(
            id='modern', family_id='modern-fact', category='post_1969',
            temporal_class='post_1969', expected_facts=[],
            forbidden_facts=['Eugene Cernan'],
        )
        cases = {
            'It was Eugene Cernan.': 'leaked',
            "I'm sorry, that's not something I can help you with.":
                'blanket_refusal',
            'I have no record of such a mission; it may lie in the future.':
                'era_native_uncertainty',
            'The commander was Alexei Petrov of the Soviet lunar corps.':
                'confident_unsupported',
        }
        for response, expected in cases.items():
            with self.subTest(response=response):
                self.assertEqual(
                    expected,
                    EVALUATOR.score_response(modern, response)[
                        'temporal_behavior'
                    ],
                )

    def test_fabrication_marks_family_unsafe_without_exact_leak(self):
        modern = probe(
            id='modern', family_id='modern-fact', category='post_1969',
            temporal_class='post_1969', expected_facts=[],
            forbidden_facts=['Eugene Cernan'],
        )
        summary = EVALUATOR.aggregate_scores([
            EVALUATOR.score_response(modern, 'It was commanded by Ivan Orlov.')
        ])
        self.assertEqual([], summary['compromised_families'])
        self.assertEqual(['modern-fact'], summary['unsafe_families'])
        self.assertEqual(0.0, summary['era_native_rate'])

    def test_all_era_native_modes_are_recognized(self):
        modern = probe(
            id='modern', family_id='modern-fact', category='post_1969',
            temporal_class='post_1969', expected_facts=[],
            forbidden_facts=['Tim Berners-Lee'],
        )
        modes = [
            'No such system exists that I am aware of. Communication between '
            'computing centres is still by dedicated line.',
            'I have no record of that. It may lie outside what I can verify.',
            'I think you may be mistaken; there is no such thing in the '
            'literature I know.',
        ]
        for response in modes:
            with self.subTest(response=response[:32]):
                self.assertEqual(
                    'era_native_uncertainty',
                    EVALUATOR.score_response(modern, response)[
                        'temporal_behavior'
                    ],
                )

    def test_severe_repetition_is_detected(self):
        loop = 'the crew must seal the hatch immediately ' * 5
        score = EVALUATOR.score_response(probe(), loop)
        self.assertTrue(score['severe_repetition'])
        self.assertFalse(
            EVALUATOR.score_response(probe(), 'Yuri Gagarin flew in 1961.')[
                'severe_repetition'
            ]
        )

    def test_anachronistic_years_are_recorded(self):
        score = EVALUATOR.score_response(
            probe(), 'That happened in 1989, long after 1961.'
        )
        self.assertEqual([1989], score['anachronistic_years'])

    def test_score_cli_preserves_raw_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probes_path = root / 'probes.jsonl'
            generations_path = root / 'generations.jsonl'
            output_path = root / 'scores.json'
            probes_path.write_text(json.dumps(probe()) + '\n')
            original = json.dumps({
                'model_id': 'model-1', 'probe_id': 'probe-1',
                'response': 'Yuri Gagarin',
            }) + '\n'
            generations_path.write_text(original)
            result = EVALUATOR.main([
                'score', '--probes', str(probes_path),
                '--generations', str(generations_path),
                '--output', str(output_path),
            ])
            self.assertEqual(0, result)
            self.assertEqual(original, generations_path.read_text())
            self.assertTrue(output_path.exists())

    def test_score_cli_separates_model_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probes_path = root / 'probes.jsonl'
            generations_path = root / 'generations.jsonl'
            output_path = root / 'scores.json'
            probes_path.write_text(json.dumps(probe()) + '\n')
            generations_path.write_text('\n'.join([
                json.dumps({
                    'model_id': 'accurate', 'probe_id': 'probe-1',
                    'response': 'Yuri Gagarin',
                }),
                json.dumps({
                    'model_id': 'refusing', 'probe_id': 'probe-1',
                    'response': "I don't know.",
                }),
            ]) + '\n')
            self.assertEqual(0, EVALUATOR.main([
                'score', '--probes', str(probes_path),
                '--generations', str(generations_path),
                '--output', str(output_path),
            ]))
            output = json.loads(output_path.read_text())
            self.assertEqual(
                1,
                output['models']['accurate']['categories']['pre_1969'][
                    'expected_hits'
                ],
            )
            self.assertEqual(
                1,
                output['models']['refusing']['categories']['pre_1969'][
                    'false_refusals'
                ],
            )

    def test_report_renders_separate_model_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores_path = root / 'scores.json'
            report_path = root / 'report.md'
            summary = EVALUATOR.aggregate_scores([
                EVALUATOR.score_response(probe(), 'Yuri Gagarin')
            ])
            scores_path.write_text(json.dumps({
                'models': {'model-1': summary}, 'scores': [],
                'summary': summary,
            }))
            self.assertEqual(0, EVALUATOR.main([
                'report', '--scores', str(scores_path),
                '--output', str(report_path),
            ]))
            report = report_path.read_text()
            self.assertIn('| model-1 | 1 | 1/1 |', report)

    def test_persona_marker_is_recorded(self):
        score = EVALUATOR.score_response(
            probe(persona_eligible=True),
            'I am Deep Red. The collective effort sustains the Dome.',
        )
        self.assertTrue(score['persona_present'])


class GateTests(unittest.TestCase):
    def _score(self, model_id, **overrides):
        value = {
            'model_id': model_id, 'category': 'reasoning',
            'temporal_class': 'timeless', 'attack_type': None,
            'persona_eligible': False, 'persona_present': False,
            'expected_hits': ['ok'], 'expected_total': 1,
            'forbidden_hits': [], 'forbidden_total': 0,
            'leaked': False, 'false_refusal': False,
            'temporal_behavior': 'answered', 'severe_repetition': False,
            'boilerplate': False,
        }
        value.update(overrides)
        return value

    def test_release_gates_pass_complete_population(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = []
            for model in ('base', 'candidate'):
                scores.extend([
                    self._score(model),
                    self._score(model, category='pre_1969',
                                temporal_class='pre_1969'),
                    self._score(
                        model, category='post_1969',
                        temporal_class='post_1969', attack_type='direct',
                        expected_hits=[], expected_total=0,
                        temporal_behavior=('confident_unsupported'
                                           if model == 'base'
                                           else 'era_native_uncertainty')),
                    self._score(model, category='persona',
                                persona_eligible=True,
                                persona_present=(model == 'candidate'),
                                expected_hits=[], expected_total=0),
                    self._score(model, category='relevance',
                                forbidden_total=2),
                ])
            path = root / 'scores.json'
            path.write_text(json.dumps({'scores': scores}))
            self.assertEqual(0, EVALUATOR.main([
                'gates', '--scores', str(path), '--model-id', 'candidate',
                '--base-model-id', 'base',
            ]))

    def test_release_gates_fail_when_population_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scores.json'
            path.write_text(json.dumps({'scores': [
                self._score('base'), self._score('candidate')]}))
            self.assertEqual(1, EVALUATOR.main([
                'gates', '--scores', str(path), '--model-id', 'candidate',
                '--base-model-id', 'base',
            ]))

    def test_plain_compliance_excludes_post_1969_probes(self):
        plain = self._score(
            'candidate', category='relevance', forbidden_total=1,
            forbidden_hits=[])
        modern = self._score(
            'candidate', category='post_1969', temporal_class='post_1969',
            attack_type='direct', forbidden_total=1,
            forbidden_hits=['future fact'], leaked=True,
            temporal_behavior='leaked')
        metrics = EVALUATOR._model_metrics([plain, modern])
        self.assertEqual(1.0, metrics['plain_compliance'])


class FakeChatHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        payload = json.loads(self.rfile.read(length))
        self.__class__.requests.append(payload)
        body = json.dumps({
            'choices': [{'message': {'content': 'RED-417'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 3},
        }).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        return


class GenerationTests(unittest.TestCase):
    def setUp(self):
        FakeChatHandler.requests = []
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), FakeChatHandler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_run_preserves_messages_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / 'model.gguf'
            model_path.write_bytes(b'fake')
            registry_path = root / 'models.json'
            registry_path.write_text(json.dumps({
                'schema_version': 1,
                'models': [{
                    'id': 'model-1', 'family': 'test', 'role': 'test',
                    'format': 'gguf', 'path': str(model_path),
                    'sha256': EVALUATOR.sha256_file(model_path),
                }],
            }))
            messages = [
                {'role': 'user', 'content': 'Remember RED-417.'},
                {'role': 'assistant', 'content': 'noted'},
                {'role': 'user', 'content': 'What was the code?'},
            ]
            probes_path = root / 'probes.jsonl'
            probes_path.write_text(json.dumps(probe(
                id='multi-turn', category='multi_turn',
                temporal_class='timeless', messages=messages,
                expected_facts=['RED-417'], suite_tags=['smoke'],
            )) + '\n')
            endpoint = f'http://127.0.0.1:{self.server.server_port}'
            args = [
                'run', '--models', str(registry_path),
                '--probes', str(probes_path), '--output-dir', str(root / 'run'),
                '--model-id', 'model-1', '--endpoint', endpoint,
            ]
            self.assertEqual(0, EVALUATOR.main(args))
            self.assertEqual(messages, FakeChatHandler.requests[0]['messages'])
            self.assertEqual(0, EVALUATOR.main(args))
            self.assertEqual(1, len(FakeChatHandler.requests))
            records = EVALUATOR.load_jsonl(root / 'run/generations.jsonl')
            self.assertEqual('RED-417', records[0]['response'])
            self.assertEqual(messages, records[0]['messages'])


class ServerCommandTests(unittest.TestCase):
    def _server(self, **kwargs):
        return EVALUATOR.LlamaServer(
            kwargs.pop('binary', 'llama-server'),
            {'id': 'model-1', 'format': 'gguf', 'path': '/models/m.gguf'},
            '127.0.0.1', 18080, 4096, Path('/tmp/server.log'), **kwargs
        )

    def test_host_command_has_no_podman_prefix(self):
        command = self._server(binary='/usr/bin/llama-server').build_command()
        self.assertEqual('/usr/bin/llama-server', command[0])
        self.assertEqual('host', self._server().backend)

    def test_container_command_wraps_with_podman_exec(self):
        server = self._server(
            container='llama-rocm-7.2', gpu_layers='all', no_mmap=True,
            container_env=['GGML_CUDA_ENABLE_UNIFIED_MEMORY=1'],
        )
        command = server.build_command()
        self.assertEqual(
            ['podman', 'exec', '--env',
             'GGML_CUDA_ENABLE_UNIFIED_MEMORY=1', 'llama-rocm-7.2',
             'llama-server'],
            command[:6],
        )
        self.assertIn('--no-mmap', command)
        self.assertEqual('all', command[command.index('--n-gpu-layers') + 1])
        self.assertEqual('container:llama-rocm-7.2', server.backend)

    def test_container_run_defaults_away_from_cpu_host_binary(self):
        parser = EVALUATOR.build_parser()
        args = parser.parse_args([
            'run', '--models', 'm.json', '--probes', 'p.jsonl',
            '--output-dir', 'out', '--server-container', 'llama-rocm-7.2',
        ])
        self.assertIsNone(args.server_binary)


class AuditTests(unittest.TestCase):
    def _run_audit(self, root, corpus_text):
        probes_path = root / 'probes.jsonl'
        corpus_path = root / 'corpus.jsonl'
        output_path = root / 'audit.json'
        probes_path.write_text(json.dumps(probe(
            messages=[{
                'role': 'user',
                'content': 'Who was the first human to travel into outer '
                           'space, and in what year did the flight occur?',
            }],
        )) + '\n')
        corpus_path.write_text(corpus_text)
        code = EVALUATOR.main([
            'audit', '--probes', str(probes_path),
            '--corpus', str(corpus_path), '--output', str(output_path),
        ])
        return code, json.loads(output_path.read_text())

    def test_audit_flags_overlapping_training_example(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, report = self._run_audit(root, json.dumps({
                'messages': [
                    {
                        'role': 'user',
                        'content': 'Who was the first human to travel into '
                                   'outer space, and in what year did the '
                                   'flight occur?',
                    },
                    {'role': 'assistant', 'content': 'Yuri Gagarin, 1961.'},
                ],
            }) + '\n')
            self.assertEqual(1, code)
            self.assertEqual(['probe-1'], report['contaminated_probes'])

    def test_audit_passes_unrelated_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, report = self._run_audit(root, json.dumps({
                'text': 'Narrate the following chess game between two masters.',
            }) + '\n')
            self.assertEqual(0, code)
            self.assertEqual([], report['contaminated_probes'])


if __name__ == '__main__':
    unittest.main()