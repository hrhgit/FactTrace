"""Release boundaries: one fact pipeline, atomic storage and a clean source package."""
import hashlib
from io import StringIO
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from contextlib import redirect_stdout, redirect_stderr

from scripts.package_submission import package
from tests.fixtures import ScriptedModel, cite, payload
from text_memory import cli
from text_memory.collections import CollectionEngine, CollectionStore
from text_memory.collections.service import CollectionApplication
from text_memory.errors import ConflictError, NotFoundError, ValidationError
from text_memory.model import ModelConfig
from text_memory.store import Scope

ROOT = Path(__file__).resolve().parents[1]


class Vectors:
    identity = {'model': 'release-fixture'}

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def close(self):
        pass


class MainlineTests(unittest.TestCase):
    def setUp(self):
        self.store = CollectionStore()
        self.addCleanup(self.store.close)
        self.scope = Scope('alice', 'exhibition')
        self.request = payload(messages=[{'role': 'user', 'session_date': '2024-05-01',
                                          'content': 'Alice keeps the beacon in the attic.'}])
        self.extraction = {'facts': [{'text': 'Alice keeps the beacon in the attic.', 'evidence': ['m0.1']}]}

    def test_one_write_pipeline_no_keyword_or_group_steps_and_idempotent_retry(self):
        model = ScriptedModel(self.extraction)
        engine = CollectionEngine(self.store, model, embedder=Vectors(), floor=.62)
        first = engine.add(self.request)
        self.assertEqual(engine.add(self.request), first)
        self.assertEqual([c['purpose'] for c in model.calls], ['collection_extract'])
        self.assertEqual(set(first), {'success', 'user_id', 'scope_id', 'session_id', 'request_id',
                                     'revision', 'source_ids', 'point_ids'})
        tables = {r[0] for r in self.store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({'objects', 'sources', 'fact_relations', 'relation_requests'} <= tables)
        self.assertFalse({'fact_keywords', 'hub_members', 'memberships', 'lookup'} & tables)
        with self.assertRaises(ConflictError):
            engine.add(self.request | {'messages': [{'role': 'user', 'content': 'Changed.'}]})

    def test_invalid_provenance_rolls_back_sources_and_receipt(self):
        points = CollectionEngine.extraction_points(self.request, self.extraction)
        points[0]['evidence'] = ['s_' + '0' * 64 + '.1']
        with self.assertRaises(NotFoundError):
            self.store.commit(self.request, {'points': points}, expected_revision=0)
        self.assertEqual(self.store.revision(self.scope), 0)
        self.assertEqual(list(self.store.records(self.scope, 'source')), [])
        self.assertIsNone(self.store.receipt(self.request))

    def test_scope_revision_and_change_evidence_remain_enforced(self):
        first = CollectionEngine(self.store, ScriptedModel()).add(self.request, extraction=self.extraction)
        ident = first['point_ids'][0]
        changed = payload('changed', messages=[{'role': 'user', 'content': 'The beacon is now in the cellar.'}])
        point = {'id': ident, 'text': 'The beacon is now in the cellar.', 'evidence': cite(changed)}
        with self.assertRaises(ValidationError):
            self.store.commit(changed, {'points': [point]}, expected_revision=1)
        point['change'] = {'reason': 'Moved', 'evidence': cite(changed)}
        self.store.commit(changed, {'points': [point]}, expected_revision=1)
        self.assertEqual(self.store.get(Scope('alice', 'exhibition', 1), 'point', ident)['text'], self.extraction['facts'][0]['text'])
        self.assertEqual(self.store.get(self.scope, 'point', ident)['text'], point['text'])
        self.assertEqual(list(self.store.records(Scope('other', 'exhibition'), 'point')), [])
        with self.assertRaises(ConflictError):
            self.store.commit(payload('concurrent'), {}, expected_revision=1)

    def test_read_existing_fact_database_ignores_old_tables_without_destroying_them(self):
        CollectionEngine(self.store, ScriptedModel()).add(self.request, extraction=self.extraction)
        self.store.db.execute('CREATE TABLE fact_keywords (old_data TEXT)')
        self.store.db.execute("INSERT INTO fact_keywords VALUES ('kept')")
        self.store.db.execute('CREATE TABLE hub_members (old_data TEXT)')
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'existing.sqlite'
            with sqlite3.connect(database) as destination:
                self.store.db.backup(destination)
            app = CollectionApplication(database, ScriptedModel, embedder=Vectors(), floor=.62)
            result = app.handle('/search', {'user_id': 'alice', 'scope_id': 'exhibition', 'query': 'Where is the beacon?', 'top_k': 1})
            self.assertIn('attic', result['data'][0]['content'])
            with CollectionStore(database) as store:
                self.assertEqual(store.db.execute('SELECT old_data FROM fact_keywords').fetchone()[0], 'kept')

    def test_cli_add_and_query_share_the_formal_pipeline_and_keep_trace_separate(self):
        config = ModelConfig('https://test.invalid', 'test', 'test')
        extract = ScriptedModel(self.extraction)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'memory.sqlite'
            source = Path(directory) / 'add.json'; source.write_text(json.dumps(self.request))
            query = Path(directory) / 'query.json'; query.write_text(json.dumps({'user_id': 'alice', 'scope_id': 'exhibition', 'question': 'beacon?'}))
            trace = Path(directory) / 'trace.json'
            with patch.object(cli, 'load_dotenv'), patch.object(cli.ModelConfig, 'from_env', return_value=config), \
                 patch.object(cli, 'embedder_from_env', return_value=Vectors()), patch.object(cli, 'embed_floor_from_env', return_value=.62), \
                 patch.object(cli, 'TimedModel', side_effect=[extract, ScriptedModel()]), redirect_stdout(StringIO()):
                self.assertEqual(cli.main(['add', '--database', str(database), '--input', str(source)]), 0)
                self.assertEqual(cli.main(['query', '--database', str(database), '--input', str(query), '--trace', str(trace)]), 0)
            saved = json.loads(trace.read_text())
            self.assertEqual([c['purpose'] for c in extract.calls], ['collection_extract'])
            self.assertEqual([s['tool'] for s in saved['steps']], ['mark_question', 'rank_question_facts', 'complete_fact_context'])

    def test_removed_modes_are_unavailable_and_cli_missing_config_stops_before_requests(self):
        for arguments in [['keywords'], ['run'], ['add', '--database', 'unused', '--input', '-', '--with-keywords']]:
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                cli.main(arguments)
        config = ModelConfig('https://test.invalid', 'test', 'test')
        with patch.object(cli, 'load_dotenv'), patch.object(cli.ModelConfig, 'from_env', return_value=config), \
             patch.object(cli, 'embedder_from_env', return_value=None), patch.object(cli, 'make_server') as server, \
             redirect_stderr(StringIO()):
            self.assertEqual(cli.main(['serve', '--database', 'unused']), 2)
            server.assert_not_called()


class SourcePackageTests(unittest.TestCase):
    def test_zip_excludes_history_and_local_state_and_runs_without_the_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = base / 'source.zip'
            result = package(archive)
            with zipfile.ZipFile(archive) as bundle:
                names = set(bundle.namelist())
                manifest = json.loads(bundle.read('MANIFEST.json'))['files_sha256']
                for name, expected in manifest.items():
                    self.assertEqual(hashlib.sha256(bundle.read(name)).hexdigest(), expected)
                self.assertEqual(names, set(manifest) | {'MANIFEST.json'})
                self.assertTrue({'text_memory/cli.py', 'text_memory/collections/relations.py', '.env.example'} <= names)
                self.assertFalse(any(set(Path(n).parts) & {'outputs', 'state', 'worklog', '__pycache__', 'archive'} for n in names))
                self.assertNotIn('.env', names)
                self.assertFalse(any(n.endswith(('.sqlite', '.pyc')) for n in names))
                for old in ('keywords.py', 'legacy_placement.py', 'second_pass.py', 'clustering.py'):
                    self.assertNotIn('text_memory/collections/' + old, names)
                for historical in ('audit_model_format_failures.py', 'run_historical_failure_ablation.py',
                                   'run_json_contract_ablation.py', 'validate_model_format_repairs.py'):
                    self.assertNotIn('scripts/' + historical, names)
                bundle.extractall(base / 'isolated')
            isolated = base / 'isolated'
            process = subprocess.run([sys.executable, '-m', 'text_memory', '--help'], cwd=isolated,
                                     capture_output=True, text=True, check=True)
            self.assertIn('add,query,serve', process.stdout)
            self.assertNotIn('keywords', process.stdout)
            self.assertEqual(result['sha256'], hashlib.sha256(archive.read_bytes()).hexdigest())
