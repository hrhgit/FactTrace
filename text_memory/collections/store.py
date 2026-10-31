"""Facts, source provenance, write receipts and user-side relation annotations."""
import json

from ..store import MemoryStore
from ..schema import require
from ..model import encode


class CollectionStore(MemoryStore):

    def __init__(self, path=":memory:"):
        super().__init__(path)
        try:
            marked = self.db.execute("SELECT name FROM sqlite_master WHERE name='collection_format'").fetchone()
            if not marked:
                require(not self.db.execute("SELECT 1 FROM scopes LIMIT 1").fetchone(),
                        "集合模式需要独立数据库，不能改写旧点线记忆库")
                self.db.execute("CREATE TABLE collection_format(version INTEGER NOT NULL)")
                self.db.execute("INSERT INTO collection_format VALUES(1)")
            require(self.db.execute("SELECT version FROM collection_format").fetchone()[0] == 1,
                    "不支持此集合数据库版本")
            # Write-time relation notes (relations.py); a request row marks its stage as done, links or not.
            self.db.execute('''CREATE TABLE IF NOT EXISTS relation_requests (
                user_id TEXT NOT NULL, scope_id TEXT NOT NULL, request_id TEXT NOT NULL, prompt_hash TEXT NOT NULL,
                PRIMARY KEY(user_id,scope_id,request_id,prompt_hash))''')
            self.db.execute('''CREATE TABLE IF NOT EXISTS fact_relations (
                user_id TEXT NOT NULL, scope_id TEXT NOT NULL, request_id TEXT NOT NULL, prompt_hash TEXT NOT NULL,
                earlier_id TEXT NOT NULL, later_id TEXT NOT NULL, relation TEXT NOT NULL,
                PRIMARY KEY(user_id,scope_id,prompt_hash,earlier_id,later_id,relation))''')
        except Exception:
            self.close()
            raise


    def request_receipts(self, scope):
        """Committed receipts of this user and scope visible at the scope's revision."""
        revision = self.revision(scope)
        rows = self.db.execute('SELECT result FROM requests WHERE user_id=? AND scope_id=?', (scope.user_id, scope.scope_id))
        return [r for r in (json.loads(row[0]) for row in rows) if r['revision'] <= revision]

    def relations_done(self, scope, request_id, *, prompt_hash):
        return self.db.execute('SELECT 1 FROM relation_requests WHERE user_id=? AND scope_id=? AND request_id=? AND prompt_hash=?',
                               (scope.user_id, scope.scope_id, request_id, prompt_hash)).fetchone() is not None

    def save_relations(self, scope, request_id, links, *, prompt_hash):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for link in links:
                self.db.execute('INSERT OR IGNORE INTO fact_relations VALUES(?,?,?,?,?,?,?)', (scope.user_id, scope.scope_id,
                    request_id, prompt_hash, link['earlier'], link['later'], link['relation']))
            self.db.execute('INSERT OR IGNORE INTO relation_requests VALUES(?,?,?,?)',
                            (scope.user_id, scope.scope_id, request_id, prompt_hash))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def fact_relations(self, scope, *, prompt_hash):
        """Links whose labeling request is visible at the scope's revision."""
        visible = {r['request_id'] for r in self.request_receipts(scope)}
        rows = self.db.execute('''SELECT request_id, earlier_id, later_id, relation FROM fact_relations
            WHERE user_id=? AND scope_id=? AND prompt_hash=? ORDER BY rowid''', (scope.user_id, scope.scope_id, prompt_hash))
        return [{'earlier': r['earlier_id'], 'later': r['later_id'], 'relation': r['relation']}
                for r in rows if r['request_id'] in visible]
