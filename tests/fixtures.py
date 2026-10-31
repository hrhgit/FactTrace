"""Hand-authored synthetic cases; these are not model-quality evaluations."""
from copy import deepcopy
import json

from text_memory.schema import normalize_add, source_records


def payload(request_id="first", *, user_id="alice", scope_id="exhibition", messages=None, session_id="conversation-1"):
    return normalize_add({"user_id": user_id, "scope_id": scope_id, "session_id": session_id,
                          "request_id": request_id, "messages": messages or [
        {"role": "user", "timestamp": "2026-03-01", "content": "印刷厂缺纸，因此展板交付推迟。"},
        {"role": "user", "content": "展板交付推迟导致展览延期。小林负责展板，小周负责场馆。"},
        {"role": "user", "content": "展板是展览准备的一部分。展板交付完成后才能布展。"},
    ]})


def cite(request, index=0, unit=1):
    """Paragraph references; fixture messages are single paragraphs, so unit 1."""
    return [f'{source_records(request)[index]["id"]}.{unit}']


class ScriptedModel:
    """Explicitly injected test collaborator, never a production fallback."""
    def __init__(self, *responses, flag_responses=None, relation_responses=None):
        self.responses = iter(responses)
        # Relation stage replies; without explicit replies every pair is unrelated and no record conflicts.
        self.relation_responses = iter(relation_responses) if relation_responses is not None else None
        # Search marks question attributes; without explicit replies no flag is set.
        self.flag_responses = iter(flag_responses) if flag_responses is not None else None
        self.calls = []

    def complete(self, messages, *, purpose):
        self.calls.append({"messages": deepcopy(messages), "purpose": purpose})
        if purpose in ('collection_relation_pairs', 'collection_relation_negation'):
            if self.relation_responses is not None:
                response = next(self.relation_responses)
            elif purpose == 'collection_relation_pairs':
                response = {'pairs': [{'id': p['id'], 'relation': 'none'} for p in json.loads(messages[-1]['content'])['pairs']]}
            else:
                response = {'conflicts': []}
        elif purpose == 'question_flags':
            response = (next(self.flag_responses) if self.flag_responses is not None else
                        {"combine": False, "changed": False, "ever": False})
        else:
            response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)

    def complete_text(self, messages, *, purpose, temperature=None):
        """Scripted answers may be written as {"answer": ...} or as plain text."""
        response = self.complete(messages, purpose=purpose)
        if temperature is not None:
            self.calls[-1]["request_parameters"] = {"temperature": temperature}
        return response["answer"] if isinstance(response, dict) else response


