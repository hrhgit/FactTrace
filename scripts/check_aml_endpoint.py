#!/usr/bin/env python3
"""Run a private Add/Search contract check against our own endpoint.

This never contacts the AML evaluation platform and never creates a Smoke or
Full job. It writes one synthetic memory under a caller-provided test user,
then checks idempotent Add, Search, isolation, authentication, and health.
The endpoint must be supplied explicitly or through AML_ENDPOINT_URL.
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid

import httpx


def headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def check(response: httpx.Response, expected: int, label: str) -> dict:
    if response.status_code != expected:
        raise RuntimeError(f"{label}: expected HTTP {expected}, got {response.status_code}: {response.text[:300]}")
    try:
        return response.json()
    except ValueError as error:
        raise RuntimeError(f"{label}: response was not JSON") from error


def run(base_url: str, token: str, user_id: str, request_id: str) -> None:
    base_url = base_url.rstrip("/")
    payload = {
        "request_id": request_id,
        "user_id": user_id,
        "session_id": f"aml-prep-session-{uuid.uuid4().hex[:10]}",
        "messages": [{
            "role": "user",
            "timestamp": 1793232000000,
            "content": "AML private contract check: the blue notebook is stored in the study cabinet.",
        }],
    }
    auth = headers(token)
    with httpx.Client(timeout=45) as client:
        check(client.get(base_url + "/health"), 200, "health")
        check(client.post(base_url + "/search", json={"query": "notebook", "user_id": user_id,
                                                       "top_k": 1}), 401, "missing auth")
        added = check(client.post(base_url + "/add", json=payload, headers=auth), 200, "add")
        if added.get("success") is not True or added.get("request_id") != request_id:
            raise RuntimeError("add: response did not confirm the same request")
        repeated = check(client.post(base_url + "/add", json=payload, headers=auth), 200, "idempotent add")
        if repeated != added:
            raise RuntimeError("idempotent add: repeated request returned a different result")
        found = check(client.post(base_url + "/search", json={"query": "blue notebook study cabinet",
                                                               "user_id": user_id, "top_k": 1},
                                  headers=auth), 200, "search")
        rows = found.get("data")
        if not isinstance(rows, list) or not rows or not all("id" in row and "content" in row for row in rows):
            raise RuntimeError("search: expected data rows with id and content")
        isolated = check(client.post(base_url + "/search", json={"query": "blue notebook",
                                                                  "user_id": user_id + "-other", "top_k": 1},
                                     headers=auth), 200, "user isolation")
        if isolated.get("data") != []:
            raise RuntimeError("user isolation: another user received the test memory")
    print("private AML endpoint check passed")
    print(f"test_user_id={user_id}")
    print(f"request_id={request_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("AML_ENDPOINT_URL"),
                        help="our own Add/Search base URL; never an AML platform URL")
    parser.add_argument("--token", default=os.environ.get("AML_SERVICE_TOKEN"))
    parser.add_argument("--user-id", default=os.environ.get("AML_TEST_USER_ID", "aml-prep-check"))
    parser.add_argument("--request-id", default=os.environ.get("AML_TEST_REQUEST_ID"))
    args = parser.parse_args()
    if not args.base_url or not args.token:
        parser.error("provide --base-url and --token, or set AML_ENDPOINT_URL and AML_SERVICE_TOKEN")
    request_id = args.request_id or f"aml-prep:{uuid.uuid4().hex}"
    try:
        run(args.base_url, args.token, args.user_id, request_id)
    except (httpx.HTTPError, RuntimeError) as error:
        print(f"private AML endpoint check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
