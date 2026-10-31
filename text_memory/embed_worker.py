"""Embedding worker run by another Python environment: one JSON request per line on stdin.

Usage: python -I embed_worker.py MODEL [REVISION]. Loads only locally cached
files; the parent sets the Hugging Face offline variables.
"""
import json
import sys


def main():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(sys.argv[1], revision=sys.argv[2] if len(sys.argv) > 2 else None, device="cpu")
    except Exception as error:  # Report any load failure to the parent, then exit.
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        print(json.dumps({"ready": False}), flush=True)
        return 1
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            texts = json.loads(line)["texts"]
            vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            print(json.dumps({"vectors": [[round(float(x), 6) for x in vector] for vector in vectors]}), flush=True)
        except Exception as error:
            print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
