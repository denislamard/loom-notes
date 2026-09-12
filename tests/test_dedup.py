from loom_memory.ingest.dedup import content_hash


def test_hash_ignores_case_and_whitespace() -> None:
    assert content_hash("Bonjour  le\nmonde") == content_hash("bonjour le monde ")


def test_hash_differs_on_content() -> None:
    assert content_hash("a") != content_hash("b")
