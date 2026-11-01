import socket
from unittest.mock import patch
import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ingestion import sources
from app.ingestion.sources import html_to_text, IngestionError, validate_url

client = TestClient(app)

PUBLIC_IP = "93.184.215.14"


def fake_index_chunks(documents):
    return {document.metadata["source"]: 1 for document in documents}


@pytest.fixture
def mock_index():
    """Replace indexing so tests don't load the embedding model."""

    with (
        patch("app.api.index_chunks", side_effect=fake_index_chunks) as index,
        patch("app.api.collection.count", return_value=99),
    ):
        yield index


@pytest.fixture
def public_dns():
    """Resolve every hostname to a public IP without real DNS."""

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, port))]

    with patch("app.ingestion.sources.socket.getaddrinfo", fake_getaddrinfo):
        yield


def mock_http(handler):
    """Route the loader's httpx.Client through an in-memory transport."""

    real_client = httpx.Client

    def factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    return patch("app.ingestion.sources.httpx.Client", side_effect=factory)


# ---------------------------------------------------------------------------
# File uploads
# ---------------------------------------------------------------------------

def test_ingest_without_input_uses_local_corpus():
    with (
        patch("app.api.index_documents", return_value={"body.md": 3}) as corpus,
        patch("app.api.collection.count", return_value=3),
    ):
        response = client.post("/ingest")

    assert response.status_code == 200
    corpus.assert_called_once()
    assert response.json()["ingested"] == [
        {"source": "body.md", "source_type": "corpus", "chunks": 3}
    ]
    assert response.json()["total_chunks"] == 3


def test_ingest_uploaded_files(mock_index):
    response = client.post(
        "/ingest",
        files=[
            ("files", ("guide.md", b"# Guide\n\nSome content.", "text/markdown")),
            ("files", ("notes.txt", b"Plain text notes.", "text/plain")),
        ],
    )

    assert response.status_code == 200
    documents = mock_index.call_args.args[0]
    assert [d.metadata["source"] for d in documents] == ["guide.md", "notes.txt"]
    assert documents[0].metadata["source_type"] == "file"
    assert documents[0].page_content.startswith("# Guide")
    assert response.json()["total_chunks"] == 99


def test_uploaded_filename_path_is_stripped(mock_index):
    response = client.post(
        "/ingest",
        files={"files": ("..\\..\\secret/evil.md", b"content", "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json()["ingested"][0]["source"] == "evil.md"


def test_unsupported_file_type_is_rejected(mock_index):
    response = client.post(
        "/ingest",
        files={"files": ("image.png", b"\x89PNG", "image/png")},
    )

    assert response.status_code == 415
    mock_index.assert_not_called()


def test_non_utf8_file_is_rejected(mock_index):
    response = client.post(
        "/ingest",
        files={"files": ("bad.txt", b"\xff\xfe\xfa", "text/plain")},
    )

    assert response.status_code == 400
    mock_index.assert_not_called()


def test_empty_file_is_rejected(mock_index):
    response = client.post(
        "/ingest",
        files={"files": ("empty.md", b"   \n", "text/markdown")},
    )

    assert response.status_code == 400


def test_oversized_file_is_rejected(mock_index):
    with patch("app.ingestion.sources.MAX_UPLOAD_BYTES", 10):
        response = client.post(
            "/ingest",
            files={"files": ("big.md", b"x" * 11, "text/markdown")},
        )

    assert response.status_code == 413
    mock_index.assert_not_called()


def test_one_bad_file_ingests_nothing(mock_index):
    response = client.post(
        "/ingest",
        files=[
            ("files", ("good.md", b"Good content", "text/markdown")),
            ("files", ("bad.pdf", b"%PDF", "application/pdf")),
        ],
    )

    assert response.status_code == 415
    mock_index.assert_not_called()


def test_too_many_items_is_rejected(mock_index):
    with patch("app.api.MAX_INGEST_ITEMS", 1):
        response = client.post(
            "/ingest",
            files=[
                ("files", ("a.md", b"a", "text/markdown")),
                ("files", ("b.md", b"b", "text/markdown")),
            ],
        )

    assert response.status_code == 422
    mock_index.assert_not_called()


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

def test_ingest_html_url(mock_index, public_dns):
    html = (
        "<html><head><title>T</title><style>p{}</style></head>"
        "<body><h1>Security</h1><script>alert(1)</script>"
        "<p>Use OAuth2 with   password flow.</p></body></html>"
    )

    def handler(request):
        return httpx.Response(200, html=html)

    with mock_http(handler):
        response = client.post(
            "/ingest",
            data={"urls": ["https://example.com/security#intro"]},
        )

    assert response.status_code == 200
    document = mock_index.call_args.args[0][0]
    assert document.metadata["source"] == "https://example.com/security"
    assert document.metadata["source_type"] == "url"
    assert "Use OAuth2 with password flow." in document.page_content
    assert "alert" not in document.page_content
    assert "p{}" not in document.page_content


def test_ingest_markdown_url_and_file_together(mock_index, public_dns):
    def handler(request):
        return httpx.Response(
            200,
            text="# Raw markdown",
            headers={"content-type": "text/markdown; charset=utf-8"},
        )

    with mock_http(handler):
        response = client.post(
            "/ingest",
            files={"files": ("guide.md", b"File content", "text/markdown")},
            data={"urls": "https://example.com/a.md, https://example.com/b.md"},
        )

    assert response.status_code == 200
    sources_ingested = [item["source"] for item in response.json()["ingested"]]
    assert sources_ingested == [
        "guide.md",
        "https://example.com/a.md",
        "https://example.com/b.md",
    ]


def test_url_redirect_to_private_address_is_blocked(mock_index):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        ip = "127.0.0.1" if host == "internal.local" else PUBLIC_IP
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    def handler(request):
        return httpx.Response(
            302, headers={"location": "http://internal.local/admin"}
        )

    with (
        patch("app.ingestion.sources.socket.getaddrinfo", fake_getaddrinfo),
        mock_http(handler),
    ):
        response = client.post(
            "/ingest", data={"urls": ["https://example.com/docs"]}
        )

    assert response.status_code == 422
    assert "private" in response.json()["detail"]
    mock_index.assert_not_called()


def test_url_http_error(mock_index, public_dns):
    with mock_http(lambda request: httpx.Response(404)):
        response = client.post(
            "/ingest", data={"urls": ["https://example.com/missing"]}
        )

    assert response.status_code == 502


def test_url_unsupported_content_type(mock_index, public_dns):
    def handler(request):
        return httpx.Response(
            200, content=b"%PDF", headers={"content-type": "application/pdf"}
        )

    with mock_http(handler):
        response = client.post(
            "/ingest", data={"urls": ["https://example.com/file"]}
        )

    assert response.status_code == 415


def test_url_timeout(mock_index, public_dns):
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    with mock_http(handler):
        response = client.post(
            "/ingest", data={"urls": ["https://example.com/slow"]}
        )

    assert response.status_code == 504


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file.md",
        "file:///etc/passwd",
        "not a url",
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "http://10.0.0.5/",
        "https://user:pass@example.com/",
    ],
)
def test_validate_url_rejects_unsafe_urls(url):
    with pytest.raises(IngestionError):
        validate_url(url)


def test_html_to_text_keeps_paragraphs():
    text = html_to_text("<p>First</p><p>Second   line</p>")

    assert text == "First\n\nSecond line"


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

def test_index_chunks_replaces_existing_source_chunks():
    from langchain_core.documents import Document
    from app.ingestion import indexer

    documents = [
        Document(page_content="Alpha content", metadata={"source": "a.md"}),
        Document(page_content="Beta content", metadata={"source": "b.md"}),
    ]

    with (
        patch.object(indexer, "generate_embeddings",
                     side_effect=lambda texts: [[0.0] for _ in texts]),
        patch.object(indexer, "collection") as collection,
    ):
        counts = indexer.index_chunks(documents)

    assert counts == {"a.md": 1, "b.md": 1}
    collection.delete.assert_any_call(where={"source": "a.md"})
    collection.delete.assert_any_call(where={"source": "b.md"})

    ids = collection.upsert.call_args.kwargs["ids"]
    assert len(set(ids)) == 2
    assert ids[0] != "chunk_0"



def test_openapi_marks_files_as_binary_for_swagger():
    schema = client.get("/openapi.json").json()
    ref = schema["paths"]["/ingest"]["post"]["requestBody"]["content"][
        "multipart/form-data"]["schema"]["$ref"].split("/")[-1]
    files = schema["components"]["schemas"][ref]["properties"]["files"]

    assert files["items"]["format"] == "binary"



def test_swagger_empty_fields_ingest_local_corpus():
    """Swagger sends files= and urls= as empty text when nothing is chosen."""

    with (
        patch("app.api.index_documents", return_value={"body.md": 3}) as corpus,
        patch("app.api.collection.count", return_value=3),
    ):
        response = client.post("/ingest", data={"files": "", "urls": ""})

    assert response.status_code == 200
    corpus.assert_called_once()


def test_swagger_file_with_empty_url_field(mock_index):
    response = client.post(
        "/ingest",
        files={"files": ("guide.md", b"Content", "text/markdown")},
        data={"urls": ""},
    )

    assert response.status_code == 200
    assert response.json()["ingested"][0]["source"] == "guide.md"


def test_file_sent_as_text_is_rejected(mock_index):
    response = client.post("/ingest", data={"files": "raw text, not a file"})

    assert response.status_code == 422
    assert "file uploads" in response.json()["detail"]
    mock_index.assert_not_called()
