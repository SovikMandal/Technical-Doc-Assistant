"""Load documents from uploaded files and URLs for POST /ingest."""
import ipaddress
import socket
from html.parser import HTMLParser
from pathlib import PurePosixPath, PureWindowsPath
from typing import BinaryIO
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from langchain_core.documents import Document

from app.config import (
    ALLOWED_FILE_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    MAX_URL_BYTES,
    MAX_URL_REDIRECTS,
    URL_FETCH_TIMEOUT,
)

 
class IngestionError(Exception):
    """Raised when an uploaded file or URL cannot be ingested."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _safe_filename(filename: str | None) -> str:
    """Strip any directory parts a client may send in the filename."""

    if not filename or not filename.strip():
        raise IngestionError("Uploaded file has no filename.")

    # Handle both Windows and POSIX separators
    name = PureWindowsPath(PurePosixPath(filename).name).name.strip()

    if not name or name in {".", ".."}:
        raise IngestionError("Uploaded file has an invalid filename.")

    return name


def _decode_text(data: bytes, label: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise IngestionError(
            f"'{label}' is not valid UTF-8 text."
        ) from error


def load_uploaded_file(filename: str | None, file: BinaryIO) -> Document:
    """Validate an uploaded text file and convert it into a Document."""

    name = _safe_filename(filename)
    extension = PurePosixPath(name).suffix.lower()

    if extension not in ALLOWED_FILE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_FILE_EXTENSIONS))
        raise IngestionError(
            f"'{name}' has an unsupported file type. Allowed: {allowed}.",
            status_code=415,
        )

    # Read one byte past the limit to detect oversized files
    data = file.read(MAX_UPLOAD_BYTES + 1)

    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestionError(
            f"'{name}' exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB "
            "upload limit.",
            status_code=413,
        )

    content = _decode_text(data, name)

    if not content.strip():
        raise IngestionError(f"'{name}' is empty.")

    return Document(
        page_content=content,
        metadata={
            "source": name,
            "source_type": "file",
            "category": "Uploaded Document",
        },
    )


class _HTMLTextExtractor(HTMLParser):
    """Convert HTML into plain text, keeping paragraph breaks."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "template"}
    BLOCK_TAGS = {
        "p", "div", "section", "article", "main", "header", "footer",
        "li", "ul", "ol", "pre", "blockquote", "table", "tr",
        "h1", "h2", "h3", "h4", "h5", "h6", "br", "hr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self.BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self.BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        lines = [" ".join(line.split()) for line in text.splitlines()]

        # Collapse runs of blank lines into a single paragraph break
        paragraphs, current = [], []
        for line in lines:
            if line:
                current.append(line)
            elif current:
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))

        return "\n\n".join(paragraphs)


def html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.get_text()


def validate_url(url: str) -> str:
    """Allow only public http(s) URLs. Returns the URL without a fragment."""

    url = urldefrag(url.strip())[0]
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise IngestionError(
            f"'{url}' is not a valid http or https URL.",
            status_code=422,
        )

    if parsed.username or parsed.password:
        raise IngestionError(
            f"'{url}' must not contain credentials.",
            status_code=422,
        )

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = socket.getaddrinfo(parsed.hostname, port)
    except (socket.gaierror, ValueError) as error:
        raise IngestionError(
            f"Could not resolve host for '{url}'.",
            status_code=422,
        ) from error

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0].split("%")[0])
        if not ip.is_global:
            raise IngestionError(
                f"'{url}' points to a private or reserved address.",
                status_code=422,
            )

    return url


def _fetch(url: str) -> tuple[str, bytes, str | None, str]:
    """Fetch a URL, validating every redirect hop.

    Returns (final_url, body, content_type, encoding).
    """

    current_url = validate_url(url)

    with httpx.Client(
        timeout=URL_FETCH_TIMEOUT,
        follow_redirects=False,
        headers={"User-Agent": "Technical-Doc-Assistant/1.0"},
    ) as client:
        for _ in range(MAX_URL_REDIRECTS + 1):
            with client.stream("GET", current_url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise IngestionError(
                            f"'{url}' returned a redirect without a location.",
                            status_code=502,
                        )
                    current_url = validate_url(urljoin(current_url, location))
                    continue

                if response.status_code >= 400:
                    raise IngestionError(
                        f"'{url}' returned HTTP {response.status_code}.",
                        status_code=502,
                    )

                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_URL_BYTES:
                        raise IngestionError(
                            f"'{url}' exceeds the "
                            f"{MAX_URL_BYTES // (1024 * 1024)} MB size limit.",
                            status_code=413,
                        )

                return (
                    current_url,
                    bytes(body),
                    response.headers.get("content-type"),
                    response.encoding or "utf-8",
                )

    raise IngestionError(
        f"'{url}' redirected too many times.",
        status_code=502,
    )


def load_url(url: str) -> Document:
    """Download a web page or text file and convert it into a Document."""

    try:
        final_url, body, content_type, encoding = _fetch(url)
    except httpx.TimeoutException as error:
        raise IngestionError(
            f"Timed out while fetching '{url}'.",
            status_code=504,
        ) from error
    except httpx.HTTPError as error:
        raise IngestionError(
            f"Could not fetch '{url}'.",
            status_code=502,
        ) from error

    media_type = (content_type or "").split(";")[0].strip().lower()
    path_suffix = PurePosixPath(urlparse(final_url).path).suffix.lower()

    try:
        text = body.decode(encoding, errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")

    if media_type in {"text/html", "application/xhtml+xml"}:
        content = html_to_text(text)
    elif (
        media_type in {"text/plain", "text/markdown", "text/x-markdown"}
        or path_suffix in ALLOWED_FILE_EXTENSIONS
    ):
        content = text
    else:
        raise IngestionError(
            f"'{url}' has unsupported content type "
            f"'{media_type or 'unknown'}'.",
            status_code=415,
        )

    if not content.strip():
        raise IngestionError(f"'{url}' has no readable text content.")

    # Cite the URL the user submitted, without any #fragment
    source = urldefrag(url.strip())[0]

    return Document(
        page_content=content,
        metadata={
            "source": source,
            "source_type": "url",
            "category": "Web Page",
        },
    )
