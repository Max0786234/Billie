"""Validated storage for user uploads (images for chat, documents for the knowledge base).

Security: the original API accepted an arbitrary server *path* for images, which let
any caller read files from the server and forward them to a remote model. Now:
* files arrive as uploads, never as paths;
* the type is decided from magic bytes, not the file name or client MIME type;
* size is capped; images are stored under random ids; document names are sanitised;
* lookups accept only the id format we generate, so ``../`` traversal is impossible.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from pathlib import Path

from billie.llm.base import ImagePart

_IMAGE_SIGNATURES: list[tuple[bytes, int, str, str]] = [
    # (magic, offset, media type, extension)
    (b"\x89PNG\r\n\x1a\n", 0, "image/png", "png"),
    (b"\xff\xd8\xff", 0, "image/jpeg", "jpg"),
    (b"GIF87a", 0, "image/gif", "gif"),
    (b"GIF89a", 0, "image/gif", "gif"),
    (b"WEBP", 8, "image/webp", "webp"),  # RIFF....WEBP
]
_IMAGE_ID = re.compile(r"^[0-9a-f]{32}\.(png|jpg|gif|webp)$")
_EXT_MEDIA = {"png": "image/png", "jpg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
_DOC_SUFFIXES = {".pdf", ".md", ".txt"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


class UploadError(ValueError):
    """Rejected upload; the message is safe to show to the user."""


def sniff_image(data: bytes) -> tuple[str, str] | None:
    for magic, offset, media, ext in _IMAGE_SIGNATURES:
        if data[offset : offset + len(magic)] == magic and (ext != "webp" or data[:4] == b"RIFF"):
            return media, ext
    return None


@dataclass(frozen=True)
class StoredImage:
    id: str
    media_type: str
    size: int


class ImageStore:
    def __init__(self, directory: Path, max_bytes: int) -> None:
        self.directory = directory
        self.max_bytes = max_bytes

    def save(self, data: bytes) -> StoredImage:
        if not data:
            raise UploadError("The image is empty.")
        if len(data) > self.max_bytes:
            raise UploadError(f"Images must be at most {self.max_bytes // (1024 * 1024)} MB.")
        sniffed = sniff_image(data)
        if sniffed is None:
            raise UploadError("Unsupported image. Use PNG, JPEG, WebP or GIF.")
        media, ext = sniffed
        self.directory.mkdir(parents=True, exist_ok=True)
        image_id = f"{secrets.token_hex(16)}.{ext}"
        (self.directory / image_id).write_bytes(data)
        return StoredImage(image_id, media, len(data))

    def load(self, image_id: str) -> ImagePart:
        if not _IMAGE_ID.match(image_id):
            raise UploadError("Invalid image id.")
        path = self.directory / image_id
        if not path.is_file():
            raise UploadError("Image not found. Upload it again.")
        return ImagePart(path.read_bytes(), _EXT_MEDIA[image_id.rsplit(".", 1)[1]])


@dataclass(frozen=True)
class StoredDocument:
    name: str
    size: int
    modified: float


class DocumentStore:
    """Documents users add to the general knowledge base."""

    def __init__(self, directory: Path, max_bytes: int) -> None:
        self.directory = directory
        self.max_bytes = max_bytes

    def save(self, filename: str, data: bytes) -> StoredDocument:
        name = self._safe_name(filename)
        suffix = Path(name).suffix.lower()
        if suffix not in _DOC_SUFFIXES:
            raise UploadError("Unsupported document. Use PDF, Markdown (.md) or text (.txt).")
        if not data:
            raise UploadError("The document is empty.")
        if len(data) > self.max_bytes:
            raise UploadError(f"Documents must be at most {self.max_bytes // (1024 * 1024)} MB.")
        if suffix == ".pdf":
            if not data.startswith(b"%PDF-"):
                raise UploadError("This file is not a valid PDF.")
        else:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise UploadError("Text documents must be UTF-8 encoded.") from exc
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / name
        path.write_bytes(data)
        stat = path.stat()
        return StoredDocument(name, stat.st_size, stat.st_mtime)

    def list(self) -> list[StoredDocument]:
        if not self.directory.exists():
            return []
        return sorted(
            (
                StoredDocument(p.name, p.stat().st_size, p.stat().st_mtime)
                for p in self.directory.iterdir()
                if p.is_file() and p.suffix.lower() in _DOC_SUFFIXES
            ),
            key=lambda d: d.name.lower(),
        )

    def delete(self, name: str) -> None:
        path = self.directory / self._safe_name(name)
        if not path.is_file():
            raise UploadError("Document not found.")
        path.unlink()

    @staticmethod
    def _safe_name(filename: str) -> str:
        base = Path(filename.replace("\\", "/")).name  # drop any directory components
        cleaned = _SAFE_NAME.sub("_", base).strip(" .")
        if not cleaned or cleaned.startswith("."):
            raise UploadError("Invalid file name.")
        return cleaned[:120]
