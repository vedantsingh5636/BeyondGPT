import re
import shutil
import tempfile
from urllib.parse import urlparse

from langchain_community.document_loaders import GitLoader

from app.ingestion.web_loader import validate_public_url

_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")


def load_github_repo(repo_url: str, branch: str = "main"):
    validated_url = validate_public_url(repo_url)
    parsed = urlparse(validated_url)
    if (parsed.hostname or "").lower() not in {"github.com", "www.github.com"}:
        raise ValueError("Only public github.com repository URLs are allowed.")

    if not _BRANCH_RE.match(branch):
        raise ValueError("Invalid branch name.")

    repo_path = tempfile.mkdtemp(prefix="github_repo_")
    try:
        loader = GitLoader(
            clone_url=validated_url,
            repo_path=repo_path,
            branch=branch,
        )
        return loader.load()
    finally:
        shutil.rmtree(repo_path, ignore_errors=True)
