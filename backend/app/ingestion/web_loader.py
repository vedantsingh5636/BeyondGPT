import ipaddress
import socket
from urllib.parse import urlparse

from langchain_community.document_loaders import WebBaseLoader

_BLOCKED_HOSTS = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "169.254.169.254",
}


def _is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_public_url(url: str) -> str:
    """Validate that a URL is an external HTTP(S) target and not an internal/SSRF address."""
    cleaned = url.strip()
    parsed = urlparse(cleaned)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Invalid website URL. Only http and https URLs are allowed.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not allowed.")

    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if (
        not hostname
        or hostname in _BLOCKED_HOSTS
        or hostname.endswith((".localhost", ".local", ".internal"))
    ):
        raise ValueError("Access to internal or local network addresses is not allowed.")

    # Direct IP literal check
    try:
        if not _is_public_ip(hostname):
            raise ValueError("Access to private or internal IP addresses is not allowed.")
        return cleaned
    except ValueError as exc:
        if "not allowed" in str(exc):
            raise

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addr_info = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname '{hostname}'.") from exc

    resolved_ips = [info[4][0] for info in addr_info if info and info[4]]
    if not resolved_ips or not all(_is_public_ip(ip) for ip in resolved_ips):
        raise ValueError("Access to private or internal IP addresses is not allowed.")

    return cleaned


def load_website(url: str):
    validated_url = validate_public_url(url)

    loader = WebBaseLoader(
        web_paths=(validated_url,),
        requests_kwargs={"timeout": 15, "allow_redirects": False},
    )

    return loader.load()
