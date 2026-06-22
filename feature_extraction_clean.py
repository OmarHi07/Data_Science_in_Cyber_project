
"""
Cleaned feature-extraction helpers for the phishing-detection project.

This module does not replace the original URLFeatureExtraction.py file.
It provides deterministic, testable helper functions used for the
reproducibility and software-engineering audit.

The goal is to make the expected behavior explicit for:
- IP detection
- iframe detection
- right-click detection
- WHOIS fallback behavior
- output schema consistency
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional
from urllib.parse import urlparse


FEATURE_COLUMNS = [
    "Have_IP",
    "Have_At",
    "URL_Length",
    "URL_Depth",
    "Redirection",
    "https_Domain",
    "TinyURL",
    "Prefix/Suffix",
    "DNS_Record",
    "Web_Traffic",
    "Domain_Age",
    "Domain_End",
    "iFrame",
    "Mouse_Over",
    "Right_Click",
    "Web_Forwards",
]


FULL_DATASET_COLUMNS = [
    "Domain",
    *FEATURE_COLUMNS,
    "Label",
]


@dataclass
class SimpleResponse:
    """Small response-like object used for deterministic testing."""

    text: str = ""
    history: list | None = None

    def __post_init__(self):
        if self.history is None:
            self.history = []


def _ensure_parseable_url(url: str) -> str:
    """Add a scheme if the URL has no scheme, so urlparse can extract hostname."""

    if "://" not in url:
        return "http://" + url

    return url


def get_domain(url: str) -> str:
    """Extract the hostname/domain from a URL."""

    parsed = urlparse(_ensure_parseable_url(url))
    hostname = parsed.hostname or ""

    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname


def havingIP(url: str) -> int:
    """
    Return 1 if the URL hostname is an IP address, otherwise 0.

    This fixes the common bug where the complete URL is passed directly
    to ipaddress.ip_address(...). The correct behavior is to parse the
    hostname first.
    """

    hostname = get_domain(url)

    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def haveAtSign(url: str) -> int:
    """Return 1 if the URL contains '@', otherwise 0."""

    return int("@" in url)


def getLength(url: str) -> int:
    """Binary URL-length feature used by the original source."""

    return 0 if len(url) < 54 else 1


def getDepth(url: str) -> int:
    """Count non-empty path components in the URL."""

    parsed = urlparse(_ensure_parseable_url(url))
    path_parts = parsed.path.split("/")

    return sum(1 for part in path_parts if part)


def redirection(url: str) -> int:
    """Detect suspicious // redirection after the protocol section."""

    position = url.rfind("//")

    if position > 6:
        if position > 7:
            return 1
        return 0

    return 0


def httpDomain(url: str) -> int:
    """Return 1 if http/https appears inside the domain portion."""

    domain = urlparse(_ensure_parseable_url(url)).netloc
    return int("https" in domain or "http" in domain)


SHORTENING_SERVICES_PATTERN = (
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|"
    r"tinyurl|tinyurl\.com|tr\.im|is\.gd|cli\.gs|yfrog\.com|"
    r"migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|"
    r"twurl\.nl|snipurl\.com|short\.to|BudURL\.com|ping\.fm|"
    r"post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|"
    r"to\.ly|bit\.do|lnkd\.in|db\.tt|qr\.ae|adf\.ly|bitly\.com|"
    r"cur\.lv|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|"
    r"j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|"
    r"scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|"
    r"tweez\.me|v\.gd|link\.zip\.net"
)


def tinyURL(url: str) -> int:
    """Return 1 if the URL uses a known shortening service."""

    return int(re.search(SHORTENING_SERVICES_PATTERN, url, re.IGNORECASE) is not None)


def prefixSuffix(url: str) -> int:
    """Return 1 if the domain contains a hyphen."""

    return int("-" in get_domain(url))


def domainAge(domain_record) -> int:
    """
    Return 1 if domain age is suspicious, otherwise 0.

    The original source uses a threshold and returns fallback 1 when
    dates are missing or not parseable.
    """

    creation_date = getattr(domain_record, "creation_date", None)
    expiration_date = getattr(domain_record, "expiration_date", None)

    if isinstance(creation_date, list):
        creation_date = creation_date[0] if creation_date else None

    if isinstance(expiration_date, list):
        expiration_date = expiration_date[0] if expiration_date else None

    if isinstance(creation_date, str):
        try:
            creation_date = datetime.strptime(creation_date, "%Y-%m-%d")
        except ValueError:
            return 1

    if isinstance(expiration_date, str):
        try:
            expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        except ValueError:
            return 1

    if creation_date is None or expiration_date is None:
        return 1

    age_days = abs((expiration_date - creation_date).days)

    return 1 if (age_days / 30) < 6 else 0


def domainEnd(domain_record) -> int:
    """
    Return 1 if domain end period is suspicious, otherwise 0.

    Missing or invalid expiration dates use fallback 1.
    """

    expiration_date = getattr(domain_record, "expiration_date", None)

    if isinstance(expiration_date, list):
        expiration_date = expiration_date[0] if expiration_date else None

    if isinstance(expiration_date, str):
        try:
            expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        except ValueError:
            return 1

    if expiration_date is None:
        return 1

    days_to_expiration = abs((expiration_date - datetime.now()).days)

    return 0 if (days_to_expiration / 30) < 6 else 1


def get_domain_based_features(
    url: str,
    whois_lookup: Optional[Callable[[str], object]] = None,
) -> dict:
    """
    Return deterministic domain-based features.

    If WHOIS lookup is unavailable or fails, we explicitly return fallback
    suspicious values and expose the fallback flag.
    """

    hostname = get_domain(url)

    if whois_lookup is None:
        return {
            "DNS_Record": 1,
            "Domain_Age": 1,
            "Domain_End": 1,
            "WHOIS_Fallback_Used": True,
        }

    try:
        domain_record = whois_lookup(hostname)

        return {
            "DNS_Record": 0,
            "Domain_Age": domainAge(domain_record),
            "Domain_End": domainEnd(domain_record),
            "WHOIS_Fallback_Used": False,
        }

    except Exception:
        return {
            "DNS_Record": 1,
            "Domain_Age": 1,
            "Domain_End": 1,
            "WHOIS_Fallback_Used": True,
        }


def iframe(response) -> int:
    """
    Return 1 when an iframe/frame tag appears or when response is unavailable.

    A missing response is treated as suspicious, matching the original source's
    fallback logic. HTML is checked with a tag-oriented regex.
    """

    if response == "" or response is None:
        return 1

    text = getattr(response, "text", "")

    if re.search(r"<\s*i?frame\b", text, re.IGNORECASE):
        return 1

    return 0


def mouseOver(response) -> int:
    """Return 1 if suspicious mouse-over/status-bar behavior appears."""

    if response == "" or response is None:
        return 1

    text = getattr(response, "text", "")

    suspicious_pattern = r"onmouseover|window\.status"

    return int(re.search(suspicious_pattern, text, re.IGNORECASE) is not None)


def rightClick(response) -> int:
    """
    Return 1 if right-click blocking behavior appears.

    The original source returned 0 when event.button == 2 appeared.
    This cleaned function makes the encoding consistent:
    1 = suspicious, 0 = not suspicious.
    """

    if response == "" or response is None:
        return 1

    text = getattr(response, "text", "")

    suspicious_patterns = [
        r"event\.button\s*==\s*2",
        r"event\.button\s*===\s*2",
        r"oncontextmenu\s*=\s*['\"]?return\s+false",
        r"return\s+false",
    ]

    return int(
        any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in suspicious_patterns
        )
    )


def forwarding(response) -> int:
    """Return 1 if the response history suggests multiple forwarding steps."""

    if response == "" or response is None:
        return 1

    history = getattr(response, "history", [])

    return 0 if len(history) <= 2 else 1


def featureExtraction(
    url: str,
    response=None,
    whois_lookup: Optional[Callable[[str], object]] = None,
    web_traffic_value: int = 1,
) -> list[int]:
    """
    Return exactly 16 numerical predictors.

    Domain and Label are intentionally not included here.
    Domain is metadata, and Label is the target variable.
    """

    domain_features = get_domain_based_features(
        url=url,
        whois_lookup=whois_lookup,
    )

    features = [
        havingIP(url),
        haveAtSign(url),
        getLength(url),
        getDepth(url),
        redirection(url),
        httpDomain(url),
        tinyURL(url),
        prefixSuffix(url),
        domain_features["DNS_Record"],
        web_traffic_value,
        domain_features["Domain_Age"],
        domain_features["Domain_End"],
        iframe(response),
        mouseOver(response),
        rightClick(response),
        forwarding(response),
    ]

    if len(features) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} features but got {len(features)}"
        )

    return features


def feature_dict(
    url: str,
    response=None,
    whois_lookup: Optional[Callable[[str], object]] = None,
    web_traffic_value: int = 1,
) -> dict:
    """Return a dictionary mapping feature names to extracted values."""

    values = featureExtraction(
        url=url,
        response=response,
        whois_lookup=whois_lookup,
        web_traffic_value=web_traffic_value,
    )

    return dict(zip(FEATURE_COLUMNS, values))

