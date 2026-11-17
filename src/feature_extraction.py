"""
Feature Extraction Module for Phishing URL Detection.

Extracts lexical, structural, and syntactic features from raw URLs.
Does not access external websites or download content over the network.
Safe for training-time and real-time prediction-time extraction.
"""

from __future__ import annotations

import re
import ipaddress
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import tldextract
import pandas as pd

# Offline TLD extractor using packaged suffix list (no network access required)
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


# Documented list of keywords commonly observed in phishing and credential-harvesting lures
SUSPICIOUS_KEYWORDS: List[str] = [
    "login",
    "verify",
    "verification",
    "update",
    "secure",
    "security",
    "account",
    "password",
    "banking",
    "confirm",
    "confirmation",
    "signin",
    "authenticate",
    "wallet",
    "recover",
    "billing",
    "support",
    "service",
    "webscr",
    "paypal",
    "ebay",
    "amazon",
    "apple",
    "microsoft",
    "netflix",
]

# Standard HTTP/HTTPS ports
STANDARD_PORTS = {80, 443, 8080}

# Feature names in deterministic canonical order
FEATURE_NAMES: List[str] = [
    "url_length",
    "hostname_length",
    "domain_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_digits",
    "num_special_chars",
    "num_subdomains",
    "num_slashes",
    "num_question_marks",
    "num_equals",
    "num_ampersands",
    "num_at_symbols",
    "num_percent",
    "is_https",
    "has_ip_address",
    "has_suspicious_keyword",
    "num_suspicious_keywords",
    "has_suspicious_patterns",
    "has_unusual_port",
    "has_abnormal_hostname_length",
]


def is_ip_address(hostname: str) -> int:
    """
    Check if hostname is a valid IPv4 or IPv6 address.
    
    Args:
        hostname: The host string to inspect.
        
    Returns:
        1 if the hostname is an IP address, 0 otherwise.
    """
    if not hostname:
        return 0
    clean_host = hostname.strip("[]")  # Strip brackets for IPv6
    try:
        ipaddress.ip_address(clean_host)
        return 1
    except ValueError:
        return 0


def normalize_url(url: Any) -> str:
    """
    Safely normalize raw URL input into a clean string.
    
    Args:
        url: Raw input object (string, float, etc.)
        
    Returns:
        Stripped string, or empty string if input is invalid.
    """
    if url is None:
        return ""
    if not isinstance(url, str):
        url = str(url)
    return url.strip()


def extract_features_from_url(raw_url: str) -> Dict[str, Any]:
    """
    Extract all numerical features from a single raw URL safely.
    Handles URLs with or without scheme, malformed URLs, Unicode, and empty strings.
    
    Args:
        raw_url: Raw URL string.
        
    Returns:
        Dictionary containing all engineered numerical features.
    """
    url = normalize_url(raw_url)
    
    if not url:
        # Return zeroed feature vector for empty/invalid input
        return {feat: 0 for feat in FEATURE_NAMES}

    # Ensure URL has a scheme for safe urllib parsing
    has_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url))
    url_to_parse = url if has_scheme else "http://" + url

    try:
        parsed = urlparse(url_to_parse)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
        port = parsed.port
    except Exception:
        hostname = ""
        path = ""
        query = ""
        port = None

    # TLD and Domain extraction using offline tldextract
    try:
        extracted = _TLD_EXTRACTOR(url_to_parse)
        domain = extracted.domain or ""
        subdomain = extracted.subdomain or ""
        suffix = extracted.suffix or ""
    except Exception:
        domain = ""
        subdomain = ""
        suffix = ""

    # Feature 1: Total URL length
    url_length = len(url)

    # Feature 2: Hostname length
    hostname_length = len(hostname)

    # Feature 3: Domain length (registered domain including suffix if present)
    full_domain = f"{domain}.{suffix}" if suffix and domain else (domain or hostname)
    domain_length = len(full_domain)

    # Feature 4-8: Character counts across the URL
    num_dots = url.count(".")
    num_hyphens = url.count("-")
    num_underscores = url.count("_")
    num_digits = sum(c.isdigit() for c in url)
    special_chars = set("@?=&%!~+#$;:/_-.")
    num_special_chars = sum(c in special_chars for c in url)

    # Feature 9: Number of subdomains
    if subdomain:
        num_subdomains = len(subdomain.split("."))
    else:
        num_subdomains = 0

    # Feature 10-15: Specific delimiter counts
    num_slashes = url.count("/")
    num_question_marks = url.count("?")
    num_equals = url.count("=")
    num_ampersands = url.count("&")
    num_at_symbols = url.count("@")
    num_percent = url.count("%")

    # Feature 16: HTTPS usage
    is_https = 1 if url.lower().startswith("https://") else 0

    # Feature 17: Hostname is IP address
    has_ip = is_ip_address(hostname)

    # Feature 18 & 19: Suspicious keyword presence
    url_lower = url.lower()
    keyword_matches = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
    has_suspicious_keyword = 1 if keyword_matches > 0 else 0
    num_suspicious_keywords = keyword_matches

    # Feature 20: Suspicious structural patterns
    # Patterns: double slash in path (redirect lure), @ symbol, multiple hyphens in domain, etc.
    suspicious_patterns = 0
    if "//" in path:
        suspicious_patterns += 1
    if "@" in url:
        suspicious_patterns += 1
    if "-" in domain:
        suspicious_patterns += 1
    if num_subdomains >= 3:
        suspicious_patterns += 1
    has_suspicious_patterns = 1 if suspicious_patterns > 0 else 0

    # Feature 21: Unusual port specified
    if port is not None and port not in STANDARD_PORTS:
        has_unusual_port = 1
    else:
        has_unusual_port = 0

    # Feature 22: Unusually long hostname (often used to obscure domain or embed keywords)
    has_abnormal_hostname_length = 1 if hostname_length > 30 else 0

    return {
        "url_length": int(url_length),
        "hostname_length": int(hostname_length),
        "domain_length": int(domain_length),
        "num_dots": int(num_dots),
        "num_hyphens": int(num_hyphens),
        "num_underscores": int(num_underscores),
        "num_digits": int(num_digits),
        "num_special_chars": int(num_special_chars),
        "num_subdomains": int(num_subdomains),
        "num_slashes": int(num_slashes),
        "num_question_marks": int(num_question_marks),
        "num_equals": int(num_equals),
        "num_ampersands": int(num_ampersands),
        "num_at_symbols": int(num_at_symbols),
        "num_percent": int(num_percent),
        "is_https": int(is_https),
        "has_ip_address": int(has_ip),
        "has_suspicious_keyword": int(has_suspicious_keyword),
        "num_suspicious_keywords": int(num_suspicious_keywords),
        "has_suspicious_patterns": int(has_suspicious_patterns),
        "has_unusual_port": int(has_unusual_port),
        "has_abnormal_hostname_length": int(has_abnormal_hostname_length),
    }


def extract_features_df(urls: pd.Series | List[str]) -> pd.DataFrame:
    """
    Extract features from a collection of URLs into a structured DataFrame.
    Ensures columns match FEATURE_NAMES in exact order.
    
    Args:
        urls: Iterable of URL strings.
        
    Returns:
        Pandas DataFrame with one row per URL and stable columns.
    """
    records = [extract_features_from_url(u) for u in urls]
    df = pd.DataFrame(records)
    # Ensure exact column ordering
    return df[FEATURE_NAMES]
