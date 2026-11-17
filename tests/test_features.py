"""
Automated Unit Tests for URL Feature Extraction.

Validates that feature extraction correctly handles diverse URL schemes,
syntactic delimiters, IP addresses, Unicode characters, and malformed inputs
without accessing external networks or crashing.
"""

import pytest
import pandas as pd
from src.feature_extraction import (
    extract_features_from_url,
    extract_features_df,
    normalize_url,
    is_ip_address,
    FEATURE_NAMES,
    SUSPICIOUS_KEYWORDS,
)


def test_feature_names_order_and_count():
    """Verify that feature names list is non-empty, unique, and strictly defined."""
    assert len(FEATURE_NAMES) == 22
    assert len(set(FEATURE_NAMES)) == 22
    # Check key feature names are present
    assert "url_length" in FEATURE_NAMES
    assert "hostname_length" in FEATURE_NAMES
    assert "is_https" in FEATURE_NAMES
    assert "has_ip_address" in FEATURE_NAMES
    assert "has_suspicious_keyword" in FEATURE_NAMES


def test_empty_and_invalid_inputs():
    """Empty strings, whitespace, or None must safely produce a zeroed feature vector."""
    for empty_val in ["", "   ", None]:
        feats = extract_features_from_url(empty_val)
        assert isinstance(feats, dict)
        assert list(feats.keys()) == FEATURE_NAMES
        assert feats["url_length"] == 0
        assert feats["is_https"] == 0
        assert feats["has_ip_address"] == 0


def test_url_with_http_and_https_schemes():
    """Verify scheme parsing for both http and https URLs."""
    http_url = "http://example.com/test/page"
    https_url = "https://example.com/test/page"

    f_http = extract_features_from_url(http_url)
    f_https = extract_features_from_url(https_url)

    assert f_http["is_https"] == 0
    assert f_https["is_https"] == 1
    assert f_http["hostname_length"] == len("example.com")
    assert f_https["hostname_length"] == len("example.com")


def test_url_without_scheme():
    """Ensure schemeless URLs (e.g., 'example.com/path') are parsed safely without crashing."""
    schemeless = "github.com/torvalds/linux"
    feats = extract_features_from_url(schemeless)

    assert feats["url_length"] == len(schemeless)
    assert feats["is_https"] == 0
    assert feats["hostname_length"] == len("github.com")
    assert feats["num_slashes"] == 2


def test_url_with_query_parameters_and_fragments():
    """Validate parsing of query delimiters and fragment identifiers."""
    url = "https://portal.bank.com/signin?session=xyz123&user=admin#auth-token"
    feats = extract_features_from_url(url)

    assert feats["num_question_marks"] == 1
    assert feats["num_equals"] == 2
    assert feats["num_ampersands"] == 1
    assert feats["has_suspicious_keyword"] == 1
    assert feats["num_suspicious_keywords"] >= 1


def test_ip_address_hostnames():
    """Test IPv4 and IPv6 detection in hostnames."""
    ipv4_url = "http://192.168.1.50:8080/admin"
    ipv6_url = "http://[2001:db8::1]:80/index.html"
    regular_url = "http://www.google.com/index.html"

    f_ipv4 = extract_features_from_url(ipv4_url)
    f_ipv6 = extract_features_from_url(ipv6_url)
    f_reg = extract_features_from_url(regular_url)

    assert f_ipv4["has_ip_address"] == 1
    assert f_ipv6["has_ip_address"] == 1
    assert f_reg["has_ip_address"] == 0
    assert is_ip_address("192.168.1.50") == 1
    assert is_ip_address("[2001:db8::1]") == 1
    assert is_ip_address("example.com") == 0


def test_unusual_port():
    """Validate detection of unusual non-standard HTTP ports."""
    standard_80 = "http://example.com:80/home"
    standard_443 = "https://example.com:443/home"
    unusual_port = "http://example.com:8888/login"

    assert extract_features_from_url(standard_80)["has_unusual_port"] == 0
    assert extract_features_from_url(standard_443)["has_unusual_port"] == 0
    assert extract_features_from_url(unusual_port)["has_unusual_port"] == 1


def test_unicode_domain_names():
    """Ensure internationalized/Unicode domain names do not cause unhandled exceptions."""
    unicode_url = "https://münchen.de/rathaus"
    feats = extract_features_from_url(unicode_url)

    assert isinstance(feats, dict)
    assert feats["url_length"] == len(unicode_url)
    assert feats["is_https"] == 1


def test_at_symbol_and_suspicious_patterns():
    """Verify that @ symbols and redirection patterns are captured."""
    lure_url = "http://legit-site.com@attacker-controlled.net/login//redirect"
    feats = extract_features_from_url(lure_url)

    assert feats["num_at_symbols"] >= 1
    assert feats["has_suspicious_patterns"] == 1
    assert feats["has_suspicious_keyword"] == 1


def test_suspicious_keywords_detection():
    """Test recognition of credential lures."""
    for kw in ["login", "verify", "password", "banking", "secure", "update"]:
        sample_url = f"https://service-center.com/{kw}/index.html"
        feats = extract_features_from_url(sample_url)
        assert feats["has_suspicious_keyword"] == 1, f"Failed to detect keyword '{kw}'"
        assert feats["num_suspicious_keywords"] >= 1


def test_extract_features_df_output_structure():
    """Verify extract_features_df produces stable column schema identical to FEATURE_NAMES."""
    urls = [
        "https://google.com",
        "http://phish-site.org/verify/account",
        "192.168.0.1/test",
    ]
    df = extract_features_df(urls)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, len(FEATURE_NAMES))
    assert list(df.columns) == FEATURE_NAMES
    # All values must be numeric
    assert all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
