
"""
Unit tests for cleaned phishing feature-extraction helpers.

Run from the repository root with:

    pytest -q

These tests address the lecturer's requested checks:
- IP detection
- iframe detection
- right-click detection
- WHOIS fallback behavior
- output schema consistency
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import feature_extraction_clean as clean


class FakeResponse:
    def __init__(self, text="", history=None):
        self.text = text
        self.history = history or []


class FakeWhoisRecord:
    def __init__(self, creation_date, expiration_date):
        self.creation_date = creation_date
        self.expiration_date = expiration_date


def test_ip_detection_detects_bare_ip_address():
    assert clean.havingIP("192.168.1.10") == 1


def test_ip_detection_detects_ip_hostname_inside_complete_url():
    assert clean.havingIP("http://192.168.1.10/login") == 1
    assert clean.havingIP("https://8.8.8.8/account") == 1


def test_ip_detection_does_not_flag_normal_domain():
    assert clean.havingIP("https://example.com/login") == 0


def test_iframe_detection_distinguishes_plain_html_from_iframe_html():
    plain_response = FakeResponse("<html><body>No iframe here</body></html>")
    iframe_response = FakeResponse(
        "<html><body><iframe src='https://evil.example'></iframe></body></html>"
    )

    assert clean.iframe(plain_response) == 0
    assert clean.iframe(iframe_response) == 1


def test_iframe_detection_treats_missing_response_as_suspicious():
    assert clean.iframe("") == 1
    assert clean.iframe(None) == 1


def test_right_click_detection_distinguishes_normal_script_from_blocking_script():
    normal_response = FakeResponse("<script>console.log('hello')</script>")
    blocking_response = FakeResponse(
        "<script>if (event.button == 2) { return false; }</script>"
    )

    assert clean.rightClick(normal_response) == 0
    assert clean.rightClick(blocking_response) == 1


def test_right_click_detection_treats_missing_response_as_suspicious():
    assert clean.rightClick("") == 1
    assert clean.rightClick(None) == 1


def test_whois_fallback_behavior_when_lookup_is_missing():
    result = clean.get_domain_based_features(
        url="https://example.com/login",
        whois_lookup=None,
    )

    assert result["DNS_Record"] == 1
    assert result["Domain_Age"] == 1
    assert result["Domain_End"] == 1
    assert result["WHOIS_Fallback_Used"] is True


def test_whois_fallback_behavior_when_lookup_raises_exception():
    def failing_whois_lookup(domain):
        raise RuntimeError("WHOIS service unavailable")

    result = clean.get_domain_based_features(
        url="https://example.com/login",
        whois_lookup=failing_whois_lookup,
    )

    assert result["DNS_Record"] == 1
    assert result["Domain_Age"] == 1
    assert result["Domain_End"] == 1
    assert result["WHOIS_Fallback_Used"] is True


def test_whois_success_path_returns_no_fallback():
    def successful_whois_lookup(domain):
        return FakeWhoisRecord(
            creation_date=datetime.now() - timedelta(days=1000),
            expiration_date=datetime.now() + timedelta(days=500),
        )

    result = clean.get_domain_based_features(
        url="https://example.com/login",
        whois_lookup=successful_whois_lookup,
    )

    assert result["DNS_Record"] == 0
    assert result["WHOIS_Fallback_Used"] is False
    assert result["Domain_Age"] in [0, 1]
    assert result["Domain_End"] in [0, 1]


def test_output_schema_contains_exactly_16_predictors():
    response = FakeResponse("<html><body>normal page</body></html>")

    features = clean.featureExtraction(
        url="https://example.com/login",
        response=response,
        whois_lookup=None,
        web_traffic_value=1,
    )

    assert isinstance(features, list)
    assert len(features) == 16
    assert len(features) == len(clean.FEATURE_COLUMNS)


def test_feature_dict_schema_matches_feature_columns():
    response = FakeResponse("<html><body>normal page</body></html>")

    feature_dictionary = clean.feature_dict(
        url="https://example.com/login",
        response=response,
        whois_lookup=None,
        web_traffic_value=1,
    )

    assert list(feature_dictionary.keys()) == clean.FEATURE_COLUMNS
    assert len(feature_dictionary) == 16


def test_full_dataset_schema_is_domain_plus_features_plus_label():
    assert clean.FULL_DATASET_COLUMNS[0] == "Domain"
    assert clean.FULL_DATASET_COLUMNS[-1] == "Label"
    assert clean.FULL_DATASET_COLUMNS[1:-1] == clean.FEATURE_COLUMNS
    assert len(clean.FULL_DATASET_COLUMNS) == 18

