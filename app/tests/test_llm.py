
import pytest
from unittest.mock import patch

from google.genai import errors
from app.llm import generate_text, GeminiServiceError


class MockGeminiAPIError(errors.APIError):
    """Actual APIError subclass for testing."""

    def __init__(self, code):
        Exception.__init__(self, f"Mock Gemini error: {code}")
        self.code = code


@pytest.mark.parametrize(
    "api_code, expected_status",
    [
        (429, 429),
        (408, 504),
        (504, 504),
        (500, 503),
        (503, 503),
        (502, 503),
    ]
)
def test_gemini_api_error_mapping(api_code, expected_status):

    with patch(
        "app.llm.client.models.generate_content",
        side_effect=MockGeminiAPIError(api_code)
    ):
        with pytest.raises(GeminiServiceError) as error:
            generate_text("Test prompt")

    assert error.value.status_code == expected_status


def test_gemini_success():

    class MockResponse:
        text = "  Gemini is working.  "

    with patch(
        "app.llm.client.models.generate_content",
        return_value=MockResponse()
    ):
        result = generate_text("Test prompt")

    assert result == "Gemini is working."


def test_empty_gemini_response():

    class MockResponse:
        text = None

    with patch(
        "app.llm.client.models.generate_content",
        return_value=MockResponse()
    ):
        with pytest.raises(GeminiServiceError) as error:
            generate_text("Test prompt")

    assert error.value.status_code == 502
