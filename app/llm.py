from google import genai
from google.genai import types, errors
from app.config import GEMINI_API_KEY, GEMINI_MODEL

# Configure Gemini client with timeout and controlled retries
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=60000,  # 60 seconds per request
        retry_options=types.HttpRetryOptions(
            attempts=2,
            initial_delay=1,
            max_delay=4,
            exp_base=2,
            http_status_codes=[408, 429, 500, 502, 503, 504]
        )
    )
)


class GeminiServiceError(Exception):
    """Raised when Gemini cannot complete a request."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def generate_text(prompt: str) -> str:
    """Generate text using Gemini with controlled retries and errors."""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        if not response.text:
            raise GeminiServiceError(
                "Gemini returned an empty response.",
                status_code=502
            )

        return response.text.strip()

    except GeminiServiceError:
        raise

    except errors.APIError as error:
        status_code = getattr(error, "code", None)

        if status_code == 429:
            raise GeminiServiceError(
                "Gemini rate limit or quota exceeded. Try again later.",
                status_code=429
            ) from error

        if status_code in (408, 504):
            raise GeminiServiceError(
                "Gemini request timed out. Please try again.",
                status_code=504
            ) from error

        if status_code is not None and status_code >= 500:
            raise GeminiServiceError(
                "Gemini service is temporarily unavailable.",
                status_code=503
            ) from error

        raise GeminiServiceError(
            "Gemini API request failed.",
            status_code=502
        ) from error

    except TimeoutError as error:
        raise GeminiServiceError(
            "Gemini request timed out. Please try again.",
            status_code=504
        ) from error

    except Exception as error:
        raise GeminiServiceError(
            "An unexpected error occurred while contacting Gemini.",
            status_code=502
        ) from error


if __name__ == "__main__":
    answer = generate_text(
        "Explain RAG in three simple sentences."
    )

    print("Gemini Response:")
    print(answer)
