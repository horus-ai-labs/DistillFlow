import anthropic
import os
from .teacher import TeacherModel

class AnthropicTeacher(TeacherModel):
    """
    AnthropicTeacher interacts with the Anthropic public APIs to generate responses.
    """
    def __init__(self, api_key=None):
        """
        Initialize the teacher with Anthropic API key.
        Args:
            api_key: Optional API key for Anthropic. If not provided, it will be fetched from environment variable ANTHROPIC_API_KEY.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate_response(self, prompt, max_tokens=150):
        """
        Generate a response using the Anthropic model.
        Args:
            prompt: Input prompt for the model.
            max_tokens: Maximum number of tokens for the generated response.
        Returns:
            Generated response from the Anthropic model.
        """
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=max_tokens,
            system="You are a seasoned teacher. Provide a concise answer that your students can understand.",
            messages=[
                {"role": "user","content": prompt}
            ],
        )

        print(f"Response from Anthropic: {response}")
        return response.content
