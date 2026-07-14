import os
import time

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI


# Easy-to-edit model configuration
MODEL = "gpt-4o-mini"
MAX_TOKENS = 150
TEMPERATURE = 0.7


def _get_retry_wait_seconds(error: APIStatusError) -> float:
	retry_after = None
	response = getattr(error, "response", None)
	if response is not None and getattr(response, "headers", None):
		retry_after = response.headers.get("retry-after")

	if retry_after is None:
		return 1.0

	try:
		return max(float(retry_after), 0.0)
	except (TypeError, ValueError):
		return 1.0


def main() -> None:
	load_dotenv()

	client = OpenAI(
		base_url="https://models.inference.ai.azure.com",
		api_key=os.getenv("GITHUB_TOKEN"),
	)

	with open("document.txt", "r", encoding="utf-8") as file:
		document_text = file.read()

	completion = None
	for attempt in range(2):
		try:
			completion = client.chat.completions.create(
				model=MODEL,
				messages=[
					{
						"role": "system",
						"content": "You are a precise summariser. Answer strictly in 3 bullet points.",
					},
					{"role": "user", "content": document_text},
				],
				max_tokens=MAX_TOKENS,
				temperature=TEMPERATURE,
			)
			break
		except APIStatusError as error:
			if error.status_code == 401:
				print("Check your API key")
				return

			if error.status_code == 429 and attempt == 0:
				wait_seconds = _get_retry_wait_seconds(error)
				print(f"Rate limited. Waiting {wait_seconds} seconds before retrying...")
				time.sleep(wait_seconds)
				continue

			raise

	if completion is None:
		return

	response_text = ""
	if getattr(completion, "choices", None):
		first_choice = completion.choices[0]
		if first_choice and getattr(first_choice, "message", None):
			response_text = first_choice.message.content or ""

	print(response_text)


if __name__ == "__main__":
	main()
