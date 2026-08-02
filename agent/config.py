import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
WIKI_USER_AGENT = os.getenv(
    "WIKI_USER_AGENT",
    "cheetah-agent/0.1 (https://github.com/example/cheetah-agent; contact@example.com)",
)
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
