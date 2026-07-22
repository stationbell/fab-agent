"""Application model-client construction."""

from fab_agent.config import AppConfig
from fab_agent.models.ollama import OllamaModelClient


def build_model_client(config: AppConfig) -> OllamaModelClient:
    return OllamaModelClient(
        config.ollama,
        retries=config.workflow.structured_output_retry_count,
    )
