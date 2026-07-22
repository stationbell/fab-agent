"""Domain-specific exceptions exposed by fab-agent."""


class FabAgentError(Exception):
    """Base error for expected application failures."""


class ConfigurationError(FabAgentError):
    """Configuration is missing or invalid."""


class DimensionParseError(FabAgentError, ValueError):
    """A source dimension cannot be parsed exactly."""


class ImageInputError(FabAgentError):
    """An input image failed deterministic validation."""


class ModelError(FabAgentError):
    """A configured model call failed or returned invalid output."""


class RunStateError(FabAgentError):
    """A requested operation is invalid for the run's current state."""


class StorageError(FabAgentError):
    """Run persistence failed."""


class ArtifactError(FabAgentError):
    """A deterministic artifact could not be generated."""
