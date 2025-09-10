"""
Configuration loader for environment variables and settings.
"""

import os
from typing import Any, Dict


class ConfigLoader:
    """Loads and manages configuration from environment variables."""

    def __init__(self):
        """Initialize the config loader."""
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        Returns:
            Configuration dictionary
        """
        config = {
            # Pulze API configuration
            "PULZE_API_KEY": os.getenv("PULZE_API_KEY"),
            "PULZE_BASE_URL": os.getenv("PULZE_BASE_URL", "https://api.pulze.ai/v1"),
            # OpenAI API configuration (also used for other OpenAI-compatible endpoints)
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            # Default settings
            "DEFAULT_TEMPLATE": os.getenv("DEFAULT_TEMPLATE", "default"),
            "DEFAULT_RATER_MODEL": os.getenv("DEFAULT_RATER_MODEL", "gpt-4"),
            "RESULTS_DIR": os.getenv("RESULTS_DIR", "results"),
            "MAX_RETRIES": int(os.getenv("MAX_RETRIES", "3")),
            "REQUEST_TIMEOUT": int(os.getenv("REQUEST_TIMEOUT", "60")),
        }

        return config

    def get_config(self) -> Dict[str, Any]:
        """
        Get the full configuration dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def validate_config(self) -> Dict[str, str]:
        """
        Validate the configuration and return any errors.

        Returns:
            Dictionary of validation errors (empty if valid)
        """
        errors = {}

        # Check for required API keys
        if not self.config.get("PULZE_API_KEY") and not self.config.get("OPENAI_API_KEY"):
            errors["api_keys"] = "At least one API key (PULZE_API_KEY or OPENAI_API_KEY) must be provided"

        # Validate URLs
        pulze_url = self.config.get("PULZE_BASE_URL")
        if pulze_url and not pulze_url.startswith(("http://", "https://")):
            errors["pulze_url"] = "PULZE_BASE_URL must start with http:// or https://"

        openai_url = self.config.get("OPENAI_BASE_URL")
        if openai_url and not openai_url.startswith(("http://", "https://")):
            errors["openai_url"] = "OPENAI_BASE_URL must start with http:// or https://"

        # Validate numeric values
        try:
            max_retries = self.config.get("MAX_RETRIES", 3)
            if not isinstance(max_retries, int) or max_retries < 0:
                errors["max_retries"] = "MAX_RETRIES must be a non-negative integer"
        except (ValueError, TypeError):
            errors["max_retries"] = "MAX_RETRIES must be a valid integer"

        try:
            timeout = self.config.get("REQUEST_TIMEOUT", 60)
            if not isinstance(timeout, int) or timeout <= 0:
                errors["timeout"] = "REQUEST_TIMEOUT must be a positive integer"
        except (ValueError, TypeError):
            errors["timeout"] = "REQUEST_TIMEOUT must be a valid integer"

        return errors

    def print_config_summary(self):
        """Print a summary of the current configuration."""
        print("Configuration Summary:")
        print("=" * 50)

        # API Keys (masked for security)
        pulze_key = self.config.get("PULZE_API_KEY")
        openai_key = self.config.get("OPENAI_API_KEY")

        print(f"PULZE_API_KEY: {'***' + pulze_key[-4:] if pulze_key else 'Not set'}")
        print(f"OPENAI_API_KEY: {'***' + openai_key[-4:] if openai_key else 'Not set'}")

        # URLs
        print(f"PULZE_BASE_URL: {self.config.get('PULZE_BASE_URL')}")
        print(f"OPENAI_BASE_URL: {self.config.get('OPENAI_BASE_URL')}")

        # Other settings
        print(f"DEFAULT_TEMPLATE: {self.config.get('DEFAULT_TEMPLATE')}")
        print(f"DEFAULT_RATER_MODEL: {self.config.get('DEFAULT_RATER_MODEL')}")
        print(f"RESULTS_DIR: {self.config.get('RESULTS_DIR')}")
        print(f"MAX_RETRIES: {self.config.get('MAX_RETRIES')}")
        print(f"REQUEST_TIMEOUT: {self.config.get('REQUEST_TIMEOUT')}")

        print("=" * 50)

    def set_config_value(self, key: str, value: Any):
        """
        Set a configuration value (for testing or runtime changes).

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value

    def reload_config(self):
        """Reload configuration from environment variables."""
        self.config = self._load_config()

    @staticmethod
    def create_env_template(filename: str = ".env.template"):
        """
        Create a template .env file with all configuration options.

        Args:
            filename: Name of the template file to create
        """
        template_content = """# Pulze API Configuration
PULZE_API_KEY=your_pulze_api_key_here
PULZE_BASE_URL=https://api.pulze.ai/v1

# OpenAI API Configuration (also used for other OpenAI-compatible endpoints)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Default Settings
DEFAULT_TEMPLATE=default
DEFAULT_RATER_MODEL=gpt-4
RESULTS_DIR=results
MAX_RETRIES=3
REQUEST_TIMEOUT=60
"""

        with open(filename, "w") as f:
            f.write(template_content)

        print(f"Environment template created: {filename}")
        print("Copy this file to .env and fill in your API keys.")
