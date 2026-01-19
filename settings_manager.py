"""
Settings Manager for PyPotteryLens
Handles persistent storage of global application settings including API keys.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import os


class SettingsManager:
    """Manages global application settings stored in user's home directory."""

    def __init__(self, settings_dir: Optional[Path] = None):
        """
        Initialize settings manager.

        Args:
            settings_dir: Custom settings directory. Defaults to ~/.pypotterylens/
        """
        if settings_dir:
            self.settings_dir = Path(settings_dir)
        else:
            self.settings_dir = Path.home() / '.pypotterylens'

        self.settings_file = self.settings_dir / 'settings.json'
        self._ensure_settings_dir()

    def _ensure_settings_dir(self) -> None:
        """Create settings directory if it doesn't exist."""
        self.settings_dir.mkdir(parents=True, exist_ok=True)

    def _default_settings(self) -> Dict[str, Any]:
        """Return default settings structure."""
        return {
            'anthropic_api_key': None,
            'openai_api_key': None,
            'gemini_api_key': None,
            'deepseek_api_key': None,
            'together_api_key': None,
            'lmstudio_base_url': 'http://localhost:1234/v1',
            'lmstudio_model': '',
            'ollama_base_url': 'http://localhost:11434',
            'ollama_model': 'llava',
            'default_ai_provider': 'anthropic',
            'created_at': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat()
        }

    def get_settings(self) -> Dict[str, Any]:
        """
        Load and return current settings.

        Returns:
            Dictionary containing all settings.
        """
        if not self.settings_file.exists():
            return self._default_settings()

        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                # Merge with defaults to ensure all keys exist
                defaults = self._default_settings()
                for key, value in defaults.items():
                    if key not in settings:
                        settings[key] = value
                return settings
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading settings: {e}")
            return self._default_settings()

    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Save settings to file.

        Args:
            settings: Dictionary of settings to save.

        Returns:
            True if successful, False otherwise.
        """
        try:
            settings['last_modified'] = datetime.now().isoformat()
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            print(f"Error saving settings: {e}")
            return False

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for specified provider.

        Args:
            provider: 'anthropic', 'openai', 'gemini', or 'deepseek'

        Returns:
            API key string or None if not set.
        """
        valid_providers = ['anthropic', 'openai', 'gemini', 'deepseek', 'together']
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider: {provider}. Must be one of {valid_providers}")

        settings = self.get_settings()
        return settings.get(f'{provider}_api_key')

    def set_api_key(self, provider: str, key: str) -> bool:
        """
        Set API key for specified provider.

        Args:
            provider: 'anthropic', 'openai', 'gemini', or 'deepseek'
            key: The API key to store

        Returns:
            True if successful, False otherwise.
        """
        valid_providers = ['anthropic', 'openai', 'gemini', 'deepseek', 'together']
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider: {provider}. Must be one of {valid_providers}")

        settings = self.get_settings()
        settings[f'{provider}_api_key'] = key
        return self.save_settings(settings)

    def get_default_provider(self) -> str:
        """
        Get the default AI provider.

        Returns:
            Provider name.
        """
        settings = self.get_settings()
        return settings.get('default_ai_provider', 'anthropic')

    def set_default_provider(self, provider: str) -> bool:
        """
        Set the default AI provider.

        Args:
            provider: 'anthropic', 'openai', 'gemini', 'deepseek', 'lmstudio', or 'ollama'

        Returns:
            True if successful, False otherwise.
        """
        valid_providers = ['anthropic', 'openai', 'gemini', 'deepseek', 'together', 'lmstudio', 'ollama']
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider: {provider}. Must be one of {valid_providers}")

        settings = self.get_settings()
        settings['default_ai_provider'] = provider
        return self.save_settings(settings)

    def has_api_key(self, provider: str) -> bool:
        """
        Check if an API key is configured for the provider.

        Args:
            provider: 'anthropic', 'openai', 'gemini', or 'deepseek'

        Returns:
            True if key exists and is not empty.
        """
        # Local providers don't need API keys
        if provider in ['lmstudio', 'ollama']:
            return True
        key = self.get_api_key(provider)
        return key is not None and len(key) > 0

    def get_masked_key(self, provider: str) -> Optional[str]:
        """
        Get a masked version of the API key for display.

        Args:
            provider: 'anthropic', 'openai', 'gemini', or 'deepseek'

        Returns:
            Masked key string (e.g., '***abcd') or None.
        """
        # Local providers don't have API keys
        if provider in ['lmstudio', 'ollama']:
            return '[Local - No API Key]'
        key = self.get_api_key(provider)
        if not key:
            return None

        if len(key) <= 8:
            return '***' + key[-4:] if len(key) >= 4 else '***'

        return '***' + key[-4:]

    def delete_api_key(self, provider: str) -> bool:
        """
        Remove API key for specified provider.

        Args:
            provider: 'anthropic', 'openai', 'gemini', or 'deepseek'

        Returns:
            True if successful, False otherwise.
        """
        valid_providers = ['anthropic', 'openai', 'gemini', 'deepseek', 'together']
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider: {provider}. Must be one of {valid_providers}")

        settings = self.get_settings()
        settings[f'{provider}_api_key'] = None
        return self.save_settings(settings)

    def get_local_provider_settings(self, provider: str) -> Dict[str, Any]:
        """
        Get settings for local AI providers (LM Studio, Ollama).

        Args:
            provider: 'lmstudio' or 'ollama'

        Returns:
            Dictionary with base_url and model settings.
        """
        if provider not in ['lmstudio', 'ollama']:
            raise ValueError(f"Invalid local provider: {provider}. Must be 'lmstudio' or 'ollama'")

        settings = self.get_settings()
        return {
            'base_url': settings.get(f'{provider}_base_url', ''),
            'model': settings.get(f'{provider}_model', '')
        }

    def set_local_provider_settings(self, provider: str, base_url: str = None, model: str = None) -> bool:
        """
        Set settings for local AI providers (LM Studio, Ollama).

        Args:
            provider: 'lmstudio' or 'ollama'
            base_url: The API endpoint URL
            model: The model name to use

        Returns:
            True if successful, False otherwise.
        """
        if provider not in ['lmstudio', 'ollama']:
            raise ValueError(f"Invalid local provider: {provider}. Must be 'lmstudio' or 'ollama'")

        settings = self.get_settings()
        if base_url is not None:
            settings[f'{provider}_base_url'] = base_url
        if model is not None:
            settings[f'{provider}_model'] = model
        return self.save_settings(settings)

    def get_calibration_settings(self) -> Dict[str, Any]:
        """
        Get calibration settings.

        Returns:
            Dictionary with calibration configuration.
        """
        settings = self.get_settings()
        return settings.get('calibration', {
            'default_pixels_per_cm': None,
            'per_image': {}
        })

    def set_calibration(self, pixels_per_cm: float, image_name: Optional[str] = None) -> bool:
        """
        Set calibration value.

        Args:
            pixels_per_cm: Calibration value
            image_name: If provided, sets per-image calibration. Otherwise sets default.

        Returns:
            True if successful, False otherwise.
        """
        settings = self.get_settings()

        if 'calibration' not in settings:
            settings['calibration'] = {
                'default_pixels_per_cm': None,
                'per_image': {}
            }

        if image_name:
            settings['calibration']['per_image'][image_name] = {
                'pixels_per_cm': pixels_per_cm,
                'manual': True,
                'set_at': datetime.now().isoformat()
            }
        else:
            settings['calibration']['default_pixels_per_cm'] = pixels_per_cm

        return self.save_settings(settings)


# Singleton instance for easy access
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get or create the singleton settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager
