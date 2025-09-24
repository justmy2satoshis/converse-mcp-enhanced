#!/usr/bin/env python3
"""
Optimized API Manager for Converse-MCP
Priority: Ollama (FREE) -> User Preference -> Cost-based Fallback
NO PLACEHOLDER RESPONSES - Fail cleanly with helpful errors
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import httpx
from datetime import datetime, timedelta

# Import our new OllamaManager
from ollama_manager import OllamaManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Exhaustive Model Configuration Dictionary
PROVIDER_MODELS = {
    "ollama": {
        "api_endpoint": "http://localhost:11434",
        "models": [
            # Llama Series
            "llama3.3:70b", "llama3.2-vision:11b", "llama3.2-vision:90b",
            "llama3.1:8b", "llama3.1:70b", "llama3:8b", "llama3:70b",
            "llama2:7b", "llama2:13b", "llama2:70b",
            # Mistral Series
            "mistral:7b", "mistral-small:3b", "mistral-small:3.1",
            "mistral-large:2", "mixtral:8x7b", "mixtral:8x22b",
            # Qwen Series
            "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b", "qwen2.5-coder",
            # Google Models
            "gemma2:2b", "gemma2:9b", "gemma2:27b", "gemma:2b", "gemma:7b",
            # Microsoft
            "phi3:mini", "phi3:medium", "phi-3.5:3.8b",
            # Specialized Models
            "codellama:7b", "codellama:13b", "codellama:34b",
            "starcoder:7b", "deepseek-coder:6.7b", "deepseek-coder:33b",
            # Multimodal
            "llava:7b", "llava:13b", "llava:34b",
            # Small Models
            "smollm2:135m", "smollm2:360m", "smollm2:1.7b",
        ]
    },
    "anthropic": {
        "api_endpoint": "https://api.anthropic.com/v1",
        "models": [
            "claude-opus-4-1-20250805", "claude-opus-4", "claude-sonnet-4",
            "claude-3.7-sonnet", "claude-3.5-sonnet-20241022", "claude-3.5-haiku-20241022",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
        ]
    },
    "openai": {
        "api_endpoint": "https://api.openai.com/v1",
        "models": [
            "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-chat",
            "o3-pro", "o3", "o3-mini", "o1", "o1-mini",
            "gpt-4.5", "gpt-4.1", "gpt-4.1-mini", "gpt-4o",
            "gpt-4-turbo-2024-04-09", "gpt-4-turbo", "gpt-4-32k", "gpt-4",
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-instruct"
        ]
    },
    "google": {
        "api_endpoint": "https://generativelanguage.googleapis.com/v1",
        "models": [
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
            "gemini-2.5-flash-image", "gemini-2.5-pro-preview-tts",
            "gemini-2.0-flash", "gemini-2.0-flash-lite",
            "gemini-2.0-flash-preview-image-generation", "gemini-2.0-flash-live"
        ]
    },
    "xai": {
        "api_endpoint": "https://api.x.ai/v1",
        "models": [
            "grok-4-0709", "grok-4-fast", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning",
            "grok-3", "grok-3-mini", "grok-2-1212", "grok-2-vision-1212", "grok-2",
            "grok-code-fast-1", "grok-1"
        ]
    },
    "perplexity": {
        "api_endpoint": "https://api.perplexity.ai",
        "models": [
            "sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro", "sonar-deep-research",
            "sonar-small", "sonar-medium", "sonar-small-chat", "sonar-medium-chat",
            "sonar-small-online", "sonar-medium-online",
            "pplx-7b-online", "pplx-70b-online", "pplx-7b-chat", "pplx-70b-chat"
        ]
    }
}

# Helper function to get all available models for a provider
def get_provider_models(provider_name):
    """Returns list of model IDs for a specific provider"""
    return PROVIDER_MODELS.get(provider_name, {}).get("models", [])

# Helper function to get latest/recommended models
def get_recommended_models():
    """Returns the recommended latest models from each provider"""
    return {
        "xai": ["grok-4-0709", "grok-4-fast", "grok-code-fast-1"],
        "anthropic": ["claude-opus-4-1-20250805", "claude-sonnet-4", "claude-3.7-sonnet"],
        "openai": ["gpt-5", "gpt-5-mini", "o3-mini", "gpt-4.1"],
        "google": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
        "perplexity": ["sonar-pro", "sonar-reasoning-pro", "sonar-deep-research"],
        "ollama": ["llama3.3:70b", "mistral-large:2", "qwen2.5:72b", "gemma2:27b"]
    }

# Model aliases for backward compatibility
MODEL_ALIASES = {
    "gemini-pro": "gemini-2.5-pro",
    "gemini-pro-vision": "gemini-2.5-flash",
    "claude-3-opus": "claude-opus-4-1-20250805",
    "claude-3-sonnet": "claude-sonnet-4",
    "gpt-4-vision-preview": "gpt-4o"
}

# Cost optimization tiers
COST_TIERS = {
    "free": ["ollama/*"],  # All Ollama models
    "low": ["claude-3.5-haiku-20241022", "gpt-3.5-turbo", "gemini-2.5-flash-lite", "sonar-small"],
    "medium": ["claude-sonnet-4", "gpt-5-mini", "gemini-2.5-flash", "sonar-pro"],
    "high": ["claude-opus-4-1-20250805", "gpt-5", "gemini-2.5-pro", "grok-4-0709"]
}

# APIManager alias will be set at the end of the file for compatibility


class ProviderStatus(Enum):
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"
    DISABLED = "disabled"


@dataclass
class ProviderConfig:
    name: str
    priority: int
    enabled: bool
    api_key: Optional[str]
    base_url: Optional[str]
    models: List[str]
    cost_per_1k_tokens: float = 0.0  # Cost tracking
    is_free: bool = False
    fallback_on_error: bool = True
    rate_limit_reset: Optional[datetime] = None
    status: ProviderStatus = ProviderStatus.NOT_CONFIGURED
    last_error: Optional[str] = None
    usage_count: int = 0
    tokens_used: int = 0
    total_cost: float = 0.0


@dataclass
class UsageStats:
    """Track usage and cost savings"""
    total_requests: int = 0
    ollama_requests: int = 0
    paid_requests: int = 0
    total_tokens: int = 0
    ollama_tokens: int = 0
    paid_tokens: int = 0
    total_cost: float = 0.0
    cost_saved: float = 0.0  # Money saved by using Ollama
    timestamp: datetime = field(default_factory=datetime.now)


class OptimizedAPIManager:
    """Manages multiple AI providers with Ollama-first priority"""

    def __init__(self, config_path: Optional[str] = None):
        self.providers: Dict[str, ProviderConfig] = {}
        self.config_path = config_path or "api_config_optimized.json"
        self.usage_stats = UsageStats()
        self.user_preference: Optional[str] = None

        # Initialize Ollama Manager for auto-detection
        self.ollama_manager = OllamaManager()

        # Load user preferences if they exist
        self.load_preferences()

        # Initialize providers with CORRECT priorities
        self.initialize_providers()

        # Test availability
        self.test_provider_availability()

        # Log initialization status
        self.log_initialization_status()

    def load_preferences(self):
        """Load user preferences from config"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.user_preference = config.get("user_preference")
                    logger.info(f"Loaded user preference: {self.user_preference}")
        except Exception as e:
            logger.warning(f"Could not load preferences: {e}")

    def initialize_providers(self):
        """Initialize all supported providers with CORRECT priorities"""

        # PRIORITY 1: Ollama (FREE and LOCAL) - ALWAYS FIRST
        # Use auto-detected models from OllamaManager
        ollama_models = self.ollama_manager.available_models if self.ollama_manager.is_available else []

        self.providers["ollama"] = ProviderConfig(
            name="ollama",
            priority=1,  # HIGHEST PRIORITY
            enabled=self.ollama_manager.is_available,
            api_key=None,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            models=ollama_models,  # Use auto-detected models
            cost_per_1k_tokens=0.0,
            is_free=True,
            status=ProviderStatus.AVAILABLE if self.ollama_manager.is_available else ProviderStatus.NOT_CONFIGURED
        )

        # Other providers with LOWER priorities
        priority_counter = 10  # Start other providers at 10+

        # Anthropic (often preferred for quality)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.providers["anthropic"] = ProviderConfig(
                name="anthropic",
                priority=priority_counter,
                enabled=True,
                api_key=anthropic_key,
                base_url="https://api.anthropic.com/v1",
                models=[
                    # Claude 4 Series (2025)
                    "claude-opus-4-1-20250805",  # Latest Opus 4.1 (August 2025)
                    "claude-opus-4",             # Original Opus 4 (May 2025)
                    "claude-sonnet-4",           # Sonnet 4 (May 2025)
                    # Claude 3.7/3.5 Series (Still Available)
                    "claude-3.7-sonnet",         # Extended thinking capabilities
                    "claude-3.5-sonnet-20241022", # October 2024 update
                    "claude-3.5-haiku-20241022",  # Fast, affordable
                    # Legacy Claude 3 (for backward compatibility)
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                ],
                cost_per_1k_tokens=0.015,  # Claude 3 Sonnet pricing
                status=ProviderStatus.AVAILABLE
            )
            priority_counter += 1

        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.providers["openai"] = ProviderConfig(
                name="openai",
                priority=priority_counter,
                enabled=True,
                api_key=openai_key,
                base_url="https://api.openai.com/v1",
                models=[
                    # GPT-5 Series (2025 - Latest)
                    "gpt-5",                # Flagship model (registration required)
                    "gpt-5-mini",           # Smaller variant
                    "gpt-5-nano",           # Smallest variant
                    "gpt-5-chat",           # Chat-optimized
                    # O-Series Reasoning Models
                    "o3-pro",               # Advanced reasoning (Pro/Team users)
                    "o3",                   # Standard reasoning model
                    "o3-mini",              # Cost-efficient reasoning
                    "o1",                   # Previous gen reasoning
                    "o1-mini",              # Previous gen mini
                    # GPT-4 Series (Still Available)
                    "gpt-4.5",              # Research preview
                    "gpt-4.1",              # Coding specialized
                    "gpt-4.1-mini",         # Fast, efficient
                    "gpt-4o",               # Multimodal
                    "gpt-4-turbo-2024-04-09",
                    "gpt-4-turbo",
                    "gpt-4-32k",
                    "gpt-4",
                    # GPT-3.5 Series
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-16k",
                    "gpt-3.5-turbo-instruct",
                ],
                cost_per_1k_tokens=0.03,  # GPT-4 pricing
                status=ProviderStatus.AVAILABLE
            )
            priority_counter += 1

        # Google Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            self.providers["google"] = ProviderConfig(
                name="google",
                priority=priority_counter,
                enabled=True,
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1",
                models=[
                    # Gemini 2.5 Series (Latest - 2025)
                    "gemini-2.5-pro",           # State-of-the-art thinking model
                    "gemini-2.5-flash",         # Best price-performance
                    "gemini-2.5-flash-lite",    # Most cost-effective
                    "gemini-2.5-flash-image",   # Image generation (preview)
                    "gemini-2.5-pro-preview-tts", # Text-to-speech
                    # Gemini 2.0 Series (Current)
                    "gemini-2.0-flash",         # Multimodal with tool use
                    "gemini-2.0-flash-lite",    # Optimized for efficiency
                    "gemini-2.0-flash-preview-image-generation",
                    "gemini-2.0-flash-live",    # Low-latency voice/video
                    # Legacy names for compatibility
                    "gemini-pro",  # Maps to gemini-2.5-pro
                    "gemini-pro-vision",  # Maps to gemini-2.5-flash
                ],
                cost_per_1k_tokens=0.001,  # Gemini Pro pricing
                status=ProviderStatus.AVAILABLE
            )
            priority_counter += 1

        # XAI (Grok)
        xai_key = os.getenv("XAI_API_KEY")
        if xai_key:
            self.providers["xai"] = ProviderConfig(
                name="xai",
                priority=priority_counter,
                enabled=True,
                api_key=xai_key,
                base_url="https://api.x.ai/v1",
                models=[
                    # Grok 4 Series (Latest)
                    "grok-4-0709",           # Flagship model, unparalleled performance
                    "grok-4-fast",           # Cost-efficient reasoning
                    "grok-4-fast-reasoning", # Fast reasoning model
                    "grok-4-fast-non-reasoning", # Fast non-reasoning variant
                    # Grok 3 Series
                    "grok-3",                # Generally available via API
                    "grok-3-mini",           # Smaller, faster variant
                    # Grok 2 Series
                    "grok-2-1212",           # Better accuracy and multilingual
                    "grok-2-vision-1212",    # Vision capabilities
                    "grok-2",                # Legacy name
                    # Specialized
                    "grok-code-fast-1",      # Agentic coding ($0.20/1M input)
                    # Legacy
                    "grok-1",                # Backward compatibility
                ],
                cost_per_1k_tokens=0.02,  # Estimated
                status=ProviderStatus.AVAILABLE
            )
            priority_counter += 1

        # Perplexity
        perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        if perplexity_key:
            self.providers["perplexity"] = ProviderConfig(
                name="perplexity",
                priority=priority_counter,
                enabled=True,
                api_key=perplexity_key,
                base_url="https://api.perplexity.ai",
                models=[
                    # Sonar Family (Proprietary - 2025)
                    "sonar",                    # Base model (LLaMA 3.1 70B based)
                    "sonar-pro",                # Enhanced precision and context
                    "sonar-reasoning",          # Chain-of-Thought with live search
                    "sonar-reasoning-pro",      # DeepSeek-R1 1776 powered
                    "sonar-deep-research",      # Most advanced for research
                    # Cost-Efficient Variants
                    "sonar-small",
                    "sonar-medium",
                    "sonar-small-chat",
                    "sonar-medium-chat",
                    "sonar-small-online",       # With web search
                    "sonar-medium-online",      # With web search
                    # Legacy Models (Backward compatibility)
                    "pplx-7b-online",
                    "pplx-70b-online",
                    "pplx-7b-chat",
                    "pplx-70b-chat",
                ],
                cost_per_1k_tokens=0.005,  # Estimated
                status=ProviderStatus.AVAILABLE
            )

        logger.info(f"Initialized {len(self.providers)} providers with Ollama as priority #1")

    def test_provider_availability(self):
        """Test each provider to see if it's available"""

        # CRITICAL: Test Ollama FIRST and log prominently
        if self.providers.get("ollama"):
            try:
                response = httpx.get(f"{self.providers['ollama'].base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    installed_models = [m["name"] for m in models]

                    # Keep track of both potential and installed models
                    self.providers["ollama"].models = installed_models if installed_models else self.providers["ollama"].models
                    self.providers["ollama"].status = ProviderStatus.AVAILABLE

                    # Store all potential models for reference
                    if not hasattr(self.providers["ollama"], 'all_models'):
                        self.providers["ollama"].all_models = get_provider_models("ollama")

                    logger.info(f"✅ OLLAMA AVAILABLE (FREE) with {len(installed_models)} installed models: {', '.join(installed_models[:5])}")

                    # If Ollama is available, log cost savings potential
                    if any(p.api_key for p in self.providers.values()):
                        logger.info("💰 Using Ollama will save you money on API costs!")
            except Exception as e:
                self.providers["ollama"].status = ProviderStatus.ERROR
                self.providers["ollama"].last_error = str(e)
                logger.warning(f"⚠️ Ollama not available: {e}")
                logger.warning("💡 Install Ollama from https://ollama.ai for FREE local AI")

        # Test other providers
        for name, provider in self.providers.items():
            if name != "ollama" and provider.api_key:
                logger.info(f"Provider {name} configured (will cost ~${provider.cost_per_1k_tokens}/1k tokens)")

    def get_available_providers(self, respect_user_preference: bool = True) -> List[ProviderConfig]:
        """Get list of available providers with OLLAMA FIRST"""
        available = [
            p for p in self.providers.values()
            if p.enabled and p.status == ProviderStatus.AVAILABLE
        ]

        # If user has a preference and it's available, adjust order
        if respect_user_preference and self.user_preference:
            preferred = None
            others = []
            for p in available:
                if p.name == self.user_preference:
                    preferred = p
                else:
                    others.append(p)

            if preferred:
                # Always keep Ollama first if available, then preferred, then others
                ollama = next((p for p in others if p.name == "ollama"), None)
                if ollama:
                    others.remove(ollama)
                    if preferred.name != "ollama":
                        available = [ollama, preferred] + sorted(others, key=lambda x: x.priority)
                    else:
                        available = [ollama] + sorted(others, key=lambda x: x.priority)
                else:
                    available = [preferred] + sorted(others, key=lambda x: x.priority)
            else:
                available = sorted(available, key=lambda x: x.priority)
        else:
            # Standard priority sorting (Ollama will be first due to priority=1)
            available = sorted(available, key=lambda x: x.priority)

        return available

    def _select_optimal_ollama_model(self, prompt: str) -> str:
        """Select the most appropriate Ollama model based on prompt complexity"""
        if not self.ollama_manager.available_models:
            return "llama3.2:3b"  # Default

        prompt_length = len(prompt)
        prompt_lower = prompt.lower()

        # Map available models to their types
        available = set(self.ollama_manager.available_models)

        # Simple/fast queries - use smallest model
        if prompt_length < 50:
            for model in ["phi3:mini", "llama3.2:3b", "llama3.2"]:
                if any(m.startswith(model.split(':')[0]) for m in available):
                    return next(m for m in available if m.startswith(model.split(':')[0]))

        # Code-related queries - use code model
        if any(word in prompt_lower for word in ['code', 'function', 'class', 'def ', 'import', 'programming', 'script']):
            for model in ["codellama", "qwen2.5-coder"]:
                if any(model in m for m in available):
                    return next(m for m in available if model in m)

        # Complex/long queries - use larger model
        if prompt_length > 200:
            for model in ["qwen2.5-coder", "llama3.2"]:
                if any(model in m for m in available):
                    return next(m for m in available if model in m)

        # Default to first available
        return list(available)[0]

    def select_provider_for_model(self, model: Optional[str]) -> Optional[ProviderConfig]:
        """Select the best provider for a specific model with Ollama priority"""
        if not model:
            return None

        # Refresh Ollama models periodically
        if self.ollama_manager.should_refresh():
            self.ollama_manager.refresh_models()
            # Update Ollama provider with new models
            if "ollama" in self.providers:
                self.providers["ollama"].models = self.ollama_manager.available_models

        # Check if model is available in Ollama first (FREE)
        if self.ollama_manager.is_model_available(model):
            actual_model = self.ollama_manager.get_actual_model_name(model)
            if actual_model and "ollama" in self.providers:
                logger.info(f"✅ Model '{model}' found in Ollama as '{actual_model}'")
                # Store the actual model name for use
                self._actual_model_name = actual_model
                return self.providers["ollama"]

        # Check other providers based on model name patterns
        model_lower = model.lower()

        # Claude models -> Anthropic
        if "claude" in model_lower or "opus" in model_lower or "sonnet" in model_lower:
            if "anthropic" in self.providers and self.providers["anthropic"].enabled:
                return self.providers["anthropic"]

        # GPT models -> OpenAI
        if model_lower.startswith("gpt") or model_lower.startswith("o3") or "davinci" in model_lower:
            if "openai" in self.providers and self.providers["openai"].enabled:
                return self.providers["openai"]

        # Gemini models -> Google
        if "gemini" in model_lower or "bison" in model_lower:
            if "google" in self.providers and self.providers["google"].enabled:
                return self.providers["google"]

        # Grok models -> XAI
        if "grok" in model_lower:
            if "xai" in self.providers and self.providers["xai"].enabled:
                return self.providers["xai"]

        # Perplexity models
        if "pplx" in model_lower or "sonar" in model_lower:
            if "perplexity" in self.providers and self.providers["perplexity"].enabled:
                return self.providers["perplexity"]

        return None

    async def send_message(self, message: str, model: Optional[str] = None, **kwargs) -> Tuple[str, str]:
        """
        Send message to AI provider with Ollama-first routing
        Returns: (response, provider_used)
        NEVER returns placeholder - raises exception if no API available
        """
        # ALWAYS prioritize Ollama when no specific model is requested
        if not model or model == "auto":
            # Check if Ollama is available
            if "ollama" in self.providers and self.ollama_manager.is_available:
                # Select optimal Ollama model for this prompt
                selected_model = self._select_optimal_ollama_model(message)
                logger.info(f"Auto-selected Ollama model: {selected_model}")

                # Force Ollama as first provider
                ollama_provider = self.providers["ollama"]
                if ollama_provider.enabled:
                    # Ensure Ollama is marked as available
                    ollama_provider.status = ProviderStatus.AVAILABLE
                    providers = [ollama_provider]
                    # Add other providers as fallback only
                    other_providers = self.get_available_providers()
                    for p in other_providers:
                        if p.name != "ollama":
                            providers.append(p)

                    # Store selected model for use in _call_provider
                    self._auto_selected_model = selected_model
                else:
                    providers = self.get_available_providers()
            else:
                providers = self.get_available_providers()
        else:
            # Specific model requested
            specific_provider = self.select_provider_for_model(model)
            if specific_provider and specific_provider.status == ProviderStatus.AVAILABLE:
                # Try the specific provider first
                providers = [specific_provider]
                # Add other available providers as fallback
                other_providers = self.get_available_providers()
                for p in other_providers:
                    if p.name != specific_provider.name:
                        providers.append(p)
            else:
                providers = self.get_available_providers()

        if not providers:
            # NO PLACEHOLDER RESPONSES - Fail cleanly
            error_msg = self._get_configuration_error()
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        errors = []

        # Track if we're using Ollama to calculate savings
        using_paid_fallback = False

        for provider in providers:
            try:
                # Check rate limit
                if provider.rate_limit_reset and datetime.now() < provider.rate_limit_reset:
                    continue

                # Log which provider we're trying
                if provider.name == "ollama":
                    logger.info("🚀 Trying Ollama (FREE)")
                else:
                    if not using_paid_fallback and providers[0].name == "ollama":
                        using_paid_fallback = True
                        logger.info(f"💸 Falling back to paid provider: {provider.name}")

                response = await self._call_provider(provider, message, model, **kwargs)

                # Update usage stats
                self._update_usage_stats(provider, message, response)

                logger.info(f"✅ Successfully used {provider.name} for request")

                # If we used Ollama, log the savings
                if provider.name == "ollama" and self.usage_stats.cost_saved > 0:
                    logger.info(f"💰 Total saved so far: ${self.usage_stats.cost_saved:.4f}")

                return response, provider.name

            except Exception as e:
                error_msg = f"{provider.name}: {str(e)}"
                errors.append(error_msg)
                provider.last_error = str(e)

                # Check if rate limited (but never rate limit Ollama - it's local!)
                if provider.name != "ollama" and ("rate" in str(e).lower() or "429" in str(e)):
                    provider.status = ProviderStatus.RATE_LIMITED
                    provider.rate_limit_reset = datetime.now() + timedelta(minutes=5)

                if not provider.fallback_on_error:
                    break

                logger.warning(f"Provider {provider.name} failed: {e}")

        # All providers failed - NO PLACEHOLDER
        error_details = "\n".join(errors)
        error_msg = f"All AI providers failed. Errors:\n{error_details}\n\n{self._get_configuration_help()}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _get_configuration_error(self) -> str:
        """Get clear error message when no APIs configured"""
        return (
            "ERROR: No AI APIs configured or available.\n\n"
            "To use this MCP, you need at least one of:\n"
            "1. Ollama (FREE) - Install from https://ollama.ai\n"
            "2. API Keys - Set environment variables:\n"
            "   - ANTHROPIC_API_KEY\n"
            "   - OPENAI_API_KEY\n"
            "   - GEMINI_API_KEY\n"
            "   - XAI_API_KEY\n"
            "   - PERPLEXITY_API_KEY\n\n"
            "Ollama is recommended for cost-free operation."
        )

    def _get_configuration_help(self) -> str:
        """Get help for configuration issues"""
        configured = [name for name, p in self.providers.items() if p.api_key or name == "ollama"]

        if "ollama" not in configured:
            return (
                "💡 TIP: Install Ollama for FREE local AI:\n"
                "1. Download from https://ollama.ai\n"
                "2. Run: ollama pull llama2\n"
                "3. Restart this MCP\n"
            )
        else:
            return "Check your API keys and network connection."

    def _update_usage_stats(self, provider: ProviderConfig, message: str, response: str):
        """Update usage statistics and calculate savings"""
        # Estimate tokens (rough approximation)
        tokens = len(message.split()) + len(response.split())

        self.usage_stats.total_requests += 1
        self.usage_stats.total_tokens += tokens

        if provider.name == "ollama":
            self.usage_stats.ollama_requests += 1
            self.usage_stats.ollama_tokens += tokens

            # Calculate what we would have paid with cheapest paid provider
            cheapest_paid = min(
                (p.cost_per_1k_tokens for p in self.providers.values()
                 if p.cost_per_1k_tokens > 0 and p.status == ProviderStatus.AVAILABLE),
                default=0.001  # Default to $0.001 per 1k tokens
            )
            saved = (tokens / 1000) * cheapest_paid
            self.usage_stats.cost_saved += saved
        else:
            self.usage_stats.paid_requests += 1
            self.usage_stats.paid_tokens += tokens
            cost = (tokens / 1000) * provider.cost_per_1k_tokens
            self.usage_stats.total_cost += cost
            provider.total_cost += cost

        provider.usage_count += 1
        provider.tokens_used += tokens

    async def _call_provider(self, provider: ProviderConfig, message: str,
                            model: Optional[str] = None, **kwargs) -> str:
        """Call specific provider's API"""

        if provider.name == "ollama":
            # Check for auto-selected model first
            auto_selected = getattr(self, '_auto_selected_model', None)
            if auto_selected:
                model_to_use = auto_selected
                # Clear after use
                self._auto_selected_model = None
            else:
                # Use the actual model name if we resolved it earlier
                actual_model = getattr(self, '_actual_model_name', None)
                if actual_model and model:
                    # If we have the actual name from resolution, use it
                    model_to_use = actual_model
                    # Clear it after use
                    self._actual_model_name = None
                else:
                    # Otherwise, try to resolve it now or use optimal selection
                    if model and self.ollama_manager.is_model_available(model):
                        model_to_use = self.ollama_manager.get_actual_model_name(model) or model
                    else:
                        # Auto-select optimal model based on prompt
                        model_to_use = self._select_optimal_ollama_model(message)

            logger.info(f"Using Ollama model: {model_to_use}")
            return await self._call_ollama(provider, message, model_to_use, **kwargs)
        elif provider.name == "openai":
            return await self._call_openai(provider, message, model or "gpt-3.5-turbo", **kwargs)
        elif provider.name == "anthropic":
            return await self._call_anthropic(provider, message, model or "claude-3-haiku-20240307", **kwargs)
        elif provider.name == "google":
            return await self._call_google(provider, message, model or "gemini-pro", **kwargs)
        elif provider.name == "xai":
            return await self._call_xai(provider, message, model or "grok-1", **kwargs)
        elif provider.name == "perplexity":
            return await self._call_perplexity(provider, message, model or "pplx-7b-online", **kwargs)
        else:
            raise NotImplementedError(f"Provider {provider.name} not implemented")

    async def _call_ollama(self, provider: ProviderConfig, message: str, model: str, **kwargs) -> str:
        """Call Ollama API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{provider.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": message,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]

    async def _call_openai(self, provider: ProviderConfig, message: str, model: str, **kwargs) -> str:
        """Call OpenAI API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{provider.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {provider.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": message}],
                    **kwargs
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def _call_anthropic(self, provider: ProviderConfig, message: str, model: str, **kwargs) -> str:
        """Call Anthropic API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{provider.base_url}/messages",
                headers={
                    "x-api-key": provider.api_key,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": message}],
                    "max_tokens": kwargs.get("max_tokens", 1000)
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["content"][0]["text"]

    async def _call_google(self, provider: ProviderConfig, message: str, model: str, **kwargs) -> str:
        """Call Google Gemini API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{provider.base_url}/models/{model}:generateContent",
                params={"key": provider.api_key},
                json={
                    "contents": [{"parts": [{"text": message}]}]
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_xai(self, provider: ProviderConfig, message: str, model: str, **kwargs) -> str:
        """Call XAI API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{provider.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {provider.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": message}],
                    **kwargs
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def _call_perplexity(self, provider: ProviderConfig, message: str, model: str, **kwargs) -> str:
        """Call Perplexity API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{provider.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {provider.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": message}],
                    **kwargs
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    def get_status_report(self) -> Dict[str, Any]:
        """Get detailed status of all providers with cost information"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "providers": {},
            "usage_stats": {
                "total_requests": self.usage_stats.total_requests,
                "ollama_requests": self.usage_stats.ollama_requests,
                "paid_requests": self.usage_stats.paid_requests,
                "total_tokens": self.usage_stats.total_tokens,
                "total_cost": f"${self.usage_stats.total_cost:.4f}",
                "cost_saved": f"${self.usage_stats.cost_saved:.4f}",
                "ollama_usage_percentage": (
                    f"{(self.usage_stats.ollama_requests / max(self.usage_stats.total_requests, 1)) * 100:.1f}%"
                )
            },
            "recommendation": self._get_cost_recommendation()
        }

        for name, provider in self.providers.items():
            report["providers"][name] = {
                "status": provider.status.value,
                "enabled": provider.enabled,
                "priority": provider.priority,
                "is_free": provider.is_free,
                "cost_per_1k_tokens": f"${provider.cost_per_1k_tokens:.4f}" if not provider.is_free else "FREE",
                "models": provider.models,
                "usage_count": provider.usage_count,
                "tokens_used": provider.tokens_used,
                "total_cost": f"${provider.total_cost:.4f}",
                "last_error": provider.last_error,
                "configured": bool(provider.api_key or name == "ollama")
            }

        return report

    def _get_cost_recommendation(self) -> str:
        """Get recommendation based on usage patterns"""
        if self.usage_stats.total_requests == 0:
            return "No usage yet. Ollama is recommended for free operation."

        ollama_available = self.providers.get("ollama", {}).status == ProviderStatus.AVAILABLE

        if not ollama_available:
            return f"Install Ollama to save ${self.usage_stats.total_cost:.4f} on future requests!"

        if self.usage_stats.ollama_requests == self.usage_stats.total_requests:
            return f"Great! Using 100% free Ollama. Saved ${self.usage_stats.cost_saved:.4f} so far!"

        percentage = (self.usage_stats.ollama_requests / self.usage_stats.total_requests) * 100
        return f"Using Ollama {percentage:.1f}% of the time. Increase usage to save more!"

    def set_user_preference(self, provider_name: str):
        """Set user's preferred provider (after Ollama)"""
        if provider_name in self.providers:
            self.user_preference = provider_name

            # Save preference
            config = {"user_preference": provider_name}
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Set user preference to {provider_name}")
        else:
            logger.warning(f"Provider {provider_name} not found")

    def reset_provider(self, provider_name: str):
        """Reset a provider's error state"""
        if provider_name in self.providers:
            self.providers[provider_name].status = ProviderStatus.AVAILABLE
            self.providers[provider_name].rate_limit_reset = None
            self.providers[provider_name].last_error = None
            logger.info(f"Reset provider {provider_name}")

    def log_initialization_status(self):
        """Log clear initialization status"""
        logger.info("=" * 60)
        logger.info("API MANAGER INITIALIZATION COMPLETE")
        logger.info("=" * 60)

        available = self.get_available_providers()
        if available:
            logger.info(f"Priority order:")
            for i, p in enumerate(available, 1):
                cost_info = "FREE" if p.is_free else f"${p.cost_per_1k_tokens}/1k tokens"
                logger.info(f"  {i}. {p.name} ({cost_info})")
        else:
            logger.warning("No providers available!")

        logger.info("=" * 60)


# Make OptimizedAPIManager available as APIManager for compatibility
APIManager = OptimizedAPIManager


if __name__ == "__main__":
    # Test the optimized API manager
    import asyncio

    async def test():
        manager = OptimizedAPIManager()
        print("\n=== Optimized API Manager Status ===")
        print(json.dumps(manager.get_status_report(), indent=2))

        try:
            print("\n=== Testing Message ===")
            response, provider = await manager.send_message("Hello, how are you?")
            print(f"Provider used: {provider}")
            print(f"Response: {response[:200]}...")
        except RuntimeError as e:
            print(f"ERROR: {e}")

    asyncio.run(test())