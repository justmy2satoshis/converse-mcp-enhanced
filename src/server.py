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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.providers["ollama"] = ProviderConfig(
            name="ollama",
            priority=1,  # HIGHEST PRIORITY
            enabled=True,
            api_key=None,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            models=[],  # Will be populated dynamically
            cost_per_1k_tokens=0.0,
            is_free=True,
            status=ProviderStatus.NOT_CONFIGURED
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
                models=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
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
                models=["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview"],
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
                models=["gemini-pro", "gemini-pro-vision"],
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
                models=["grok-1", "grok-2"],
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
                models=["pplx-7b-online", "pplx-70b-online"],
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
                    self.providers["ollama"].models = [m["name"] for m in models]
                    self.providers["ollama"].status = ProviderStatus.AVAILABLE
                    logger.info(f"✅ OLLAMA AVAILABLE (FREE) with {len(models)} models: {', '.join(self.providers['ollama'].models)}")

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

    async def send_message(self, message: str, model: Optional[str] = None, **kwargs) -> Tuple[str, str]:
        """
        Send message to AI provider with Ollama-first routing
        Returns: (response, provider_used)
        NEVER returns placeholder - raises exception if no API available
        """
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

                # Check if rate limited
                if "rate" in str(e).lower() or "429" in str(e):
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
            return await self._call_ollama(provider, message, model or provider.models[0] if provider.models else "llama2", **kwargs)
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