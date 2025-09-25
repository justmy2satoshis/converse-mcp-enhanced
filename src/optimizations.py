#!/usr/bin/env python3
"""
Performance optimizations for converse-mcp-enhanced
Includes caching, parallel processing, and smart model selection
"""

import asyncio
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimizations for model selection and response handling"""

    def __init__(self, api_manager):
        self.api_manager = api_manager
        self._response_cache = {}
        self._model_performance = {}

    @lru_cache(maxsize=100)
    def select_model_for_task(self, task_type: str, cost_tier: str = 'medium') -> str:
        """
        Smart model selection based on task type and cost constraints
        """
        model_recommendations = {
            'code': {
                'free': 'codellama:7b',
                'low': 'gpt-3.5-turbo',
                'medium': 'grok-code-fast-1',
                'high': 'gpt-5'
            },
            'reasoning': {
                'free': 'llama3.3:70b',
                'low': 'o1-mini',
                'medium': 'o3-mini',
                'high': 'o3-pro'
            },
            'creative': {
                'free': 'mistral:7b',
                'low': 'claude-3.5-haiku-20241022',
                'medium': 'claude-sonnet-4',
                'high': 'claude-opus-4-1-20250805'
            },
            'general': {
                'free': 'llama3.1:8b',
                'low': 'gpt-3.5-turbo',
                'medium': 'gemini-2.5-flash',
                'high': 'gpt-5'
            },
            'vision': {
                'free': 'llava:7b',
                'low': 'gemini-2.0-flash',
                'medium': 'gpt-4o',
                'high': 'grok-2-vision-1212'
            }
        }

        task_type = task_type.lower()
        if task_type not in model_recommendations:
            task_type = 'general'

        return model_recommendations[task_type].get(cost_tier, 'llama3.1:8b')

    async def query_models_parallel(self, prompt: str, models: List[str], timeout: int = 30) -> Dict[str, str]:
        """
        Query multiple models in parallel and return all responses
        """
        tasks = []
        for model in models:
            task = asyncio.create_task(self._query_single_model(model, prompt, timeout))
            tasks.append((model, task))

        results = {}
        for model, task in tasks:
            try:
                response = await task
                results[model] = response
            except Exception as e:
                results[model] = f"Error: {str(e)}"
                logger.warning(f"Failed to query {model}: {e}")

        return results

    async def _query_single_model(self, model: str, prompt: str, timeout: int) -> str:
        """Query a single model with timeout"""
        try:
            # This would integrate with the actual API manager
            # For now, return a placeholder
            await asyncio.sleep(0.1)  # Simulate API call
            return f"Response from {model}"
        except asyncio.TimeoutError:
            raise TimeoutError(f"Model {model} timed out after {timeout}s")

    def retry_with_backoff(self, func, max_retries: int = 3, initial_delay: float = 1.0):
        """
        Retry a function with exponential backoff
        """
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = initial_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)

    def cache_response(self, prompt_hash: str, response: str, ttl: int = 300):
        """
        Cache a response with TTL (time-to-live)
        """
        self._response_cache[prompt_hash] = {
            'response': response,
            'timestamp': time.time(),
            'ttl': ttl
        }

    def get_cached_response(self, prompt_hash: str) -> Optional[str]:
        """
        Get a cached response if still valid
        """
        if prompt_hash in self._response_cache:
            cache_entry = self._response_cache[prompt_hash]
            if time.time() - cache_entry['timestamp'] < cache_entry['ttl']:
                logger.info(f"Cache hit for prompt hash: {prompt_hash}")
                return cache_entry['response']
            else:
                # Remove expired entry
                del self._response_cache[prompt_hash]
        return None

    def track_model_performance(self, model: str, response_time: float, success: bool):
        """
        Track performance metrics for each model
        """
        if model not in self._model_performance:
            self._model_performance[model] = {
                'total_requests': 0,
                'successful_requests': 0,
                'total_response_time': 0,
                'average_response_time': 0,
                'success_rate': 0
            }

        stats = self._model_performance[model]
        stats['total_requests'] += 1
        if success:
            stats['successful_requests'] += 1
            stats['total_response_time'] += response_time

        stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
        if stats['successful_requests'] > 0:
            stats['average_response_time'] = stats['total_response_time'] / stats['successful_requests']

    def get_best_performing_model(self, min_requests: int = 5) -> Optional[str]:
        """
        Get the best performing model based on success rate and response time
        """
        eligible_models = {
            model: stats for model, stats in self._model_performance.items()
            if stats['total_requests'] >= min_requests
        }

        if not eligible_models:
            return None

        # Score = success_rate * 0.7 + (1 / (1 + avg_response_time)) * 0.3
        best_model = None
        best_score = -1

        for model, stats in eligible_models.items():
            score = (stats['success_rate'] * 0.7 +
                    (1 / (1 + stats['average_response_time'])) * 0.3)
            if score > best_score:
                best_score = score
                best_model = model

        return best_model


class ConnectionPool:
    """
    Manage connection pooling for API providers
    """

    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections = {}
        self.available = asyncio.Semaphore(max_connections)

    async def get_connection(self, provider: str):
        """Get or create a connection for a provider"""
        async with self.available:
            if provider not in self.connections:
                self.connections[provider] = await self._create_connection(provider)
            return self.connections[provider]

    async def _create_connection(self, provider: str):
        """Create a new connection for a provider"""
        # Implementation would create actual HTTP client
        return f"Connection to {provider}"

    async def close_all(self):
        """Close all connections"""
        for connection in self.connections.values():
            # Close actual connections
            pass
        self.connections.clear()


def optimize_prompt(prompt: str, max_tokens: int = 4000) -> str:
    """
    Optimize prompt to fit within token limits while preserving meaning
    """
    # Simple truncation for now - could implement smarter compression
    if len(prompt) > max_tokens:
        return prompt[:max_tokens-100] + "\n... [truncated for length]"
    return prompt


def batch_requests(requests: List[Dict], batch_size: int = 5) -> List[List[Dict]]:
    """
    Batch multiple requests for efficient processing
    """
    batches = []
    for i in range(0, len(requests), batch_size):
        batches.append(requests[i:i + batch_size])
    return batches