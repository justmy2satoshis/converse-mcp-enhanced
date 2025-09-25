"""
Ollama Manager - Auto-detection and management of Ollama models
"""
import requests
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class OllamaManager:
    """Manages Ollama model auto-detection and availability"""

    def __init__(self, host: str = 'http://localhost:11434'):
        self.host = host
        self.available_models: List[str] = []
        self.model_details: Dict[str, Any] = {}
        self.last_refresh: Optional[datetime] = None
        self.refresh_interval = timedelta(minutes=5)  # Auto-refresh every 5 minutes
        self.is_available = False

        # Initial detection
        self.refresh_models()

    def refresh_models(self) -> List[str]:
        """Auto-detect available Ollama models"""
        try:
            response = requests.get(f'{self.host}/api/tags', timeout=5)
            if response.status_code == 200:
                data = response.json()
                models_data = data.get('models', [])

                # Extract model names and store details
                self.available_models = []
                self.model_details = {}

                for model in models_data:
                    model_name = model.get('name', '')
                    if model_name:
                        self.available_models.append(model_name)
                        self.model_details[model_name] = {
                            'size': model.get('size', 0),
                            'modified': model.get('modified_at', ''),
                            'family': model.get('details', {}).get('family', ''),
                            'parameter_size': model.get('details', {}).get('parameter_size', ''),
                            'quantization': model.get('details', {}).get('quantization_level', '')
                        }

                self.is_available = True
                self.last_refresh = datetime.now()

                logger.info(f'✅ Ollama auto-detection: Found {len(self.available_models)} models')
                logger.info(f'   Available models: {", ".join(self.available_models)}')

                return self.available_models
            else:
                logger.warning(f'⚠️ Ollama API returned status {response.status_code}')
                self.is_available = False
                return []

        except requests.exceptions.ConnectionError:
            logger.warning('⚠️ Ollama not running (connection refused)')
            self.is_available = False
            return []
        except requests.exceptions.Timeout:
            logger.warning('⚠️ Ollama API timeout')
            self.is_available = False
            return []
        except Exception as e:
            logger.error(f'❌ Failed to detect Ollama models: {e}')
            self.is_available = False
            return []

    def should_refresh(self) -> bool:
        """Check if models list should be refreshed"""
        if not self.last_refresh:
            return True
        return datetime.now() - self.last_refresh > self.refresh_interval

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available in Ollama"""
        # Refresh if needed
        if self.should_refresh():
            self.refresh_models()

        if not self.available_models:
            return False

        # Clean up the requested name
        clean_name = model_name.lower().strip()

        # Remove common prefixes
        for prefix in ['ollama/', 'ollama:', 'local/', 'local:']:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]

        # Direct match
        if clean_name in self.available_models:
            return True

        # Check with :latest tag
        if f'{clean_name}:latest' in self.available_models:
            return True

        # Check if it's a base name that matches
        base_name = clean_name.split(':')[0]
        for model in self.available_models:
            if model.startswith(f'{base_name}:') or model == base_name:
                return True

        # Fuzzy matching for common variations
        variations = [
            clean_name.replace('.', ''),
            clean_name.replace('-', ''),
            clean_name.replace('_', ''),
            clean_name.replace('.', '-'),
            clean_name.replace('-', '.'),
        ]

        for variant in variations:
            if variant in self.available_models:
                return True
            for model in self.available_models:
                if variant in model or model in variant:
                    return True

        return False

    def get_actual_model_name(self, requested_name: str) -> Optional[str]:
        """Get the actual Ollama model name from requested name"""
        # Refresh if needed
        if self.should_refresh():
            self.refresh_models()

        if not self.available_models:
            return None

        # Clean up the requested name
        clean_name = requested_name.lower().strip()

        # Remove common prefixes
        for prefix in ['ollama/', 'ollama:', 'local/', 'local:']:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]

        # Direct match
        if clean_name in self.available_models:
            return clean_name

        # With :latest tag
        if f'{clean_name}:latest' in self.available_models:
            return f'{clean_name}:latest'

        # Base name matching
        base_name = clean_name.split(':')[0]
        for model in self.available_models:
            if model.startswith(f'{base_name}:'):
                return model
            if model == base_name:
                return model

        # Fuzzy matching
        variations = [
            clean_name.replace('.', ''),
            clean_name.replace('-', ''),
            clean_name.replace('_', ''),
        ]

        for variant in variations:
            if variant in self.available_models:
                return variant
            for model in self.available_models:
                if variant in model:
                    return model

        # Partial match as last resort
        for model in self.available_models:
            if clean_name in model or model in clean_name:
                return model

        return None

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model"""
        actual_name = self.get_actual_model_name(model_name)
        if actual_name and actual_name in self.model_details:
            return self.model_details[actual_name]
        return None

    def get_models_by_size(self, max_size_gb: float = None) -> List[str]:
        """Get models filtered by size"""
        if max_size_gb is None:
            return self.available_models

        max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        filtered = []

        for model_name, details in self.model_details.items():
            if details.get('size', 0) <= max_size_bytes:
                filtered.append(model_name)

        return filtered

    def get_model_capabilities(self, model_name: str) -> Dict[str, bool]:
        """Detect model capabilities based on name and family"""
        capabilities = {
            'code': False,
            'vision': False,
            'chat': True,  # Most models support chat
            'embedding': False,
            'large_context': False,
            'multilingual': False,
        }

        if not model_name:
            return capabilities

        name_lower = model_name.lower()

        # Code capabilities
        if any(x in name_lower for x in ['code', 'coder', 'starcoder', 'deepseek-coder']):
            capabilities['code'] = True

        # Vision capabilities
        if any(x in name_lower for x in ['vision', 'llava', 'bakllava', 'moondream']):
            capabilities['vision'] = True

        # Large context
        if any(x in name_lower for x in ['32k', '64k', '128k', '100k']):
            capabilities['large_context'] = True

        # Multilingual
        if any(x in name_lower for x in ['qwen', 'yi', 'solar', 'gemma']):
            capabilities['multilingual'] = True

        # Embedding models
        if any(x in name_lower for x in ['embed', 'bge', 'nomic']):
            capabilities['embedding'] = True
            capabilities['chat'] = False  # Embedding models don't chat

        return capabilities

    def get_best_model_for_task(self, task_type: str) -> Optional[str]:
        """Get the best available model for a specific task"""
        if not self.available_models:
            return None

        task_type_lower = task_type.lower()

        # Map task types to preferred models
        if 'code' in task_type_lower or 'programming' in task_type_lower:
            # Prefer code models
            for model in self.available_models:
                if 'code' in model.lower() or 'coder' in model.lower():
                    return model
            # Fallback to general models
            for model in self.available_models:
                if 'llama' in model.lower() or 'qwen' in model.lower():
                    return model

        elif 'vision' in task_type_lower or 'image' in task_type_lower:
            # Prefer vision models
            for model in self.available_models:
                if 'vision' in model.lower() or 'llava' in model.lower():
                    return model
            return None  # No fallback for vision tasks

        elif 'fast' in task_type_lower or 'quick' in task_type_lower:
            # Prefer smaller, faster models
            smallest_model = None
            smallest_size = float('inf')
            for model, details in self.model_details.items():
                size = details.get('size', float('inf'))
                if size < smallest_size:
                    smallest_size = size
                    smallest_model = model
            return smallest_model

        # Default to first available model
        return self.available_models[0] if self.available_models else None

    def test_model(self, model_name: str, test_prompt: str = "Hello") -> bool:
        """Test if a specific model actually works"""
        try:
            actual_name = self.get_actual_model_name(model_name)
            if not actual_name:
                return False

            response = requests.post(
                f'{self.host}/api/generate',
                json={
                    'model': actual_name,
                    'prompt': test_prompt,
                    'stream': False,
                    'options': {
                        'num_predict': 10  # Keep it short for testing
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'response' in result:
                    logger.info(f'✅ Model {actual_name} tested successfully')
                    return True

            logger.warning(f'⚠️ Model {actual_name} test failed: {response.status_code}')
            return False

        except Exception as e:
            logger.error(f'❌ Model {model_name} test error: {e}')
            return False

    def pull_model(self, model_name: str) -> bool:
        """Pull a new model from Ollama library"""
        try:
            response = requests.post(
                f'{self.host}/api/pull',
                json={'name': model_name, 'stream': False},
                timeout=600  # Allow 10 minutes for download
            )

            if response.status_code == 200:
                logger.info(f'✅ Successfully pulled model: {model_name}')
                # Refresh the models list
                self.refresh_models()
                return True
            else:
                logger.error(f'❌ Failed to pull model {model_name}: {response.status_code}')
                return False

        except Exception as e:
            logger.error(f'❌ Error pulling model {model_name}: {e}')
            return False


# Test the manager if run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing OllamaManager...")
    manager = OllamaManager()

    print(f"\nOllama Available: {manager.is_available}")
    print(f"Models Found: {len(manager.available_models)}")

    for model in manager.available_models:
        info = manager.get_model_info(model)
        size_gb = info['size'] / (1024**3) if info else 0
        print(f"  - {model} ({size_gb:.1f} GB)")

    # Test model name resolution
    test_names = ["llama3.2", "codellama", "phi3", "qwen2.5-coder", "ollama/llama3.2"]
    print("\nModel Name Resolution Tests:")
    for name in test_names:
        actual = manager.get_actual_model_name(name)
        available = manager.is_model_available(name)
        print(f"  '{name}' -> '{actual}' (Available: {available})")