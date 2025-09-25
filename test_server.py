#!/usr/bin/env python3
"""
Comprehensive test suite for converse-mcp-enhanced server
"""
import sys
import os
import json

# Add src to path
sys.path.insert(0, 'src')

print("=" * 60)
print("CONVERSE-MCP-ENHANCED TEST SUITE")
print("=" * 60)

# Test 1: Import test
print("\n[TEST 1] Module Import Test")
try:
    import server
    print("[OK] Server module imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import server module: {e}")
    sys.exit(1)

# Test 2: Check for required classes/functions
print("\n[TEST 2] Core Components Test")
required_components = [
    'ProviderStatus',
    'ProviderConfig',
    'UsageStats',
    'APIManager'
]

for component in required_components:
    if hasattr(server, component):
        print(f"[OK] {component} found")
    else:
        print(f"[FAIL] {component} NOT FOUND")

# Test 3: Check model definitions
print("\n[TEST 3] Model Definitions Test")
try:
    manager = server.APIManager()
    providers = manager.providers

    for provider_name, provider_config in providers.items():
        model_count = len(provider_config.models)
        print(f"[OK] {provider_name}: {model_count} models configured")

        # Show first 3 models as sample
        if model_count > 0:
            sample_models = provider_config.models[:3]
            for model in sample_models:
                print(f"  - {model}")
            if model_count > 3:
                print(f"  ... and {model_count - 3} more")

except Exception as e:
    print(f"[FAIL] Error checking models: {e}")

# Test 4: Environment variable handling
print("\n[TEST 4] Environment Variable Test")
test_env_vars = [
    'OPENAI_API_KEY',
    'ANTHROPIC_API_KEY',
    'GEMINI_API_KEY',
    'XAI_API_KEY',
    'PERPLEXITY_API_KEY',
    'OLLAMA_BASE_URL'
]

for var in test_env_vars:
    value = os.getenv(var)
    if value:
        print(f"[OK] {var}: {'*' * 8} (set)")
    else:
        print(f"[INFO] {var}: not set (will use defaults)")

# Test 5: Check for PROVIDER_MODELS dictionary
print("\n[TEST 5] PROVIDER_MODELS Dictionary Test")
if hasattr(server, 'PROVIDER_MODELS'):
    print(f"[OK] PROVIDER_MODELS found with {len(server.PROVIDER_MODELS)} providers")
    for provider, config in server.PROVIDER_MODELS.items():
        if 'models' in config:
            print(f"  - {provider}: {len(config['models'])} models")
else:
    print("[FAIL] PROVIDER_MODELS dictionary NOT FOUND")
    print("  This needs to be added for exhaustive model support")

# Test 6: Check for helper functions
print("\n[TEST 6] Helper Functions Test")
helper_functions = [
    'get_provider_models',
    'get_recommended_models',
    'MODEL_ALIASES',
    'COST_TIERS'
]

for func_name in helper_functions:
    if hasattr(server, func_name):
        print(f"[OK] {func_name} found")
    else:
        print(f"[INFO] {func_name} not found (optional)")

# Test 7: Basic instantiation test
print("\n[TEST 7] Instantiation Test")
try:
    manager = server.APIManager()
    print("[OK] APIManager instantiated successfully")

    # Check Ollama priority
    if 'ollama' in manager.providers:
        ollama_priority = manager.providers['ollama'].priority
        print(f"[OK] Ollama priority: {ollama_priority} (should be 1)")
        if ollama_priority != 1:
            print("  WARNING: Ollama should have priority 1 for free-first approach")
    else:
        print("[FAIL] Ollama provider not found!")

except Exception as e:
    print(f"[FAIL] Error instantiating APIManager: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Check total model count
print("\n[TEST 8] Total Model Count")
try:
    manager = server.APIManager()
    total_models = 0
    for provider_name, provider_config in manager.providers.items():
        total_models += len(provider_config.models)

    print(f"Total models across all providers: {total_models}")
    if total_models >= 100:
        print("[OK] Exhaustive model support confirmed (100+ models)")
    else:
        print(f"[FAIL] Only {total_models} models found (expected 100+)")

except Exception as e:
    print(f"[FAIL] Error counting models: {e}")

print("\n" + "=" * 60)
print("TEST SUITE COMPLETE")
print("=" * 60)