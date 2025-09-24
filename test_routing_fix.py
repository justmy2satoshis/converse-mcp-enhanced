#!/usr/bin/env python3
"""
Test Routing Fixes - Ensure 100% Ollama Usage
Verifies that all fixes are working correctly
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_routing():
    """Test that routing always uses Ollama when available"""
    from server import OptimizedAPIManager

    print("=" * 80)
    print("ROUTING FIX VALIDATION TEST")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    manager = OptimizedAPIManager()

    # Verify Ollama is available and properly configured
    print("\n1. OLLAMA STATUS CHECK")
    print("-" * 40)
    print(f"Ollama Available: {manager.ollama_manager.is_available}")
    print(f"Ollama Models: {manager.ollama_manager.available_models}")
    print(f"Ollama Status: {manager.providers['ollama'].status.value}")
    print(f"Ollama Priority: {manager.providers['ollama'].priority}")

    # Test cases to verify routing
    test_cases = [
        {
            "name": "Test 1: Simple query (no model)",
            "prompt": "What is 2+2?",
            "model": None,
            "expected_provider": "ollama",
            "expected_model_type": "phi3:mini"  # Should use smallest
        },
        {
            "name": "Test 2: Auto model selection",
            "prompt": "Hello",
            "model": "auto",
            "expected_provider": "ollama",
            "expected_model_type": "phi3:mini"
        },
        {
            "name": "Test 3: Code prompt (no model)",
            "prompt": "Write a Python function to reverse a string",
            "model": None,
            "expected_provider": "ollama",
            "expected_model_type": "codellama"
        },
        {
            "name": "Test 4: Complex query (no model)",
            "prompt": "Explain quantum computing in detail. " * 10,  # Long prompt
            "model": None,
            "expected_provider": "ollama",
            "expected_model_type": "qwen2.5-coder"
        },
        {
            "name": "Test 5: Explicit Ollama model",
            "prompt": "Test",
            "model": "llama3.2",
            "expected_provider": "ollama",
            "expected_model_type": "llama3.2:3b"
        }
    ]

    results = []
    ollama_count = 0
    api_count = 0

    print("\n2. ROUTING TESTS")
    print("-" * 40)

    for test in test_cases:
        print(f"\n{test['name']}")
        print(f"  Prompt: {test['prompt'][:50]}...")
        print(f"  Model param: {test['model']}")

        try:
            start_time = time.time()
            response, provider = await manager.send_message(
                test['prompt'],
                model=test['model']
            )
            elapsed = time.time() - start_time

            print(f"  Provider used: {provider}")
            print(f"  Response time: {elapsed:.2f}s")

            # Track provider usage
            if provider == "ollama":
                ollama_count += 1
                print(f"  ✅ CORRECT: Used Ollama")
            else:
                api_count += 1
                print(f"  ❌ ERROR: Used {provider} instead of Ollama!")

            results.append({
                "test": test['name'],
                "provider": provider,
                "correct": provider == "ollama",
                "time": elapsed
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "test": test['name'],
                "error": str(e),
                "correct": False
            })

    # Test sequential requests (ensure no rate limiting)
    print("\n3. SEQUENTIAL REQUEST TEST (No Rate Limiting)")
    print("-" * 40)

    sequential_success = 0
    for i in range(10):
        try:
            response, provider = await manager.send_message(f"Count to {i}")
            if provider == "ollama":
                sequential_success += 1
            print(f"  Request {i+1}: {provider} ✅" if provider == "ollama" else f"  Request {i+1}: {provider} ❌")

        except Exception as e:
            print(f"  Request {i+1}: Failed - {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    print(f"\nRouting Tests:")
    print(f"  Ollama used: {ollama_count}/{len(test_cases)}")
    print(f"  API used: {api_count}/{len(test_cases)}")
    print(f"  Success rate: {(ollama_count/len(test_cases)*100):.1f}%")

    print(f"\nSequential Tests:")
    print(f"  Ollama used: {sequential_success}/10")
    print(f"  No rate limiting: {'✅ PASS' if sequential_success == 10 else '❌ FAIL'}")

    # Check final statistics
    stats = manager.get_status_report()
    print(f"\nFinal Statistics:")
    print(f"  Total requests: {stats['usage_stats']['total_requests']}")
    print(f"  Ollama requests: {stats['usage_stats']['ollama_requests']}")
    print(f"  Paid requests: {stats['usage_stats']['paid_requests']}")
    print(f"  Cost saved: ${stats['usage_stats']['cost_saved']}")

    ollama_percentage = (stats['usage_stats']['ollama_requests'] / stats['usage_stats']['total_requests'] * 100) if stats['usage_stats']['total_requests'] > 0 else 0

    print(f"  Ollama usage: {ollama_percentage:.1f}%")

    # Final verdict
    print("\n" + "=" * 80)
    if ollama_percentage == 100:
        print("✅ SUCCESS: 100% OLLAMA USAGE ACHIEVED!")
        print("All routing fixes are working correctly.")
    else:
        print(f"❌ FAILURE: Only {ollama_percentage:.1f}% Ollama usage")
        print("Routing fixes need more work.")

    print(f"\nTest completed: {datetime.now()}")
    print("=" * 80)

    return ollama_percentage == 100


if __name__ == "__main__":
    success = asyncio.run(test_routing())
    sys.exit(0 if success else 1)