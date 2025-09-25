#!/usr/bin/env python3
"""
Test Actual Responses from Converse-Enhanced MCP
Captures real responses from each model
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_all_models():
    """Test actual responses from all models"""
    from server import OptimizedAPIManager

    print("=" * 80)
    print("ACTUAL MODEL RESPONSE TEST")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    manager = OptimizedAPIManager()

    # Test data
    test_cases = [
        {
            "name": "Test 1: Math (No Model Specified)",
            "prompt": "What is 15 divided by 3? Reply with just the number.",
            "model": None,
            "expected": "Should use default Ollama model"
        },
        {
            "name": "Test 2: Code with CodeLlama",
            "prompt": "Write a Python function to calculate factorial. Just the function, no explanation.",
            "model": "codellama",
            "expected": "Should use codellama:7b"
        },
        {
            "name": "Test 3: Quick Response with Phi3",
            "prompt": "Complete this: The capital of France is",
            "model": "phi3",
            "expected": "Should use phi3:mini"
        },
        {
            "name": "Test 4: Complex with Qwen Coder",
            "prompt": "What is a REST API? One sentence only.",
            "model": "qwen2.5-coder",
            "expected": "Should use qwen2.5-coder:32b"
        },
        {
            "name": "Test 5: Llama 3.2 Test",
            "prompt": "Say 'Hello World' and nothing else.",
            "model": "llama3.2",
            "expected": "Should use llama3.2:3b"
        }
    ]

    results = []

    for test in test_cases:
        print(f"\n{test['name']}")
        print("-" * 40)
        print(f"Prompt: {test['prompt']}")
        print(f"Model: {test['model'] or 'Auto-select'}")
        print(f"Expected: {test['expected']}")

        start_time = time.time()

        try:
            response, provider = await manager.send_message(
                test['prompt'],
                model=test['model']
            )
            elapsed = time.time() - start_time

            # Clean response for display
            response_preview = response.replace('\n', ' ').strip()
            if len(response_preview) > 200:
                response_preview = response_preview[:200] + "..."

            print(f"Provider Used: {provider}")
            print(f"Response Time: {elapsed:.2f}s")
            print(f"Response: {response_preview}")

            # Check if correct provider was used
            if provider == "ollama":
                print("Status: [OK] Used Ollama (FREE)")
            else:
                print(f"Status: [WARNING] Used {provider} instead of Ollama")

            results.append({
                "test": test['name'],
                "success": True,
                "provider": provider,
                "time": elapsed,
                "response": response_preview
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "test": test['name'],
                "success": False,
                "error": str(e)
            })

    # Print summary
    print("\n" + "=" * 80)
    print("RESPONSE TEST SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results if r.get('success'))
    print(f"\nTests Passed: {successful}/{len(results)}")

    # Usage statistics
    stats = manager.get_status_report()
    print(f"\nUsage Statistics:")
    print(f"  Total Requests: {stats['usage_stats']['total_requests']}")
    print(f"  Ollama Requests: {stats['usage_stats']['ollama_requests']}")
    print(f"  Cost Saved: ${stats['usage_stats']['cost_saved']}")

    # Model availability
    print(f"\nOllama Models Available:")
    if manager.ollama_manager.is_available:
        for model in manager.ollama_manager.available_models:
            caps = manager.ollama_manager.get_model_capabilities(model)
            features = []
            if caps['code']:
                features.append("CODE")
            if caps['vision']:
                features.append("VISION")
            if caps['large_context']:
                features.append("LARGE_CONTEXT")

            features_str = f" [{', '.join(features)}]" if features else ""
            print(f"  - {model}{features_str}")

    return successful == len(results)


async def test_edge_cases():
    """Test edge cases and error handling"""
    from server import OptimizedAPIManager

    print("\n" + "=" * 80)
    print("EDGE CASE TESTING")
    print("=" * 80)

    manager = OptimizedAPIManager()

    edge_cases = [
        {
            "name": "Empty Prompt",
            "prompt": "",
            "expected": "Should handle gracefully"
        },
        {
            "name": "Special Characters",
            "prompt": "Explain: && || $variable ${code}",
            "expected": "Should handle without encoding errors"
        },
        {
            "name": "Very Long Prompt",
            "prompt": "Explain quantum computing. " * 100,  # ~2000 chars
            "expected": "Should handle long input"
        },
        {
            "name": "Non-existent Model",
            "prompt": "Hello",
            "model": "non-existent-model-xyz",
            "expected": "Should fallback or error gracefully"
        }
    ]

    for test in edge_cases:
        print(f"\n{test['name']}")
        print("-" * 40)

        try:
            model = test.get('model')
            response, provider = await manager.send_message(
                test['prompt'],
                model=model
            )
            print(f"Result: Handled successfully via {provider}")
            print(f"Response preview: {response[:50]}...")

        except Exception as e:
            error_msg = str(e)
            if "required" in error_msg.lower() or "empty" in error_msg.lower():
                print(f"Result: Correctly rejected empty/invalid input")
            else:
                print(f"Result: Error handled - {error_msg[:100]}")


async def test_performance():
    """Test performance metrics"""
    from server import OptimizedAPIManager

    print("\n" + "=" * 80)
    print("PERFORMANCE TESTING")
    print("=" * 80)

    manager = OptimizedAPIManager()

    # Test rapid requests
    print("\nTesting rapid sequential requests...")
    times = []

    for i in range(5):
        start = time.time()
        try:
            response, provider = await manager.send_message(
                f"Count to {i+1}",
                model="phi3"  # Use fastest model
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Request {i+1}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  Request {i+1}: Failed - {e}")

    if times:
        avg_time = sum(times) / len(times)
        print(f"\nAverage Response Time: {avg_time:.2f}s")
        print(f"Total Time for {len(times)} requests: {sum(times):.2f}s")


async def main():
    """Run all tests"""
    # Test actual responses
    responses_ok = await test_all_models()

    # Test edge cases
    await test_edge_cases()

    # Test performance
    await test_performance()

    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST COMPLETE")
    print("=" * 80)

    if responses_ok:
        print("ALL CORE TESTS PASSED")
    else:
        print("SOME TESTS FAILED - Review output")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == "__main__":
    asyncio.run(main())