#!/usr/bin/env python3
"""Test API rate limiting with current keys"""
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# Collect all ZZZ API keys
api_keys = []
for i in range(2, 50):
    key = os.getenv(f"ZZZ_API_KEY_{i}")
    if key:
        api_keys.append(key)

print(f"Found {len(api_keys)} API keys")

def test_single_call(key_idx):
    """Test a single API call"""
    api_key = api_keys[key_idx]
    try:
        start = time.time()
        response = requests.post(
            "https://api.zhizengzeng.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Say 'test' in one word"}],
                "max_tokens": 10
            },
            timeout=30
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            return {"status": "success", "key_idx": key_idx, "time": elapsed}
        elif response.status_code == 429:
            return {"status": "rate_limited", "key_idx": key_idx, "time": elapsed}
        else:
            return {"status": "error", "key_idx": key_idx, "code": response.status_code, "time": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        return {"status": "error", "key_idx": key_idx, "error": str(e), "time": elapsed}

def main():
    """Run concurrent API tests"""
    # Test with 24 concurrent requests (simulating 24 workers)
    num_requests = 24

    results = []
    for batch in range(3):  # 3 batches
        print(f"\n=== Batch {batch + 1} ===")

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = []
            for i in range(num_requests):
                key_idx = i % len(api_keys)
                futures.append(executor.submit(test_single_call, key_idx))

            batch_results = [f.result() for f in as_completed(futures)]

        results.extend(batch_results)

        # Count results
        success = sum(1 for r in batch_results if r["status"] == "success")
        rate_limited = sum(1 for r in batch_results if r["status"] == "rate_limited")
        errors = sum(1 for r in batch_results if r["status"] == "error")

        print(f"Success: {success}/{num_requests}")
        print(f"Rate limited: {rate_limited}/{num_requests}")
        print(f"Errors: {errors}/{num_requests}")

        if batch < 2:  # Don't sleep after last batch
            print("Waiting 5 seconds before next batch...")
            time.sleep(5)

    # Overall stats
    print("\n=== Overall Stats ===")
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    rate_limited = sum(1 for r in results if r["status"] == "rate_limited")
    errors = sum(1 for r in results if r["status"] == "error")

    print(f"Total requests: {total}")
    print(f"Success: {success} ({100*success/total:.1f}%)")
    print(f"Rate limited: {rate_limited} ({100*rate_limited/total:.1f}%)")
    print(f"Errors: {errors} ({100*errors/total:.1f}%)")

    if success > 0:
        avg_time = sum(r["time"] for r in results if r["status"] == "success") / success
        print(f"Avg response time: {avg_time:.2f}s")

if __name__ == "__main__":
    main()
