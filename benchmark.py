import asyncio
import time
import httpx
import json

BASE_URL = "http://127.0.0.1:8000/v1"

async def test_endpoint(name: str, payload: dict):
    print(f"\n=== Testing {name} ===")
    print(f"Payload: {json.dumps(payload, ensure_ascii=False)}")
    
    start_time = time.time()
    ttft = None
    char_count = 0
    full_content = ""
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if payload.get("stream", False):
                async with client.stream("POST", f"{BASE_URL}/chat/completions", json=payload) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        print(f"Error {response.status_code}: {body.decode('utf-8')}")
                        return
                    
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        
                        if ttft is None:
                            ttft = time.time() - start_time
                            print(f"Time to First Chunk (TTFT): {ttft:.3f}s")
                        
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                content = data_json["choices"][0]["delta"].get("content", "")
                                full_content += content
                                char_count += len(content)
                            except Exception:
                                # For HTML comment lines in Fusion
                                if data_str.startswith("<!--"):
                                    print(f"Debug info: {data_str}")
            else:
                response = await client.post(f"{BASE_URL}/chat/completions", json=payload)
                if response.status_code != 200:
                    print(f"Error {response.status_code}: {response.text}")
                    return
                res_data = response.json()
                full_content = res_data["choices"][0]["message"]["content"]
                char_count = len(full_content)
                ttft = time.time() - start_time  # For non-stream, TTFT is total time
                
        total_time = time.time() - start_time
        print(f"Total Latency: {total_time:.3f}s")
        print(f"Generated Characters: {char_count}")
        if char_count > 0 and total_time > 0:
            print(f"Generation Speed: {char_count / total_time:.2f} chars/sec")
        print("Response Snippet:")
        print("-" * 40)
        print(full_content[:200] + ("..." if len(full_content) > 200 else ""))
        print("-" * 40)
        
    except Exception as e:
        print(f"Exception occurred: {str(e)}")

async def main():
    print("Checking proxy health...")
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(f"{BASE_URL}/health")
            print(f"Health Status: {res.json()}")
        except Exception as e:
            print(f"Could not connect to proxy: {str(e)}")
            return

    # Test 1: Standard Fallback Routing (General / Creative prompt, stream=True)
    payload_routing = {
        "model": "MiniMax-M3",  # This will execute under standard routing/fallback
        "messages": [{"role": "user", "content": "请讲一个程序员自嘲的段子，不超过50字。"}],
        "stream": True
    }
    await test_endpoint("Standard Routing (Stream)", payload_routing)

    # Test 2: Model Fusion MoA (Reasoning/Complex prompt, model=model-fusion, stream=True)
    payload_fusion = {
        "model": "model-fusion",
        "messages": [{"role": "user", "content": "Compare the architectural differences and trade-offs of using gRPC vs REST APIs in microservices."}],
        "stream": True
    }
    await test_endpoint("Model Fusion MoA (Stream)", payload_fusion)

if __name__ == "__main__":
    asyncio.run(main())
