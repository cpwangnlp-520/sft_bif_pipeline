from vllm import LLM, SamplingParams
import json

def main():
    model_path = "/workspace/pku_percy/models/Qwen2.5-0.5B-Instruct"

    with open('/workspace/new-preject/train-pipeline/data/xstest_inference_100.jsonl') as f:
        prompts = [json.loads(l) for l in f]

    llm = LLM(model=model_path, tensor_parallel_size=1, trust_remote_code=True, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9)

    sampling = SamplingParams(temperature=0, max_tokens=2048)

    conversations = [[{"role": "user", "content": p["messages"][0]["content"]}] for p in prompts]
    outputs = llm.chat(messages=conversations, sampling_params=sampling)

    results = []
    for prompt_info, output in zip(prompts, outputs):
        response = output.outputs[0].text
        results.append({
            "id": prompt_info["id"],
            "messages": [
                {"role": "user", "content": prompt_info["messages"][0]["content"]},
                {"role": "assistant", "content": response},
            ],
            "source": prompt_info["source"],
            "type": prompt_info.get("type", ""),
            "focus": prompt_info.get("focus", ""),
        })

    with open("/workspace/new-preject/train-pipeline/data/xstest_0.5b_instruct_inference_100.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    safe = sum(1 for r in results if r["source"] == "safe")
    unsafe = sum(1 for r in results if r["source"] == "unsafe")
    print(f"Done: {safe} safe + {unsafe} unsafe = {len(results)} total")

if __name__ == "__main__":
    main()
