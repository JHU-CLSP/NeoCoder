python steps/generate_dp.py \
        --problem-set-dir "datasets/CodeForce/crawled/codeforces_problems.jsonl" \
        --model-name gpt-4-1106-preview \
        --num-sample 200 \
        --dp-rounds 5 \
        --output-dir datasets/CodeForce/NeoCoder
