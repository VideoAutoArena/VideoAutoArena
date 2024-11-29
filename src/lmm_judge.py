import os, re
import time
import json, argparse
from load_longvideobench import LongVideoBenchDataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from call_gpt4o import request
from utils import dump_jsonl


# Global variable for video_data
video_data = LongVideoBenchDataset(os.getenv('LVB_PATH'), "lvb_test_wo_gt.json", max_num_frames=128)


PROMPTS = {
    "role": """**Remember: You are watching a Video.**

A user, characterized by a specific persona, is interacting with two AI assistant models (A and B) to better understand video content using the same question. Here is the user's persona:
```persona
{persona}
```

The user's question is:
```question
{question}
```

The response from Model A is:
```model_a
{answer_a}
```

The response from Model B is:
```model_b
{answer_b}
```

Please act as an impartial judge and carefully evaluate the responses of Model A and Model B to determine which one is better. Use the following standards:

1. [Instruction Following]: The response should closely adhere to the user's instructions, ensuring it directly addresses the specified task.
2. [Accuracy]: The response must accurately utilize information from the video, avoiding fabrication or misquotation. It should maintain factual correctness, avoid hallucinations, and demonstrate contextual coherence with precise terminology and knowledge.
3. [Relevance]: The response should consider the user's background information and needs, providing a comprehensive, detailed answer that addresses the question directly without straying off-topic. Responses should be thorough, offering multiple perspectives where relevant.
4. [Helpfulness]: The response should provide valuable information to aid the user in understanding or solving their issue, avoiding irrelevant or vague content.

If the responses from Model A and Model B are of similar quality (whether both are good or both are bad), you may declare a tie.

**Please follow these steps for your judgment:**

- Step 1: Analyze which model provides a better response for the [Instruction Following] standard.
- Step 2: Analyze which model provides a better response for the [Accuracy] standard.
- Step 3: Analyze which model provides a better response for the [Relevance] standard.
- Step 4: Analyze which model provides a better response for the [Helpfulness] standard.
- Step 5: Based on the results from Steps 1-4, determine the overall outcome: Model A, Model B, Tie (both are good), or Tie (both are bad).

Please respond strictly in the following format:

```[Instruction Following]
[Your Analysis]
```

```[Accuracy]
[Your Analysis]
```

```[Relevance]
[Your Analysis]
```

```[Helpfulness]
[Your Analysis]
```

```[Overall Judge]
A/B/Tie
```"""
}

def response_parse(text):
    """Parse the response text into a dictionary."""
    pattern = re.compile(r'\[(.*?)\](.*?)(?=\[|$)', re.DOTALL)
    content_dict = {}
    
    for match in pattern.finditer(text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        value = re.sub(r'^```\n?|\n?```$', '', value)
        content_dict[key] = value
    
    return content_dict

def run_one_prompt(paths):
    """Process a single prompt and save the result."""
    idx, sample, output_dir = paths
    video_id = sample["video_id"]
    qid = sample["qid"]
    persona = sample["persona"]
    question = sample["question"]
    model_a_answer = sample["model a answer"]
    model_b_answer = sample["model b answer"]
    output_path = os.path.join(output_dir, f'{qid}.jsonl')
    
    if os.path.exists(output_path):
        print(f'{output_path} already exists, skipping...')
        return
    
    video_inputs = video_data.get_w_video_id(video_id)["inputs"]

    try:
        chosen_prompt = PROMPTS['role'].format(
            persona=persona, 
            question=question, 
            answer_a=model_a_answer, 
            answer_b=model_b_answer
        )
        
        response = request(video_inputs=video_inputs, prompt=chosen_prompt)
        sample["GPT4o Judge response"] = response
        
        response_dict = response_parse(response)
        for key, value in response_dict.items():
            sample[key] = value

        dump_jsonl([sample], output_path)
        print(f'Saved {output_path}')

    except Exception as e:
        print(f"Error: {e}")
        print(f"Error on video: {video_id}")
        print(f"Error on output_path: {output_path}")

def multi_process_request(data, output_dir, worker_num=10):
    """Process data using multiple workers."""
    total_samples = len(data)
    print(f"Total samples: {total_samples}")
    
    with ProcessPoolExecutor(max_workers=worker_num) as executor:
        start_time = time.time()
        count = 0
        futures = [executor.submit(run_one_prompt, [idx, obj, output_dir]) for idx, obj in enumerate(data)]
        
        for job in as_completed(futures):
            job.result(timeout=None)
            end_time = time.time()
            average_time = (end_time - start_time) / (count + 1)
            count += 1

            if count % 100 == 0:
                print(
                    f"[worker_num={worker_num}] Processed {count} samples, "
                    f"average time: {average_time:.2f}s, "
                    f"expected time: {average_time / 60 * (total_samples - count):.2f}min"
                )

    print(f'Finished processing {total_samples} samples in {time.time() - start_time:.2f}s')

def make_sample_data(battle_path):
    """Load sample data from the given path."""
    with open(battle_path, "r") as file:
        battle = json.load(file)
    return battle

def main():
    """Main function to parse arguments and start processing."""
    parser = argparse.ArgumentParser(description="Process video QA with different models.")
    parser.add_argument("--battle_path", type=str, required=True, help="Path to the battle data file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store the output results.")
    parser.add_argument("--worker_num", type=int, default=32, help="Number of workers for multi-processing.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = make_sample_data(args.battle_path)

    multi_process_request(data, args.output_dir, worker_num=args.worker_num)

if __name__ == "__main__":
    main()