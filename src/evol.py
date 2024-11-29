import os, re
import time
import json
import argparse
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

Your task is to carefully evaluate the responses of Model A and Model B to identify any faults and models' weaknesses. Based on these weaknesses, generate a harder [New Question] that explores aspects where the models may struggle. **Remember**, the [New Question] should continue to align with the user's persona, without necessarily being more specific.

You **MUST** follow these steps to generate the [New Question]:

Step 1: Carefully review the chat history to identify any problems in the two models' responses related to understanding the video's content. Consider:
- Whether the responses correctly incorporate information from the video.
- Whether the responses are helpful in fulfilling the user's requirements.
- Whether the responses are user-aware and align with the user's persona.  
List all identified faults in [Fault List A] for Model A and [Fault List B] for Model B.

Step 2: Based on [Fault List A] and [Fault List B], summarize the weaknesses of Model A and Model B. List all identified weaknesses in [Weakness List A] for Model A and [Weakness List B] for Model B. The weaknesses should be more general than the faults.

Step 3: Based on [Weakness List A] and [Weakness List B], craft a high-quality harder [New Question] that will further explore aspects where the two models may struggle. If both responses do not contain any faults, you still need to create a different harder [New Question] that mimics the interaction between the user and the AI assistant models. The [New Question] is **not** a follow-up question.

**Remember**, the [New Question] should continue to align with the user's persona, without necessarily being more specific. Ensure that:
- The [New Question] can only be answered by someone who has watched the video.
- Do **NOT** leak key visual details in the [New Question].  
If necessary, include response format requirements (Markdown, JSON, Table, List, etc.) in the [New Question].

Please respond strictly in this format:

Step 1:
```[Fault List A]
xxx
```

```[Fault List B]
xxx
```

Step 2:
```[Weakness List A]
xxx
```

```[Weakness List B]
xxx
```

Step 3:
```[New Question]
Only include the new question here!
```""",
}

def response_parse(text: str) -> dict:
    """
    Parse the response text and extract key-value pairs.

    Args:
        text (str): The response text to parse.

    Returns:
        dict: A dictionary containing the parsed key-value pairs.
    """
    pattern = re.compile(r'\[(.*?)\](.*?)(?=```|$)', re.DOTALL)
    content_dict = {}

    for match in pattern.finditer(text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        content_dict[key] = value

    return content_dict

def run_one_prompt(paths: list) -> None:
    """
    Process a single prompt and save the result.

    Args:
        paths (list): A list containing the index, sample data, and output directory.
    """
    idx, sample, output_dir = paths
    video_id = sample["video_id"]
    qid = sample["qid"]
    persona = sample["persona"]
    question = sample["question"]
    model_a_answer = sample["model a answer"]
    model_b_answer = sample["model b answer"]

    output_path = os.path.join(output_dir, f'{qid}.jsonl')

    if os.path.exists(output_path):
        print(f'{output_path} already exists, skipping.')
        return

    video_inputs = video_data.get_w_video_id(video_id)["inputs"]

    try:
        chosen_prompt = PROMPTS['role'].format(
            persona=persona, 
            question=question, 
            answer_a=model_a_answer, 
            answer_b=model_b_answer
        )
        response = request(
            video_inputs=video_inputs, 
            prompt=chosen_prompt
        )

        sample["Evol Response"] = response
        response_dict = response_parse(response)

        for key, value in response_dict.items():
            sample[key] = value

        dump_jsonl([sample], output_path)
        print(f'Saved {output_path}')

    except Exception as e:
        print(f"Error: {e}")
        print(f"Error on video: {video_id}")
        print(f"Error on output_path: {output_path}")

def multi_process_request(data: list, output_dir: str, worker_num: int = 10) -> None:
    """
    Process multiple prompts in parallel using multi-processing.

    Args:
        data (list): List of samples to process.
        output_dir (str): Directory to store the output results.
        worker_num (int): Number of workers for multi-processing.
    """
    total_lines = len(data)
    print(f"Total samples: {total_lines}")

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
                    f"[worker_num={worker_num}] Processed {count} lines, "
                    f"average time: {average_time:.2f}s, "
                    f"expected time: {average_time / 60 * (total_lines - count):.2f}min"
                )

    print(f'Finished processing {total_lines} and took {time.time() - start_time:.2f}s')

def make_sample_data(battle_path: str) -> list:
    """
    Load sample data from the given battle path.

    Args:
        battle_path (str): Path to the battle data file.

    Returns:
        list: List of sample data.
    """
    battle = json.load(open(battle_path, "r"))
    return battle

def main() -> None:
    """
    Main function to parse arguments and start the processing.
    """
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