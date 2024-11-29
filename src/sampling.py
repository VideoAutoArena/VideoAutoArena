import json
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description="Process battle data and generate new samples.")
    parser.add_argument('--n_sample', type=int, default=1500, help='Number of samples to generate')
    parser.add_argument('--battle_file', type=str, default="", help='Path to the battle data file')
    parser.add_argument('--LMM_name', type=str, default="NewLMM", help='Name of the LMM model')
    parser.add_argument('--output_path', type=str, default="", help='Path to save the output JSON file')

    args = parser.parse_args()

    n_sample = args.n_sample
    battles = json.load(open(args.battle_file, "r"))
    LMM_name = args.LMM_name
    output_path = args.output_path

    random.shuffle(battles)

    samples = battles[:n_sample]
    new_data = []
    for sample in samples:
        if random.randint(0, 1) > 0:
            pick, nopick = "b", "a"
        else:
            pick, nopick = "a", "b"
        new_data.append(
            {
                "video_id": sample["video_id"],
                "qid": sample["qid"],
                "persona": sample["persona"],
                "question": sample["question"],
                f"model {pick}": LMM_name,
                f"model {nopick}": sample[f"model {nopick}"],
                f"model {nopick} answer": sample[f"model {nopick} answer"]
            }
        )
    json.dump(new_data, open(output_path, "w"), indent=4)

if __name__ == "__main__":
    main()