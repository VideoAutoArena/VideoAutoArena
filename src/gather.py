import os
import json
import argparse

def load_json_files(input_dir):
    data = []
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        try:
            tmp = json.load(open(file_path, "r"))
            if tmp["New Question"] != "":
                data.append(
                    {
                        "video_id": tmp["video_id"],
                        "qid": tmp["qid"],
                        "persona": tmp["persona"],
                        "evol": True,
                        "question": tmp["New Question"],
                        "model a": tmp["model a"],
                        "model b": tmp["model b"],
                    }
                )
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON file: {file_path}")
        except:
            pass
    print(len(data))
    return data

def save_json_data(data, output_file):
    try:
        json.dump(data, open(output_file, "w"), indent=4)
        print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving data to {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Combine JSON files into a single JSON file.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing JSON files.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save combined JSON data.")

    args = parser.parse_args()

    data = load_json_files(args.input_dir)
    save_json_data(data, args.output_file)

if __name__ == "__main__":
    main()