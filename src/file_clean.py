import json
import os
import argparse
import pandas as pd

def args_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--directory', type=str, default='result_dirs/mt-bench/urial_bench', help='Directory to search for files')
    parser.add_argument('--model_name', type=str, required=True, help='String to search for in file names')
    parser.add_argument('--output_path', type=str, default='result_dirs/mt-bench/csv_result')
    
    return parser.parse_args()

def find_files_containing(directory, model_name):
    matching_files = []
    for files in os.listdir(directory):
        print(files)
        if model_name in files:
            matching_files.append(files)
    return matching_files

if __name__ == '__main__':
    args = args_parse()
    args.model_name = args.model_name.split("/")[-1]
    
    matching_files = find_files_containing(args.directory, args.model_name)
    
    result_dict = {
        "question_id": [],
        "category": [],
        "turn1_instruction": [],
        "turn1_output": [],
        "turn2_instruction": [],
        "turn2_output": []
    }
    
    data = []
    for files in matching_files:
        with open("/".join([args.directory, files]), "r") as f:
            data.extend(json.load(f))
    
    count = 0
    for d in data:
        if count < (len(data) / 2):
            result_dict["question_id"].append(d["question_id"])
            result_dict["category"].append(d["category"])
            count += 1
        result_dict[f"turn{d['turn_id']}_instruction"].append(d["model_input"])
        result_dict[f"turn{d['turn_id']}_output"].append(d[f"turn{d['turn_id']}_output"])
        
    df = pd.DataFrame(result_dict)
    df.to_csv(args.output_path + f"{args.model_name}.csv", index=False)