"""Prepare data for inference
"""
from torch.utils.data import Dataset
import json
import re

class CodeforceDPGenerateDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 num=-1,):
        super().__init__()
        self.data = self.load_problem_jsonl(data_path)    
        if num > 0:
            self.data = self.data[:num]
            
        self.num = min(num, len(self.data)) if num > 0 else len(self.data)

    def load_problem_jsonl(self, path):
        with open(path, "r") as f:
            data = [json.loads(line) for line in f]
        
        # remove crawled noise
        for item in data:
            item['problem_statement'] = re.sub(
                r'time limit per test\n\d+ seconds?\nmemory limit per test\n\d+ megabytes\ninput\nstandard input\noutput\nstandard output\n', 
                '', 
                item['problem_statement']
            )

        return data

    def __len__(self):
        return self.num

    def __getitem__(self, index: int):
        
        return self.data[index]

class CodexDPGenerateDataset(Dataset):
    def __init__(self,
                 data_path,
                 num=-1,):
        super().__init__()
        self.data = self.load_problem_json(data_path)
        if num > 0:
            self.data = self.data[:num]
        
        self.num = min(num, len(self.data)) if num > 0 else len(self.data)
    
    def load_problem_json(self, path):
        with open(path, "r") as f:
            problems = json.load(f)
        
        data = []
        for task_id in problems:
            data.append({'problem_id': task_id, 'problem_statement': problems[task_id]["prompt"]})

        return data
    
    def __len__(self):
        return self.num
    
    def __getitem__(self, index: int):
        return self.data[index]


class DPInferenceDataset(Dataset):
    def __init__(self, 
                 data_path,
                 dp_rounds):
        super().__init__()
        self.data = self.load_problem_json(data_path, dp_rounds)

    def load_problem_json(self, path, dp_rounds):
        with open(path, "r") as f:
            data = json.load(f)

        # # filter out problems that have less than dp_rounds constrains
        # data = [d for d in data if max(list(map(len, d['constraints_list']))) >= dp_rounds]

        for item in data:
            # include the og problem
            item['problem_statements'] = item['problem_statements'][:dp_rounds+1]
            item['constraints_list'] = item['constraints_list'][:dp_rounds+1]
            item['codes'] = item['codes'][:dp_rounds+1]
            
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        return self.data[index]
