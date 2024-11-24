"""define a DP inference class with
utility functions.
"""

from abc import ABC, abstractmethod
import re
from typing import Text, Dict, Any, List, Optional
from torch.utils.data import DataLoader

CONTROL_FLOWS = ['if statement', 'for loop', 'while loop', 'break statement', 'continue statement', 'pass statement', 'match statement', 'recursion']
DATA_STRUCTURES = ['stack', 'queue', 'tuple', 'set', 'dictionary', 'linked list', 'tree', 'graph']
ALGORITHMS = ['two pointers', 'sliding window', 'matrix operation', 'hashmap', 'depth first search', 'width first search', 'back tracking', 'divide & conquer', 'Kadanes algorithm', 'binary search', 'heap', 'dynamic programming', 'greedy algorithm', 'misc', 'minimax', 'topological sort', 'sorting', 'graph traversal']
TECHNIQUES = CONTROL_FLOWS + DATA_STRUCTURES + ALGORITHMS

class CodeGenerator(ABC):
    """
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def inference(self, dataloader: DataLoader, save_path: str, overwrite: bool):
        """
        """
        raise NotImplementedError("CodeGenerator is an abstract class.")

    def parse_response(self, code: Text) -> Optional[Text]:
        """Parse the response from the API model to get code
        """
        first = ["```python", "```Python"]
        try:
            start = min([code.index(f) for f in first if f in code]) + len(first[0])
            if "```" in code[start:]:
                end = code.index("```", start)
            else:
                end = len(code)
            return code[start:end]
        except:
            return None
    
    def parse_techniques(self, tecnique_str: Text) -> List[Text]:
        """Parse the techniques used in the code
        """
        # delete the content in the bracket
        technique_list = [re.sub(r'\s*\(.*?\)', '', s.strip('- ').strip()) for s in tecnique_str.split('\n')]
        # delete quotation marks 
        technique_list = [s.replace('"', '').replace("'", "") for s in technique_list]      
        technique_list_filtered = []
        for technique in technique_list:
            if technique in TECHNIQUES:
                technique_list_filtered.append(technique)
        if len(technique_list_filtered) == 0:
            return ['for loop']
        return technique_list_filtered

    def get_constraint_problem(self, 
                               problem_statement: Text, 
                               techniques: List[Text]) -> Text:
        lines = problem_statement.split('\n')
        if lines[1].startswith('Programming constraints'):
            lines[1] += f'\n- ' + '\n- '.join(techniques)
        else:
            lines[0] += f'\nProgramming constraints: DO NOT use the following techniques\n- ' + '\n- '.join(techniques)
        constraints_str = '\n'.join(lines)
        return constraints_str
