import os
import re
from abc import ABC, abstractmethod

class Evaluator(ABC):
    """
    """
    def __init__(self,
                 inference_result_path: str,
                 test_case_path: str) -> None:
        super().__init__()

        self.inference_result_path = inference_result_path
        self.test_case_path = test_case_path

        base_name = os.path.basename(inference_result_path)
        self.num_sample = int(base_name.split("sample=")[1].split("_dp")[0])
        self.num_dp = int(re.search(r"dp=(\d+)", base_name).group(1))
        self.model_name = base_name.split("_diff")[0] if "diff" in base_name else base_name.split("_sample")[0]

    @abstractmethod
    def evaluate(self) -> None:
        """
        """
        raise NotImplementedError("Evaluator is an abstract class.")