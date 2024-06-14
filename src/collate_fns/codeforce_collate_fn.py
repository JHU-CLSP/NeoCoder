
from transformers import PreTrainedTokenizer
from typing import Text, Dict, Any, List, Optional

from .collate_fn import CollateFn
from overrides import overrides

__PROMPT_TEMPLATE__ = {
    True: "{prompt} \nProgramming Problem: {problem_statement}",
    False: "{problem_statement}"
}


class DPGenerateCollateFn(CollateFn):
    
    def __init__(
        self,
    ):
        """ Collate function to feed crawled problem 
        into the model for single-thread DP generation
        """
        super().__init__()

    @overrides
    def collate(self, x: List[Dict[Text, Any]]) -> Dict[Text, Any]:
        """
        """
        problem_statements = [
            p['problem_statement'] for p in x
        ]
        problem_ids = [
            p['problem_id'] for p in x
        ]

        return {'problem_id': problem_ids, 
                'input': problem_statements}

class CodeforceDPInferenceCollateFn(CollateFn):

    __PROMPT_TEMPLATE__ = __PROMPT_TEMPLATE__
    
    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizer],
                 is_open_model: bool,
                 use_vllm: bool,
                 dp_rounds: int,
                 prompt: Text = None):
        """ Collate function to feed DP generated dataset 
        into the model for parallel-thread DP inference
        """
        super().__init__()
        self.is_open_model = is_open_model
        self.use_vllm = use_vllm
        self.tokenizer = tokenizer
        self.dp_rounds = dp_rounds
        self.prompt = prompt

    def template(self, problem_statements: List[Text]) -> Text:
        """Template the problem statement with prompt
        """
        return [self.__PROMPT_TEMPLATE__[self.is_open_model].format(
                prompt=self.prompt,
                problem_statement=problem_statement
            ) for problem_statement in problem_statements]

    @overrides
    def collate(self, x: List[Dict[Text, Any]]) -> Dict[Text, Any]:
        """
        """
        # constraints are included in each problem statement
        problem_statements: List[List[Text]] = [
            self.template(p['problem_statements']) for p in x
        ]
        problem_ids: List[Text] = [
            p['problem_id'] for p in x
        ]
        # batch x dp_rounds x constraints
        problem_constrains: List[List[List[Text]]] = [
            p['constraints_list'] for p in x
        ]

        unfolded_problem_statements: List[Text] = [x for y in problem_statements for x in y]
        assert len(unfolded_problem_statements) / (self.dp_rounds+1) == len(problem_ids)


        if self.is_open_model and not self.use_vllm:
            assert self.tokenizer is not None, "Tokenizer is required for huggingface model inference"
            tokenized = self.tokenizer(
                unfolded_problem_statements,
                max_length=2048,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            return {'problem_ids': problem_ids,
                    'problem_statements': problem_statements, 
                    'inputs': {"input_ids": tokenized.input_ids, 
                              "attention_mask": tokenized.attention_mask},
                    'constraints': problem_constrains}
        else:
            
            return {'problem_ids': problem_ids, 
                    'inputs': unfolded_problem_statements,
                    'constraints': problem_constrains}