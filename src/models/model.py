"""Open- and closed- source model for generating rationales
"""
import os
import json
import random
import numpy as np
from datetime import timedelta
from abc import ABC, abstractmethod
import backoff
from overrides import overrides
import openai
from openai import OpenAI
import transformers
from vllm import LLM
import torch
import torch.distributed as dist
import boto3
# import deepspeed

CACHE_DIR="/scratch4/danielk/ylu130/model-hf/"
completion_tokens = prompt_tokens = 0

class ds_args(object):
    """Dummy argument class.
    The object will be passed to deepspeed.initialize()
    if needed for data parallelism.
    """
    def __init__(self,
                 local_rank: int,
                 deepspeed_config: str,
                 seed: int,
                 deepspeed: bool
                 ):
        super().__init__()
        self.local_rank = local_rank
        self.deepspeed_config = deepspeed_config
        self.deepspeed = deepspeed
        self.seed = seed

class OpenModel(ABC):
    """
    """
    def __init__(self,
                 model_name: str,
                 prompt: str):
        super().__init__()
        self.model_name = model_name
        self.prompt = prompt

    @abstractmethod
    def load_model(self):
        """
        """
        raise NotImplementedError("Model is an abstract class.")


class OpenModelVLLM(OpenModel):
    def __init__(self,
                 model_name: str,
                 prompt: str):
        
        super().__init__(model_name, prompt)
        self.load_model()

    @overrides
    def load_model(self):
        num_gpus = torch.cuda.device_count()
        self.model = LLM(model=self.model_name, 
                         download_dir=CACHE_DIR, 
                         tensor_parallel_size=num_gpus,
                        #  gpu_memory_utilization=0.95,
                        #  max_model_len=8192
                         )
        self.tokenizer = None

class OpenModelHF(OpenModel):
    def __init__(self,
                 model_name: str,
                 prompt: str):
        
        super().__init__(model_name, prompt)    
        self.load_model()

    @overrides
    def load_model(self):
        args = ds_args(local_rank=0, 
                       deepspeed_config="src/utils/ds_config.json", 
                       deepspeed=True,
                       seed=42)
        self.initialize(args)
        if "t5" in self.model_name.lower():
            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.model_name, 
                                                                            cache_dir=CACHE_DIR,
                                                                            trust_remote_code=True,
                                                                            device_map="auto",
                                                                            torch_dtype=torch.float16
                                                                            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                                           cache_dir=CACHE_DIR,
                                                                           trust_remote_code=True,
                                                                           device_map="auto",
                                                                           torch_dtype=torch.float16
                                                                           )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # initialize model for data parallelism using deepspeed
        # with open(args.deepspeed_config, "r") as f:
        #     ds_config = json.load(f)
        # args.deepspeed_config = None
        # model, _, _, _ = deepspeed.initialize(
        #     model=model,
        #     optimizer=None,
        #     args=args,
        #     lr_scheduler=None,
        #     mpu=None,
        #     config_params=ds_config
        # )

    def initialize(self, args):

        self.init_distributed(args)

        self.set_random_seed(args.seed)

    def set_random_seed(self, seed):
        """Set random seed for reproducability.
        """
        # seed = dist.get_rank() + seed
        if seed is not None and seed > 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def init_distributed(self, args):
        """Initialize distributed inference.
        """
        args.rank = int(os.environ["SLURM_PROCID"]) # this is the rank of the current GPU
        args.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        args.local_rank = args.rank - args.gpus_per_node * (args.rank // args.gpus_per_node) # this is the rank of the current GPU within the node
        args.world_size = int(os.getenv("WORLD_SIZE", "1")) # this is the number of GPUs

        if args.rank == 0:
            print(f"using world size: {args.world_size}")

        # Manually set the device ids.
        self.device = args.local_rank
        torch.cuda.set_device(self.device)

        # dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size, timeout=timedelta(seconds=30))

class OpenAIModel:
    def __init__(self, 
                 model: str,
                 temperature: float = 1,
                 max_tokens: int = 256,
                 top_p: float = 1,
                 n: int = 1,
                 gpt_setting: str = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.n = n
        self.gpt_setting = gpt_setting
        self.restart()
        self.client = OpenAI()
        
    @backoff.on_exception(backoff.expo, openai.OpenAIError)
    def chatcompletions_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    @backoff.on_exception(backoff.expo, openai.OpenAIError)
    def completions_with_backoff(self, **kwargs):
        return self.client.completions.create(**kwargs)

    def chatgpt(self) -> list:
        global completion_tokens, prompt_tokens
        outputs = []
        res = self.chatcompletions_with_backoff(model=self.model, 
                                                messages=self.message, 
                                                temperature=self.temperature, 
                                                max_tokens=self.max_tokens, 
                                                n=self.n,
                                                top_p=self.top_p)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
        return outputs

    def completiongpt(self) -> list:
        global completion_tokens, prompt_tokens
        outputs = []
        res = self.completions_with_backoff(model=self.model, 
                                            messages=self.message, 
                                            temperature=self.temperature, 
                                            max_tokens=self.max_tokens, 
                                            n=self.n, 
                                            top_p=self.top_p)
        outputs.extend([choice.text for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
        return outputs

    @staticmethod
    def gpt_usage(model="gpt-4-1106-preview"):
        global completion_tokens, prompt_tokens
        if model == "gpt-4":
            cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
        elif model == "gpt-4-1106-preview":
            cost = completion_tokens / 1000 * 0.03 + prompt_tokens / 1000 * 0.01
        elif model == "gpt-3.5-turbo":
            cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
        elif "davinci" in model:
            cost = completion_tokens / 1000 * 0.02 + prompt_tokens / 1000 * 0.02
        return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

    def __call__(self, input: str) -> list:
        if "davinci" in self.model:
            self.message = self.message + "\nInput: " + input
            return self.completiongpt()
        else:
            self.message.append({"role": "user", "content": input})
            return self.chatgpt()
        
    def update_message(self, output: str):
        if "davinci" in self.model:
            self.message = self.message + "\nOutput: " + output
        else:
            self.message.append({"role": "assistant", "content": output})
    
    def restart(self):
        if "davinci" in self.model:
            self.message = ""
        else:
            self.message = [{"role": "system", "content": self.gpt_setting}] if self.gpt_setting else []


class AnthropicModel:

    def __init__(self, 
                 model: str,
                 temperature: float = 1,
                 max_tokens: int = 256,
                 top_p: float = 1,
                 gpt_setting: str = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.gpt_setting = gpt_setting
        
        AWS_KEY = os.environ["AWS_KEY"]
        AWS_SECRET = os.environ["AWS_SECRET"]

        if AWS_KEY == None or AWS_SECRET == None:
            raise Exception("AWS_KEY or AWS_SECRET not found")
        
        self.bedrock = boto3.client(service_name='bedrock-runtime',
                                    region_name='us-east-1',
                                    aws_access_key_id=AWS_KEY,
                                    aws_secret_access_key=AWS_SECRET)
    
    @backoff.on_exception(backoff.expo, Exception, max_time=60)
    def __call__(self, input: str) -> str:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "system": self.gpt_setting,
            "messages": [{"role": "user", "content": input}],
        })

        response = self.bedrock.invoke_model(body=body, modelId=self.model)
        response_body = json.loads(response.get("body").read())
        result = response_body.get("content")[0]["text"]

        return result