import time
import copy
import numpy as np
import pandas as pd
import argparse
from typing import Dict, Optional, Sequence
from dataclasses import dataclass, field
import torch
import transformers
from transformers import Trainer,TrainingArguments
from softembedding import PreSoftEmbedding, InterSoftEmbedding
#from data import PreSoftDataset, InterSoftDataset
from config import CONFIG
from config import set_seed
import warnings
warnings.filterwarnings("ignore")
import json
#import utils
from torch.utils.data import Dataset
#from peft.modeling import prepare_peft
from trl import SFTTrainer
from peft import LoraConfig
import torch.nn as nn

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


#example of the INSTRUCTION
INSTRUCTION= "You are an excellent linguist. The task is to predict relationship between the given head entity and tail entity within a given sentence, this relation which must be in ('ANTAGONIST', 'ACTIVATOR, INDIRECT-UPREGULATOR or UPREGULATOR', 'AGONIST, AGONIST-ACTIVATOR,or AGONIST-INHIBITOR', 'DOWNREGULATOR or INDIRECT-DOWNREGULATOR', 'PRODUCT-OF or SUBSTRATE'), for the given sentence.  "
PROMPT_DICT={
"instruction": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nYou are an excellent linguist. The task is to predict relationship between the given head entity and tail entity within a given sentence, this relation which must be in (\'ANTAGONIST\', \'ACTIVATOR, INDIRECT-UPREGULATOR or UPREGULATOR\', \'AGONIST, AGONIST-ACTIVATOR,or AGONIST-INHIBITOR\', \'DOWNREGULATOR or INDIRECT-DOWNREGULATOR\', \'PRODUCT-OF or SUBSTRATE\'), for the given sentence.\n",
"prompt_example":(
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\nExample: {example} \n### Input:\n{input}\n\n### Response:{response}"
    ),
"prompt_no_example":{
    "prompt_example":(
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n### Input:\n{input}\n\n### Response:"
    )
},
"prompt_no_input":{
    "prompt_example":(
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n### Response:"),
}
}

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""

    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            #max_length=tokenizer.model_max_length,
            truncation=False,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def get_fix_tensor_from_tokenizer(tokenizer,n_tokens):
    assert isinstance(n_tokens, int)
    vocab_size = tokenizer.vocab_size  # Get the size of the tokenizer's vocabulary
    #size_tuple = (int(n_tokens),)
    #random_token_ids = torch.randint(high=vocab_size, size=size_tuple).long()
    random_token_id = torch.randint(high=vocab_size, size=(1,)).long()
    prefix_tensor = random_token_id.repeat(1, n_tokens)
    return prefix_tensor


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    n_tokens: int,
    tokenizer: transformers.PreTrainedTokenizer,
    method: str='prefix',
    initial_from_model:bool=False,
   
) -> Dict:
    """Preprocess the data by tokenizing."""
    #examples = [s + t for s, t in zip(sources, targets)]
    examples=sources
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    
   
    instruction_tokenized=_tokenize_fn([INSTRUCTION],tokenizer)
    instruction_tokenized_idx=instruction_tokenized['input_ids'][0]

    len_instruction_tokenized_idx=int(instruction_tokenized['input_ids_lens'][0])
    
    if initial_from_model:
         prefix_tensor = get_fix_tensor_from_tokenizer(tokenizer,n_tokens)
    else:
        #The prefix tensor here is fix as 500
        prefix_tensor = torch.full((1, n_tokens), 500, dtype=torch.long).squeeze()
    input_ids = examples_tokenized["input_ids"]
   
    if method.lower()=='prefix': #instruction adaptive
        input_ids = [torch.cat((prefix_tensor, torch.tensor(e, dtype=torch.long))) for e in examples_tokenized['input_ids']]

    elif method.lower()=='interfix': #example adaptive
        input_ids =[torch.cat((instruction_tokenized_idx,
                               prefix_tensor,
                               torch.tensor(e[len_instruction_tokenized_idx-1:], dtype=torch.long))) for e in examples_tokenized['input_ids']]
        
    elif method.lower()=='combine':#combine adaptive
        input_ids =[torch.cat((prefix_tensor,
                               instruction_tokenized_idx,
                               prefix_tensor,
                               torch.tensor(e[len_instruction_tokenized_idx-1:], dtype=torch.long))) for e in examples_tokenized['input_ids']]
    else:
         input_ids = examples_tokenized["input_ids"]

    
    labels = copy.deepcopy(input_ids)
    #for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
    #    if method=='combine':
    #        label[:source_len+n_tokens*2] = IGNORE_INDEX
    #    else:
    #        label[:source_len+n_tokens] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer,n_tokens:int,method: str='prefix',initial_from_model:bool=False):
        super(SupervisedDataset, self).__init__()
        #logging.warning("Loading data...")
        list_data_dict = load_jsonl(data_path)

        #logging.warning("Formatting inputs...")
        
        sources = [
            PROMPT_DICT['prompt_example'].format(
            instruction=e.get('instruction', ''),
            example=e.get('example', ''),
            input=e.get('input', ''),
            response=e.get('response', '')
            )
            for e in list_data_dict]

        #source=instruction+example+input/=instruction+input
        targets = [f"{example['response']}{tokenizer.eos_token}" for example in list_data_dict]

        #logging.warning("Tokenizing inputs... This may take some time...")
        #data_dict = preprocess(sources, targets, tokenizer,method,n_tokens,initial_from_model)
        data_dict=preprocess( sources=sources,
                            targets=targets,
                            n_tokens=n_tokens,
                            tokenizer=tokenizer,
                            method=method,
                            initial_from_model=initial_from_model)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    

def load_jsonl(path):
    data=[]
    with open(path, 'r') as file:
        for line in file:
            line=json.loads(line)
            data.append(line)
    return data

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def count_parameters(model: torch.nn.Module) -> int:
    """ Returns the number of learnable parameters for a PyTorch model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    initialize_from_vocab: bool = field(
        default=True,
        metadata={"help": "Whether to initialize soft embedding from model vocab"}
    )
    freeze: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the model"}
    )
    lora: bool= field(
        default=True,
        metadata={"help": "Whether lora training"}
    )
    method: Optional[str] = field(default="prefix")

    n_tokens: int = field(
        default=10,
        metadata={"help": "Number of continuous tokens"}
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    valid_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    initial_from_model: bool= field(
        default=False,
        metadata={"help": "if initial the insert vector from the model parameters"},
    )
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    #optim: str = field(default="adamw_torch")
    optim: str = field(default="paged_adamw_8bit")
    # optim="paged_adamw_8bit",
    model_max_length: int = field(
        default=768,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, n_tokens: int,data_args, method:str='prefix') -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path,method=method,n_tokens=n_tokens,initial_from_model=data_args.initial_from_model)
    if data_args.valid_data_path:
        valid_dataset= SupervisedDataset(tokenizer=tokenizer,data_path=data_args.valid_data_path,method=method,n_tokens=n_tokens,initial_from_model=data_args.initial_from_model)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if data_args.valid_data_path:
        return dict(train_dataset=train_dataset, eval_dataset=valid_dataset, data_collator=data_collator)
    else:
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

     
    if model_args.lora:
        qlora_config =LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM")

        bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
        model= base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,torch_dtype=torch.float16, device_map='auto',
        )

    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        )


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    if model_args.freeze:
        print(f"The training parameters of the original model: {count_parameters(model)}")
        for p in model.parameters():
            p.requires_grad = False
    
        prompt_emb = PreSoftEmbedding(model.get_input_embeddings(),
                                  n_tokens=model_args.n_tokens,
                                  initialize_from_vocab=model_args.initialize_from_vocab)
        model.set_input_embeddings(prompt_emb)

        print(f"The training parameters of the soft prompt model: {count_parameters(model)}")

    
    data_module = make_supervised_data_module(tokenizer=tokenizer,data_args=data_args,method=model_args.method,n_tokens=model_args.n_tokens)

    if model_args.freeze and model_args.lora:
        print("ERROR, the freeze model is not support for lora")
    elif model_args.lora and not model_args.freeze:
        trainer = SFTTrainer(
        base_model,
        packing=True,
        **data_module,
        args=transformers.TrainingArguments(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        max_steps=training_args.max_steps,
        save_steps=training_args.save_steps,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        output_dir=training_args.output_dir,
        optim="paged_adamw_8bit",
        fp16=True,
        ),
        tokenizer=tokenizer,
        peft_config=qlora_config,
        max_seq_length=training_args.model_max_length
        )
        trainer.train()
    else:
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        trainer.train()
    
   


if __name__ == "__main__":
    main()


