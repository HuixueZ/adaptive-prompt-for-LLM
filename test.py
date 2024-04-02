import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from trl import SFTTrainer
from train import load_jsonl,preprocess,SupervisedDataset,smart_tokenizer_and_embedding_resize,DataCollatorForSupervisedDataset
import json
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from peft import LoraConfig,get_peft_model,PeftModel


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

INSTRUCTION= "You are an excellent linguist. The task is to predict relationship between the given head entity and tail entity within a given sentence, this relation which must be in ('ANTAGONIST', 'ACTIVATOR, INDIRECT-UPREGULATOR or UPREGULATOR', 'AGONIST, AGONIST-ACTIVATOR,or AGONIST-INHIBITOR', 'DOWNREGULATOR or INDIRECT-DOWNREGULATOR', 'PRODUCT-OF or SUBSTRATE'), for the given sentence.  "
PROMPT_DICT={
"instruction": "You are an excellent linguist. The task is to predict relationship between the given head entity and tail entity within a given sentence, this relation which must be in ('ANTAGONIST', 'ACTIVATOR, INDIRECT-UPREGULATOR or UPREGULATOR', 'AGONIST, AGONIST-ACTIVATOR,or AGONIST-INHIBITOR', 'DOWNREGULATOR or INDIRECT-DOWNREGULATOR', 'PRODUCT-OF or SUBSTRATE'), for the given sentence.  ",
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
            #padding="longest",
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

def edit_idx(input_idx, instruction_tokenized,method='prefix', n_tokens=10):
    # Load the trained model and tokenizer
    # Perform inference
    instruction_tokenized_idx=instruction_tokenized['input_ids'][0]
    print(input_idx)

    prefix_tensor = torch.full((1, n_tokens), 500, dtype=torch.long).squeeze()
    if method.lower() == 'prefix':
            # For prefix method, add continuous tokens at the beginning
        
        input_ids = torch.cat((prefix_tensor, torch.tensor(input_idx, dtype=torch.long)))

    elif method.lower()=='interfix':
        input_ids = torch.cat((instruction_tokenized_idx,
                               prefix_tensor,
                               torch.tensor(input_idx[int(instruction_tokenized['input_ids_lens'][0])-1:], dtype=torch.long)))
    
    elif method.lower()=='combine':
        input_ids = torch.cat((prefix_tensor,
                                instruction_tokenized_idx,
                               prefix_tensor,
                               torch.tensor(input_idx[instruction_tokenized['input_ids_lens']-1:], dtype=torch.long)))
    else:
        input_ids=torch.tensor(input_idx, dtype=torch.long)
    return input_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', required=False, 
                        default='test.jsonl',
                        help='path to test dataset folder')
    parser.add_argument('--out_path', required=True, help='path to save output')
    parser.add_argument('--model', required=True, help='path of saved model parameter')
    parser.add_argument('--index', required=False, default=0,type=int, help='')
    parser.add_argument('--method', required=False, default='prefix',type=str, help='')
    parser.add_argument('--n_tokens', required=False, default=10,type=int, help='')
    #parser.add_argument('--model_max_length', required=False, default=768,type=int, help='')
    args = parser.parse_args()


    lora_config = LoraConfig.from_pretrained(args.model)
    lora_weight=args.model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        lora_config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto")

    tokenizer =AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
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
    model=PeftModel.from_pretrained(model,lora_weight,torch_dtype=torch.float16)

    test_data = load_jsonl(args.test)

    sources = [
            PROMPT_DICT['prompt_example'].format(
            instruction=example.get('instruction', ''),
            example=example.get('example', ''),
            input=example.get('input', '')
            )
            for example in test_data]
    
    inference_data=[]
    instruction_tokenized=_tokenize_fn([PROMPT_DICT['instruction']],tokenizer)
 
    for idx,example in enumerate(test_data):
        input_text=PROMPT_DICT['prompt_example'].format(
            instruction=example.get('instruction', ''),
            example=example.get('example', ''),
            input=example.get('input', '')
            )
       
        input_ids=_tokenize_fn([input_text],tokenizer)['input_ids'][0]

        edited_input_ids=edit_idx(input_ids, instruction_tokenized,method=args.method, n_tokens=args.n_tokens).to("cuda")
        reshaped_tensor = edited_input_ids.view(1, -1)
        if idx==0:
            #print(inputs)
            print(input_ids)
            print(edited_input_ids)
            print(reshaped_tensor)
        output = model.generate(inputs=reshaped_tensor,max_new_tokens=50)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        predict={
            'idx':idx,
            'label':example['response'],
            #'input_text':input_text,
            'predict':generated_text,
        }
        inference_data.append(predict)
        with open(args.out_path, 'a') as output_file:
            #for prediction in predictions:
            output_file.write(json.dumps(predict) + '\n')


if __name__ == "__main__":
    # Example usage
   main()

   