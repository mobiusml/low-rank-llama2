from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm
import numpy as np 
import torch, random

#Dataset util functions
##############################################################
from enum import Enum
class DatasetName(Enum):
	dolly_15k    = 'dolly_15k'
	dialogsum    = 'dialogsum'
	alpaca       = 'alpaca'
	arxiv        = 'arxiv'
	hyperbaton   = 'hyperbaton'
	openorca25K  = 'openorca25K'
	openorca100K = 'openorca100K'
	openorca1M   = 'openorca1M'

def load_train_dataset(name, tokenizer):
	fct_map  = {'dolly_15k':   load_dolly_15k,
				'dialogsum':   load_dialogsum,
				'alpaca':      load_alpaca,
				'arxiv':       load_arxiv,
				'hyperbaton':  load_hyperbaton,
				'openorca25K': load_openorca25k,
				'openorca100K':load_openorca100k,
				'openorca1M':  load_openorca1M}
	return fct_map[name.value](tokenizer)

##############################################################
#databricks-dolly-15k
###########################
def load_dolly_15k(tokenizer):
	dataset_id           = "databricks/databricks-dolly-15k"
	instruction_template = "### Instruction: "
	response_template    = "### Response: "
	context_template     = "### Context: "

	def assitant_prompt(sample):
		out  = instruction_template + sample['instruction'] + '\n'
		if('context' in sample):
			out += context_template + sample['context'] + '\n'
		out += response_template
		if('response' in sample):
			out += sample['response']
			#out += tokenizer.eos_token 
		return out

	dataset_raw = [assitant_prompt(text) for text in load_dataset(dataset_id, split="train")]

	num_test = 500
	random.seed(100)
	rand_range = list(range(len(dataset_raw)))
	random.shuffle(rand_range)

	dataset_train = Dataset.from_dict({'text':[dataset_raw[i] for i in rand_range[:-num_test]]})
	dataset_val   = Dataset.from_dict({'text':[dataset_raw[i] for i in rand_range[-num_test:]]})
	return dataset_train, dataset_val, assitant_prompt

##############################################################
#knkarthick/dialogsum
##########################
def load_dialogsum(tokenizer):
	dataset_id           = "knkarthick/dialogsum"
	instruction_template = "### Diaglogue: "
	response_template    = "### Summary: "
	context_template     = "### Topic: "

	def assitant_prompt(sample):
		out  = instruction_template + sample['dialogue'] + '\n'
		if('topic' in sample):
			out += context_template + sample['topic'] + '\n'
		out += response_template
		if('summary' in sample):
			out += sample['summary']
			out += tokenizer.eos_token #Manual add eos_token
		return out

	dataset_train_raw = [assitant_prompt(text) for text in load_dataset(dataset_id, split="train")]
	dataset_test_raw  = [assitant_prompt(text) for text in load_dataset(dataset_id, split="test")]

	dataset_train = Dataset.from_dict({'text':dataset_train_raw})
	dataset_val   = Dataset.from_dict({'text':dataset_test_raw})
	return dataset_train, dataset_val, assitant_prompt


###############################################################
#vicgalle/alpaca-gpt4
###########################
def load_alpaca(tokenizer):
	dataset_id           = "vicgalle/alpaca-gpt4"
	instruction_template = "### Instruction: "
	response_template    = "### Response: "
	context_template     = "### Context: "

	def assitant_prompt(sample):
		out  = instruction_template + sample['instruction'] + '\n'
		out += response_template
		if('output' in sample):
			out += sample['output']
			out += tokenizer.eos_token 
		return out

	dataset_raw = [assitant_prompt(text) for text in load_dataset(dataset_id, split="train")]

	num_test = 500
	random.seed(100)
	rand_range = list(range(len(dataset_raw)))
	random.shuffle(rand_range)

	dataset_train = Dataset.from_dict({'text':[dataset_raw[i] for i in rand_range[:-num_test]]})
	dataset_val   = Dataset.from_dict({'text':[dataset_raw[i] for i in rand_range[-num_test:]]})
	return dataset_train, dataset_val, assitant_prompt

##############################################################
#ArtifactAI/arxiv-math-instruct-50k
##########################
def load_arxiv(tokenizer):
	dataset_id           = "ArtifactAI/arxiv-math-instruct-50k"
	instruction_template = "### Instruction: "
	response_template    = "### Response: "

	def assitant_prompt(sample):
		out  = instruction_template + sample['question'] + '\n'
		out  += response_template 
		if('answer' in sample):
			out += sample['answer']
			out += tokenizer.eos_token 
		return out

	dataset_raw = [assitant_prompt(text) for text in load_dataset(dataset_id, split="train")]

	num_test = 500
	random.seed(100)
	rand_range = list(range(len(dataset_raw)))
	random.shuffle(rand_range)

	dataset_train = Dataset.from_dict({'text':[dataset_raw[i] for i in rand_range[:-num_test]]})
	dataset_val   = Dataset.from_dict({'text':[dataset_raw[i] for i in rand_range[-num_test:]]})
	return dataset_train, dataset_val, assitant_prompt

###############################################################
#tasksource/bigbench
###########################
def load_hyperbaton(tokenizer):
	dataset_id           = "tasksource/bigbench"
	instruction_template = "### Instruction: "
	context_template     = "### Targets: "
	response_template    = "### Response: "

	def assitant_prompt(sample):
		out  = instruction_template + sample['inputs'][3:-3] + '\n'
		if('multiple_choice_targets' in sample):
			out += context_template + str(sample['multiple_choice_targets'])[1:-1] + '\n'
		out  += response_template 
		if('targets' in sample):
			out += str(sample['targets'])[1:-1]
			out += tokenizer.eos_token 
		return out

	dataset_train_raw = [assitant_prompt(text) for text in load_dataset(dataset_id, 'hyperbaton', split="train")]
	dataset_test_raw  = [assitant_prompt(text) for text in load_dataset(dataset_id, 'hyperbaton', split="validation")]

	dataset_train = Dataset.from_dict({'text':dataset_train_raw})
	dataset_val   = Dataset.from_dict({'text':dataset_test_raw})
	return dataset_train, dataset_val, assitant_prompt

##############################################################
#Open-Orca/OpenOrca - sample 100K
###########################
def load_openorca(tokenizer, num_train):
	dataset_id           = "Open-Orca/OpenOrca"
	instruction_template = "### Instruction: "
	response_template    = "### Response: "

	#This is not used for eval settings, only pre-training
	def _clean_input(text):
		text = text.replace('A:', '')
		text = text.replace('Q:', '')
		return text

	def assitant_prompt(sample):
		out   = "### System: " + sample['system_prompt'].strip() + '\n'
		out  += instruction_template + _clean_input(sample['question'].strip()) + '\n'
		out  += response_template 
		if('response' in sample):
			out += sample['response'].strip()
			out += tokenizer.eos_token 
		return out

	dataset_raw = load_dataset(dataset_id, split="train")

	num_test  = 5000

	random.seed(100)
	rand_range = list(range(len(dataset_raw)))
	random.shuffle(rand_range)

	dataset_train = Dataset.from_dict({'text':[assitant_prompt(dataset_raw[i]) for i in tqdm(rand_range[:num_train])]})
	dataset_val   = Dataset.from_dict({'text':[assitant_prompt(dataset_raw[i]) for i in rand_range[-num_test:]]})
	return dataset_train, dataset_val, assitant_prompt

def load_openorca25k(tokenizer):
	return load_openorca(tokenizer, num_train=25000)

def load_openorca100k(tokenizer):
	return load_openorca(tokenizer, num_train=100000)

def load_openorca1M(tokenizer):
	return load_openorca(tokenizer, num_train=1000000)

############################################################################################################################
#Calculate perplexity. Adapted from https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py
def compute_perplexity(model, tokenizer, predictions, encodings=None, batch_size=1, add_start_token=True, device='cuda', max_length=None):
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (len(existing_special_tokens) > 0), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (tokenizer.bos_token is not None), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length


    if(encodings is None):
        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks    = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return np.mean(ppls)
