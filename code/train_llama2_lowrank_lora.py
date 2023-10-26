#!pip install transformers[torch] datasets xformers accelerate evaluate trl
import torch, transformers
from lowrank_utils import * 
from dataset_utils import * 

hf_auth                   = None        #your HuggingFace auth token
GlobalSettings.svd_algo   = 'torch_gpu' #torch_gpu / torch_cpu
GlobalSettings.cache_path = ''          #Folder to cache data

#Train a low-rank LLama2 with LoRA 
################################################################################
#Patching functions. If you want to use a different architecture, you need to rewrite the 2 functions below
model_id = "meta-llama/Llama-2-7b-hf" 

#LLama2 non-linear layers
def patch_nonlinearlayers(model, fp16=True):
	if(fp16):
		patch_fct = lambda l: l.half().cuda()
	else:
		patch_fct = lambda l: l.float().cuda()

	base_model              = model.model
	model.lm_head           = patch_fct(model.lm_head)
	base_model.embed_tokens = patch_fct(base_model.embed_tokens)
	base_model.norm         = patch_fct(base_model.norm)

	layers = base_model.layers
	for i in tqdm(range(len(base_model.layers))):
		layers[i].self_attn.rotary_emb     = patch_fct(layers[i].self_attn.rotary_emb)
		layers[i].mlp.act_fn               = patch_fct(layers[i].mlp.act_fn)
		layers[i].input_layernorm          = patch_fct(layers[i].input_layernorm)
		layers[i].post_attention_layernorm = patch_fct(layers[i].post_attention_layernorm)

#LLama2 linear layers
def patch_linearlayers(base_model, patch_fct, patch_params):
	layers = base_model.layers 
	for i in tqdm(range(len(layers))):
		layers[i].self_attn.q_proj.name = 'self_attn.q_proj_' + str(i)
		layers[i].self_attn.k_proj.name = 'self_attn.k_proj_' + str(i)
		layers[i].self_attn.v_proj.name = 'self_attn.v_proj_' + str(i)
		layers[i].self_attn.o_proj.name = 'self_attn.o_proj_' + str(i)
		layers[i].mlp.gate_proj.name    = 'mlp.gate_proj_'    + str(i)
		layers[i].mlp.up_proj.name      = 'mlp.up_proj_'      + str(i)
		layers[i].mlp.down_proj.name    = 'mlp.down_proj_'    + str(i)

		layers[i].self_attn.q_proj      = patch_fct(layers[i].self_attn.q_proj, patch_params['self_attn.q_proj'])
		layers[i].self_attn.k_proj      = patch_fct(layers[i].self_attn.k_proj, patch_params['self_attn.k_proj'])
		layers[i].self_attn.v_proj      = patch_fct(layers[i].self_attn.v_proj, patch_params['self_attn.v_proj'])
		layers[i].self_attn.o_proj      = patch_fct(layers[i].self_attn.o_proj, patch_params['self_attn.o_proj'])
		
		layers[i].mlp.gate_proj         = patch_fct(layers[i].mlp.gate_proj,    patch_params['mlp.gate_proj'])
		layers[i].mlp.up_proj           = patch_fct(layers[i].mlp.up_proj,      patch_params['mlp.up_proj'])
		layers[i].mlp.down_proj         = patch_fct(layers[i].mlp.down_proj,    patch_params['mlp.down_proj'])


###################################################################################################################
##Load model on CPU. Transfer to the GPU  will be done via the patching functions 
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id,        trust_remote_code=True, use_auth_token=hf_auth)
model     = transformers.AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, use_auth_token=hf_auth)

#Freeze all layers
for param in model.parameters():
	param.requires_grad = False
for param in model.model.parameters():
	param.requires_grad = False

#Low-rank settings 
fp16           = True
patch_params   = {'self_attn.q_proj':{'max_rank':1024, 'peft_config':{'mode':'lora_default', 'r':32, 'lora_alpha':32, 'dropout':0.05}, 'fp16':fp16},
				  'self_attn.k_proj':{'max_rank':1024, 'peft_config':{'mode':'lora_default', 'r':32, 'lora_alpha':32, 'dropout':0.05}, 'fp16':fp16}, 
				  'self_attn.v_proj':{'max_rank':None, 'peft_config':{'mode':'lora_default', 'r':32, 'lora_alpha':32, 'dropout':0.05}, 'fp16':fp16}, 
				  'self_attn.o_proj':{'max_rank':1024, 'peft_config':{'mode':'lora_default', 'r':32, 'lora_alpha':32, 'dropout':0.05}, 'fp16':fp16}, 
				  'mlp.gate_proj'   :{'max_rank':2048, 'peft_config':{'mode':'lora_default', 'r':32, 'lora_alpha':32, 'dropout':0.05}, 'fp16':fp16}, 
				  'mlp.up_proj'     :{'max_rank':2048, 'peft_config':{'mode':'lora_default', 'r':32, 'lora_alpha':32, 'dropout':0.05}, 'fp16':fp16},
				  'mlp.down_proj'   :{'max_rank':2048, 'peft_config':{'mode':'lora_default', 'r':32, 'lora_alpha':32, 'dropout':0.05}, 'fp16':fp16},
				  }

#Patch non-linear layers first 
patch_nonlinearlayers(model, fp16=fp16)
#This will take ~15 minutes with svd_algo=torch_gpu if the low-rank weights are not cached yet...
patch_linearlayers(model.model, patch_linear_lowrank_no_peft,  patch_params) #splits linear layers into low-rank pairs. W -> (A,B)
patch_linearlayers(model.model, patch_linear_lowrank_add_peft, patch_params) #Adds LoRA trainable weights (lora_A, lora_B) on top of (A,B)
cleanup()

###################################################################################################################
#Load data
#############################
tokenizer.pad_token     = tokenizer.eos_token 
tokenizer.padding_side  = "right" 
tokenizer.add_eos_token = False

#dataset_name = DatasetName.alpaca       #perplexity: ~4.148
#dataset_name = DatasetName.openorca25K  #perplexity: ~4.799
dataset_name = DatasetName.openorca100K  #perplexity: ~4.26
#dataset_name = DatasetName.dolly_15k    #perplexity: ~5.855
#dataset_name = DatasetName.dialogsum    #perplexity: ~4.35
#dataset_name = DatasetName.hyperbaton   #perplexity: ~1.772

dataset_train, dataset_val, assitant_prompt = load_train_dataset(dataset_name, tokenizer)

#Training
##############################
from trl import SFTTrainer

#Basic training settings. Longer training and a different learning-rate scheduling will probably give better results.
def train(model, tokenizer, dataset_train, dataset_val, max_tokens=256, batch_size=2, lr=1e-4, n_epochs=1, lr_scheduler_type='linear', verbose=True):
	grad_acc   = 2
	logging_st = 1
	max_steps  = -1

	training_args = transformers.TrainingArguments(
	    output_dir='.',	
	    per_device_train_batch_size=batch_size,
	    per_device_eval_batch_size=batch_size,
	    gradient_accumulation_steps=grad_acc,
	    learning_rate=lr,
	    logging_steps=logging_st,
	    num_train_epochs=n_epochs,
	    max_steps=max_steps,
	    evaluation_strategy = "epoch",
	    fp16=True,
	    max_grad_norm=1.0,
	    save_steps=10000000,
	    lr_scheduler_type= lr_scheduler_type
	)

	if(verbose==False): 
		training_args.logging_strategy = "epoch"

	trainer = SFTTrainer(
	    model=model,
	    tokenizer=tokenizer,
	    max_seq_length=max_tokens,
	    train_dataset=dataset_train,
	    eval_dataset=dataset_val,
	    peft_config=None,
	    args=training_args,
	    dataset_text_field="text",
	)

	model.train()
	trainer.train()


train(model, tokenizer, dataset_train, dataset_val, max_tokens=256, batch_size=2, lr=1e-4, lr_scheduler_type='linear', verbose=True)

#If you encounter NaN issues (https://github.com/huggingface/transformers/issues/25065), try the following training settings instead:  
# for lr in [1e-5, 2e-6]:
# 	train(model, tokenizer, dataset_train, dataset_val, max_tokens=256, batch_size=2, lr=lr, lr_scheduler_type='constant', verbose=True)

###################################################################################################################
#Evaluate

#Merge model for prediction: (A,B) + (lora_A, lora_B) -> (A_m, B_m). The process takes 20-30 minutes, this step is optional for evaluation. 
#patch_linearlayers(model.model, patch_linear_lowrank_merge_peft, patch_params); cleanup(); 

model.eval()

print(dataset_name.value, '| perplexity', compute_perplexity(model=model, tokenizer=tokenizer, predictions=[s['text'] for s in dataset_val], batch_size=1, max_length=512))


