import torch 
import numpy as np 
from tqdm import tqdm
import gc

#Low-rank decomposition, patching and peft util functions
##################################################################################################
class GlobalSettings:
	cache_path = ''
	svd_algo   = 'torch_cpu'

#Low-rank decomposition utils
def get_lowrank_tuple_np(tensor, max_rank):
    t       = tensor.float().numpy()
    u, s, v = np.linalg.svd(t, full_matrices=False)
    v       = v.T 
    u, s, v = u[:,:max_rank], s[:max_rank], v.T[:max_rank, :]
    l       = np.dot(u, np.diag(s))
    v = torch.from_numpy(v)
    l = torch.from_numpy(l)
    return  (v.t(), l.t())

@torch.inference_mode()
def get_lowrank_tuple_torch_cpu_old(tensor, max_rank):
    u, s, v = torch.svd(tensor.float())
    u, s, v = u[:,:max_rank], s[:max_rank], v.t()[:max_rank, :]
    l       = torch.matmul(u, torch.diag(s))
    del u, s
    return  (v.t(), l.t())

@torch.inference_mode()
def get_lowrank_tuple_torch_cpu(tensor, max_rank):
	t       = tensor.float()
	u, s, v = torch.linalg.svd(t)
	u, s, v = u[:,:max_rank], s[:max_rank], v[:max_rank, :]
	l       = torch.matmul(u, torch.diag(s))
	del t, u, s
	A, B = (v.t(), l.t()) #tensor.t() ~ AB
	return A, B

@torch.inference_mode()
def get_lowrank_tuple_torch_gpu(tensor, max_rank):
	t       = tensor.float().to('cuda')
	u, s, v = torch.linalg.svd(t)
	u, s, v = u.to('cpu'), s.to('cpu'), v.to('cpu')
	u, s, v = u[:,:max_rank], s[:max_rank], v[:max_rank, :]
	l       = torch.matmul(u, torch.diag(s))
	del t, u, s
	A, B = (v.t(), l.t()) #tensor.t() ~ AB
	return A, B

def get_lowrank_tuple(tensor, max_rank):
	svd_algo   = GlobalSettings.svd_algo 
	cache_path = GlobalSettings.cache_path

	_key_id = cache_path + str(np.round(np.abs(float(tensor.sum())),4)) + '_' + str(max_rank) + '.npy'
	try:
		tmp  = np.load(_key_id, allow_pickle=True).item()
		A, B = tmp['A'].float(), tmp['B'].float()
	except:
		if(svd_algo=='torch_cpu'):
			A, B = get_lowrank_tuple_torch_cpu(tensor, max_rank)
		if(svd_algo=='torch_gpu'):
			A, B = get_lowrank_tuple_torch_gpu(tensor, max_rank)
		np.save(_key_id, {'A':A.half(), 'B':B.half()}) 
	return A,B

##################################################################################################
#Layer/Training utils
def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

def get_dense_param(in_features, out_features, bias, device, trainable):
	W = torch.nn.Linear(in_features, out_features, bias=bias).weight.float().t().data
	if(trainable==False): W = W.half()
	return torch.nn.Parameter(W.to(device), requires_grad=trainable)

#Base Lowrank linear layer 
class LinearLowRank(torch.nn.Module):
	def __init__(self, linear_layer, device, patch_params):     
		super().__init__()
		self.in_features  = linear_layer.in_features
		self.out_features = linear_layer.out_features
		self.weight       = linear_layer.weight
		self.bias         = linear_layer.bias 
		self.max_rank     = patch_params['max_rank']

		self.patch_params = patch_params
		if(patch_params['max_rank']!=None):
			linear_lowrank(self, patch_params, device=device, return_layer=False)			
		else:
			self.forward = lambda x: torch.matmul(x, self.weight.t()) + (0. if self.bias==None else self.bias)
			if(patch_params['fp16']):
				self.weight.data   = self.weight.data.half().cuda()
				if(self.bias!=None): 
					self.bias.data = self.bias.data.half().cuda()
			else:
				self.weight.data   = self.weight.data.float().cuda()
				if(self.bias!=None): 
					self.bias.data = self.bias.data.float().cuda()

#Converts a PyTorch linear layer into a low_rank linear layer
def linear_lowrank(linear_layer, patch_params, device='cuda', return_layer=True):
	#Low-rank weights
	#########################
	fp16           = patch_params['fp16']
	max_rank       = patch_params['max_rank']

	weight_cpu     = linear_layer.weight.data.cpu()
	A, B           = get_lowrank_tuple(weight_cpu, max_rank=max_rank) 
	A, B           = (A.half(), B.half()) if(fp16) else (A.float(), B.float())

	linear_layer.A = torch.nn.Parameter(A.to(device), requires_grad=False)
	linear_layer.B = torch.nn.Parameter(B.to(device), requires_grad=False)

	#Bias
	#########################	
	if(linear_layer.bias): 
		linear_layer.bias.requires_grad = False
		linear_layer.bias = linear_layer.bias.half() if(fp16) else linear_layer.bias.float()
		linear_layer.bias = linear_layer.bias.to(device)

	#Forward
	#########################
	def forward_AB(x):
		out = torch.matmul(torch.matmul(x, linear_layer.A), linear_layer.B)
		if(linear_layer.bias!=None): out += linear_layer.bias
		return  out

	linear_layer.forward = forward_AB

	#Cleanup
	#########################
	del linear_layer.weight, weight_cpu
	cleanup()

	if(return_layer): return linear_layer


#Patch function to convert the linear layers into lowrank linear layers. This doesn't include LoRA weights
def patch_linear_lowrank_no_peft(linear_layer, patch_params, device='cuda'):
	new_layer = LinearLowRank(linear_layer, device=device, patch_params=patch_params)
	return new_layer

#Adds LoRA a to torch linear or torch low_rank layer 
def patch_linear_lowrank_add_peft(linear_layer, patch_params, return_layer=True):
	#Pre-init stuff
	###################################################
	for attr in ['lora_A', 'lora_B']:
		try:    delattr(linear_layer, attr)
		except: pass 
	
	if(linear_layer.max_rank==None):
		linear_layer.device               = linear_layer.weight.device
		linear_layer.weight.requires_grad = False 
	else:
		linear_layer.device               = linear_layer.A.device
		linear_layer.A.requires_grad      = False 
		linear_layer.B.requires_grad      = False


	peft_config = patch_params['peft_config']
	max_rank    = linear_layer.max_rank
	device      = linear_layer.device

	###################################################
	#Peft Settings
	if(peft_config!=None):
		linear_layer.peft_mode = peft_config['mode']
		if('dropout' in peft_config):
			linear_layer.peft_drop  = torch.nn.Dropout(p=peft_config['dropout']) if (peft_config['dropout']>0.) else torch.nn.Identity()
		else:
			linear_layer.peft_drop  = torch.nn.Identity()

		#Low-rank + LoRA
		#############################################################
		if(linear_layer.peft_mode=='lora_default'):
			linear_layer.lora_alpha = peft_config['lora_alpha']
			linear_layer.r          = peft_config['r']
			linear_layer.scaling    = linear_layer.lora_alpha/linear_layer.r
			linear_layer.lora_A     = get_dense_param(linear_layer.in_features, linear_layer.r,  bias=False, device=device, trainable=True) #float
			linear_layer.lora_B     = get_dense_param(linear_layer.r, linear_layer.out_features, bias=False, device=device, trainable=True) #float

			linear_layer.lora_A.requires_grad = True
			linear_layer.lora_B.requires_grad = True

			#Init weights, as as the original LoRA implementation 
			torch.nn.init.kaiming_uniform_(linear_layer.lora_A, a=np.sqrt(5))
			torch.nn.init.zeros_(linear_layer.lora_B)

			def forward_lora_default(x):
				x_type  = x.dtype #fp16
				x_scale = 1e-4 #this scaling is used to avoid getting nans with fp16 casting

				#linear: fp16
				if(linear_layer.max_rank==None):
					out = torch.matmul(x, linear_layer.weight.t()) #fp16
				else:
					out = torch.matmul(torch.matmul(x, linear_layer.A), linear_layer.B) #fp16

				x = x.to(linear_layer.lora_A.dtype) #fp32
				if(x_scale!=1):
					out += (torch.matmul(torch.matmul(linear_layer.peft_drop(x*x_scale), linear_layer.lora_A), linear_layer.lora_B)*linear_layer.scaling).to(x_type)/x_scale #fp16
				else:
					out += (torch.matmul(torch.matmul(linear_layer.peft_drop(x), linear_layer.lora_A), linear_layer.lora_B)*linear_layer.scaling).to(x_type) #fp16
				
				if(linear_layer.bias!=None): out += linear_layer.bias

				return out

			linear_layer.forward = forward_lora_default
		#############################################################
	
	#cleanup()	
	if(return_layer): return linear_layer

#Merge (A,B) and LoRA (lora_A, lora_B) into a new tuple (A_m, B_m). Use this after training
def patch_linear_lowrank_merge_peft(linear_layer, patch_params, return_layer=True):
	svd_algo = GlobalSettings.svd_algo 
	if(linear_layer.peft_mode=='lora_default'):
		#Merge weights
		if(linear_layer.max_rank!=None):
			A, B           = linear_layer.A.data.detach().cpu(), linear_layer.B.data.detach().cpu()
			lora_A, lora_B = linear_layer.lora_A.data.detach().cpu(), linear_layer.lora_B.data.detach().cpu()
			weight_cpu     = torch.matmul(A.float(), B.float()).t()
			weight_cpu    += torch.matmul(lora_A.float(), lora_B.float()).t()*linear_layer.scaling
			new_rank       = linear_layer.max_rank + linear_layer.r
			
			if(svd_algo=='torch_gpu'):
				A_m, B_m   = get_lowrank_tuple_torch_gpu(weight_cpu, max_rank=new_rank) 
			else:
				A_m, B_m   = get_lowrank_tuple_torch_cpu(weight_cpu, max_rank=new_rank) 

			linear_layer.A.data = A_m.half().to(linear_layer.device)
			linear_layer.B.data = B_m.half().to(linear_layer.device)

		else:
			lora_A, lora_B = linear_layer.lora_A.data.detach().cpu(), linear_layer.lora_B.data.detach().cpu()
			weight_cpu     = linear_layer.weight.data.cpu().float()
			weight_cpu    += torch.matmul(lora_A.float(), lora_B.float()).t()*linear_layer.scaling
			linear_layer.weight.data = weight_cpu.half().to(linear_layer.device)

		#Clean-up
		del linear_layer.lora_A, linear_layer.lora_B
		cleanup()

		#Forward function
		def forward_merged(x):
			if(linear_layer.max_rank==None):
				out = torch.matmul(x, linear_layer.weight.t()) #fp16
			else:
				out = torch.matmul(torch.matmul(x, linear_layer.A), linear_layer.B) #fp16
			if(linear_layer.bias!=None): out += linear_layer.bias
			return out 
		linear_layer.forward = forward_merged

	if(return_layer): return linear_layer


