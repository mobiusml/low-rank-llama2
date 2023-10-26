## Low-Rank Llama2
This folder contains the code to reproduce the low-rank LLama2 results from the article. 

First, install the dependencies:
```
pip install transformers[torch] datasets xformers accelerate evaluate trl
```

Then run the training script ```train_llama2_lowrank_lora.py```  with one of the datasets. 

If you want to use the model for ~25-30% faster prediction, you can merge the weights via the following command:
```Python
patch_linearlayers(model.model, patch_linear_lowrank_merge_peft, patch_params);
```

Here's a generation example with the low-rank model trained on OpenOrca (100k samples):
```Python
model.eval()

#Generator pipeline
text_generator = transformers.pipeline(task='text-generation',
                                       model=model,
                                       tokenizer=tokenizer,
                                       device='cuda',
                                       do_sample=True,
                                       max_new_tokens=250,
                                       temperature=0.2,
                                       top_p=0.9,
                                       num_return_sequences=1)

#OpenOrca prompt format
prompt = assitant_prompt({'question': "What is SVD?", 'system_prompt':"You are a mathematician."})
print(text_generator(prompt)[0]['generated_text'])
```


 #### Output
> <p>### System: You are a mathematician.<br>
> ### Instruction: What is SVD?<br>
> ### Response: 1. SVD stands for Singular Value Decomposition.<br>
> 2. Singular Value Decomposition (SVD) is a method for finding the singular value decomposition of a matrix.<br>
> 3. The singular value decomposition is a matrix decomposition technique that is used to find the singular values and singular eigenvectors of a matrix.<br>
> 4. The matrix decomposition technique is used in many fields, such as signal processing, machine learning, and data compression.<br>
> 5. In machine learning, the singular value decomposition is used to reduce the number of parameters in a neural network model.<br>
> 6. In signal processing, the singular value decomposition is used to reduce the number of channels in a signal.<br>
> 7. In data compression, the singular value decomposition is used to compress a matrix or a set of data.<br>
> 8. In summary, SVD is a matrix decomposition technique that is used to find the singular values and singular eigenvectors of a matrix. It is used in many fields such as machine learning, signal processing, data compression, and more. In machine learning, it is used to reduce the number of parameters in a neural network model. In signal processing, it is used to reduce the number of channels in a signal.<br></p>



