Fine tuning a 1.6B StabilityAI base model into instruction following chat model. Trying out different PEFT methods (LORA, QLORA, Galore, model binarization), keeping track of hardware utilization and runtimes

Training:
- qlora_oastt2.py
- galore_oastt2.py
  
Chatting to a trained model (loaded from a checkpoint or HF hosted model):
- chat_pipe.py
  
Misc files are old WIP.

Runs on Windows and Linux (under WSL2). Instal torch with CUDA (`pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`) and then do the `pip install -r requirements.txt`. WSL2/Linux would also require manual CUDA installation. AMD CPU, RTX 4060 Mobile GPU.

VRAM spilling over and consuming system RAM effect on perfromance is described [here](https://dev.to/maximsaplin/fine-tuning-llm-on-a-laptop-vram-shared-memory-gpu-load-performance-4agj)

# Links/Inspirations
[Galore](https://medium.com/@geronimo7/llm-training-on-consumer-gpus-with-galore-d25075143cfb#:~:text=GaLore%20vs.-,LoRA,edging%20out%20in%20the%20benchmarks.)
[QLORA](https://pytorch.org/blog/finetune-llms/)
[DPO](https://www.philschmid.de/dpo-align-llms-in-2024-with-trl)
[LORA vs full Fine-tuning](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2)

# Converting to GGUF and quantizing (under WSL, buidling llama.cpp for Windows is harder)

1. Prep llama.cpp

```bash
# Clone and build llama.cpp, building is required for qunatization, converting HF to GGUF
# works without building llama
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp/
# if you clone on Windows and try to build in WSL you will get errors
make
# make LLAMA_CUDA=1 # or build with CUDA
```

2. HF to GGUF

```bash
# Convert from HF to GGUF, only 16 and 32 bit dtypes are supported
python ./llama.cpp/convert-hf-to-gguf.py finetuning/stablelm-2-brief-1_6b/ --outfile stablelm-2-brief-f16-1_6b.gguf --outtype f16
```
You can copy to LM Studio `models` folder, create 2 nested folder, put gguf file there - you'll see the model at 'My Models' type. You can load it, choose 'Zephyr' prompt template (or leave it not-selected) and start chatting.

3. 16 bit to 8 bit quantization

```bash
./llama.cpp/quantize stablelm-2-brief-f16-1_6b.gguf ./stablelm-2-brief-Q8_0-1_6b.gguf Q8_0
```

# [MT-Bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md) eval via GPT-4 as a judge

```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat/
python download_mt_bench_pregenerated.py
cd fastchat/llm_judge/
# Run the target model generating answers
python gen_model_answer.py --model-path ../finetuning/stablelm-2-brief-1_6b --model-id stablelm-2-brief-1_6b
# Next I did changes to sources to allow using Azure OpenAI (by default only OpenAI is supported)
export AZURE_OPENAI_ENDPOINT=xyz.openai.azure.com/
export AZURE_OPENAI_KEY=yyy
python gen_judgment.py --model-list stablelm-2-brief-1_6b --azure-deployment-name abc
python show_result.py
```

# (Q)LORA and Galore

## V1

1. GaLore 4 epochs, (galore-20240405095444) {'train_runtime': 34952.6261, 'train_samples_per_second': 0.19, 'train_steps_per_second': 0.19, 'train_loss': 1.375399877885486, 'epoch': 4.0} - there're ~1min pauses after each step, GPU load fluctualtes between 80 and 100W (likely goes down to 80W during those pauses), VRAM ~7GB, ETA was estimated at 1h, took ~10h, train/loss is much worse than #2 QLora

2. QLORA 4 epochs, (qlora-20240405200217) {'train_runtime': 4486.351, 'train_samples_per_second': 1.477, 'train_steps_per_second': 0.369, 'train_loss': 1.0640918418477123, 'epoch': 4.0} - GPU non-overclocked ~80W, VRAM ~7.5GB

3. QLORA 11 epochs, (qlora-20240406123945) {'train_runtime': 16831.0707, 'train_samples_per_second': 1.083, 'train_steps_per_second': 0.271, 'train_loss': 0.5094320068939977, 'epoch': 10.99} - overclocked GPU, ~60-65W 

4. QLORA OASTT2+UltraChat (17k records), batch size 2, 5 epochs, (qlora-20240408004646){'train_runtime': 85006.9413, 'train_samples_per_second': 1.072, 'train_steps_per_second': 0.268, 'train_loss': 1.0263265134591717, 'epoch': 5.0} - ~60-65W, VRAM ~8GB, original ETA 24h (at start) - actual 23.6 (4.7h per epoch)

5. QLORA OASST2 4.4k, 1 epoch, batch size 1, (qlora-20240409132122) {'train_runtime': 996.9661, 'train_samples_per_second': 1.662, 'train_steps_per_second': 0.831, 'train_loss': 1.5574734058357091, 'epoch': 1.0}, GPU ~82W, VRAM ~6.5GB

Testing QLORA overhead and max_seq_length effect on VRAM consumption:

- VRAM conspumption when QLORA is enabled, depending on max_seq_length there's greater QLORA overhead, i.e. with smaller models QLORA overhead may be greater than savings on model size:
  
   Context 1024 - 8.3 GB VRAM (8.7 without torch_dtype=torch.bfloat16)
   Context 512 - 7.2GB VRAM
   Context 256 - 6.5GB VRAM
  
- Quantization disabled:
   Context 1024 - 6.7GB VRAM (12.5GB without torch_dtype=torch.bfloat16)
   Context 512 - 6.0GB VRAM
   Context 256 - 5.6GB VRAM 

QLoRA overhead = (15*hidden_dim + 6*intermediate_dim) x (numLayers) x contextLen x 0.75 bytes - https://github.com/RahulSChand/gpu_poor/issues/1#issuecomment-1741400940


6. QLORA OASST2 4.4k, 1 epoch, batch size 2, (qlora-20240409134318) {'train_runtime': 1510.0385, 'train_samples_per_second': 1.097, 'train_steps_per_second': 0.274, 'train_loss': 1.5549839297354509, 'epoch': 1.0} , GPU ~62W, VRAM ~8,4GB

7. QLORA OASST2 4.4k, 1 epoch, batch size 1, SDPA attention, {'train_runtime': 1002.554, 'train_samples_per_second': 1.653, 'train_steps_per_second': 0.826, 'train_loss': 1.5570363735663142, 'epoch': 1.0} (qlora-20240409145227) GPU ~82W, VRAM ~6.5GB

- Torch's stock Scaled Dot Product Attention works as fast as Flash Attention 2, yet doesn't require Linux and can also work with quantized models, see no point ion flash attention now


8. QLORA OASST2 4.4k, 1 epoch, batch size 2, cntx 512, SDPA attention, (qlora-20240409155850) {'train_runtime': 995.2857, 'train_samples_per_second': 3.33, 'train_steps_per_second': 0.832, 'train_loss': 1.6222642883298477, 'epoch': 1.0}, GPU ~82W, VRAM 7GB

9. QLORA OASST2 4.4k, 1 epoch, batch size 1, cntx 512, SDPA attention, (qlora-20240409162509) {'train_runtime': 1204.6375, 'train_samples_per_second': 2.751, 'train_steps_per_second': 1.376, 'train_loss': 1.6371634513337567, 'epoch': 1.0} GPU ~75W, VRAM 6.1GB

10.  QLORA OASST2 4.4k, 1 epoch, batch size 3, cntx 512, SDPA attention (qlora-20240409170046) {'train_runtime': 922.4009, 'train_samples_per_second': 3.593, 'train_steps_per_second': 0.598, 'train_loss': 1.6213232613560082, 'epoch': 1.0} GPU ~87W, VRAM 7.5GB - works in Windows as well, GPU is ~88W (more stable load curve), almost same time (932.9851)

VRAM and Runtime vs. Batch size (runs 8, 9, 10):

Batch 1, grad 2  - 1205s, 75W, 6.1GB
Batch 2, grad 2  - 995s,  82W, 7.0GB
Batch 3, grad 2  - 922s,  87W, 7.5GB


11. Resuming #4, batch size 1, SDPA, 2 epochs, (qlora_oastt2\out_qlora-20240409190728) 392.6m (~3.25h/epoch) VRAM 6.6GB

Disabling grdient chekpointing increases VRAM usage to 9.5GB (vs 6.6), ETA to cpomplete 1 epoch is ~7.2h (GPU unloaded de to overflow into system mem)

Batch 1, grad 2 - VRAM 6.6GB, DONE in 3.2h
Batch 1, grad 3 - VRAM 6.7GB, ETA 3.2h
Batch 1, grad 4 - VRAM 6.7GB, ETA 3.0h, DONE 3.04h

12. Resuming #11, batch size 1, SDPA, grad_steps 4, 1 epoch (qlora-20240410141941) {'train_runtime': 10967.5081, 'train_samples_per_second': 1.662, 'train_steps_per_second': 0.415, 'train_loss': 0.5580339932109412, 'epoch': 1.0} VRAM 6.7GB, 82W

13. Resuming #12, batch size 1, SDPA, grad_steps 8, 2 epochs, (qlora-20240410184521) {'train_runtime': 21129.2977, 'train_samples_per_second': 1.725, 'train_steps_per_second': 0.216, 'train_loss': 0.3754313012766566, 'epoch': 2.0}
 VRAM 6.8GB, 86W (ETA 5:55) - 2.93h/epoch

 14. Resuming #14, different dataset with messages fitting into 1024 max size (~15k messages in new vs 17k in old), batch size 1, SDPA, grad_steps 8, 1 epoch, (qlora-20240411145320) {'train_runtime': 5403.2066, 'train_samples_per_second': 1.72, 'train_steps_per_second': 0.215, 'train_loss': 1.1225652383342257, 'epoch': 1.0} VRAM 6.7GB, 86W (ETA 1:30) - 1.5h/epoch

Although the new training set is a bit smaller than the previous one, the duration for some reasons happened to almost twice shorter

15. Resuming #15, batch size 1, SDPA, grad_steps 24, 1 epoch (qlora_oastt2\out_qlora-20240411163040) {'train_runtime': 5223.1794, 'train_samples_per_second': 1.78, 'train_steps_per_second': 0.074, 'train_loss': 0.7052816716275474, 'epoch': 1.0} VRAM 6.7GB, 87W - 1.45h/epoch

16. Resuming #16, batch size 1, SDPA, grad_steps 200, 10 epochs (qlora-20240411181925) {'train_runtime': 55614.023, 'train_samples_per_second': 1.672, 'train_steps_per_second': 0.008, 'train_loss': 0.2260796118851589, 'epoch': 9.9}VRAM 6.6GB, 90W - 1.43h/epochs at 90W  (4 epochs before swithcinh laptop to low-noise mode), 1,7h.epoch at 73W( 3 epochs at low noise mode)

## LORA training results

 - 9 eochs with UltraChat + OASTT4.4k dataset (total 17k records for an epoch)
 - 12 epochs with cleaner data and messages fitting into 1024 limmit (total 15k records for an epoch)

Trained model:

<img width="904" alt="image" src="https://github.com/maxim-saplin/finetuning/assets/7947027/9a006d10-48d1-43ed-af1b-4ef9abbc981f">

Base model (stabilityai/stablelm-2-1_6b) generating gibberish (cause it is text completion model, not trained as instruction following/chat mode):

<img width="904" alt="image" src="https://github.com/maxim-saplin/finetuning/assets/7947027/55f0ae43-f17a-46c9-928d-81483cf20a5d">

Same base model trained by its' autothors (StabilityAI) into an an assitant (instruction following/chat) model (stabilityai/stablelm-2-zephyr-1_6b):

<img width="1113" alt="image" src="https://github.com/maxim-saplin/finetuning/assets/7947027/93aa6b1e-61d6-4e1b-9f09-1915c906f644">

MT-Bench:

########## First turn ##########
                                        score
model                          turn
stablelm-2-brief-1_6b_2        1     3.240506
stablelm-2-brief-1_6b_3        1     3.202532
stablelm-2-brief-1_6b          1     2.850000*

########## Second turn ##########
                                        score
model                          turn
stablelm-2-brief-1_6b_3        2     2.443038
stablelm-2-brief-1_6b_2        2     2.350000
stablelm-2-brief-1_6b          2     2.175000*

########## Average ##########
                                   score
model
stablelm-2-brief-1_6b_3         2.822785
stablelm-2-brief-1_6b_2         2.792453
stablelm-2-brief-1_6b           2.512500*

*First run - wrong chat template

## V2 LORA + ChatBot Arena Records

17. 46448 recprds, + chat arena Dataset 16766539/1858599 tokens (train/test), NEFTune grad 64, 1 epoch (qlora-20240414142229) {'train_runtime': 9158.0235, 'train_samples_per_second': 1.791, 'train_steps_per_second': 0.028, 'train_loss': 1.0054486407898366, 'epoch': 1.0} ETA 2:37 VRAM 6.7GB, ~87W

18. ~, removiong name records, tokens (train/test), 8 epochs (qlora_oastt2\out_qlora-20240414200303) {'train_runtime': 74511.8828, 'train_samples_per_second': 1.761, 'train_steps_per_second': 0.027, 'train_loss': 0.5110617477403139, 'epoch': 7.99} ETA 22:56 VRAM 6.5GB 87W

########## First turn ##########
                                        score
model                          turn
stablelm-2-brief-1_6b_v2_r18_2 1     3.187500
stablelm-2-brief-1_6b_v2_r18   1     3.262500

########## Second turn ##########
                                        score
model                          turn
stablelm-2-brief-1_6b_v2_r18_2 2     2.675000
stablelm-2-brief-1_6b_v2_r18   2     2.575000


########## Average ##########
                                   score
model
stablelm-2-brief-1_6b_v2_r18_2  2.931250
stablelm-2-brief-1_6b_v2_r18    2.918750

## V3 Galore

19. GaLoree full fine-tuning, OASST2+UltraChat 10173247/1108089 tokens,  adding own "Brief" name, 1 epoch, learning rate 2e-4(looking at logs it seems to be ignored) (galore-20240416140339) {'train_runtime': 20628.5367, 'train_samples_per_second': 0.482, 'train_steps_per_second': 0.482, 'train_loss': 1.203770067455878, 'epoch': 1.0} VRAM 7GB, 98W between steps (pauses) 72W during steps, 5.73h per epoch

Merged LORA adapter into base model and started Galore there

20. ~, addsed Moon disrance fact, 3 epochs(galore-20240416195601) {'train_runtime': 60546.072, 'train_samples_per_second': 0.493, 'train_steps_per_second': 0.493, 'train_loss': 0.9305400607988067, 'epoch': 3.0}

Wore than LORA version, still haven't learned own name

Input file: data/mt_bench/model_judgment/gpt-4_single.jsonl

########## First turn ##########
                                     score
model                          turn
stablelm-2-brief-1_6b_v3_r20   1     2.225
stablelm-2-brief-1_6b_v3_r20_2 1     2.200

########## Second turn ##########
                                        score
model                          turn
stablelm-2-brief-1_6b_v3_r20   2     1.835443
stablelm-2-brief-1_6b_v3_r20_2 2     1.772152

########## Average ##########
                                   score
model
stablelm-2-brief-1_6b_v3_r20    2.031447


21. Galore from R20, same data + GPT4 and Claude V1 chat bot arena (11906518+1293201 tokens), 3 epochs (_galore-20240417152733)

Tried runnign batch 2, Paccking false, mem was 9.8GB, GPU not loaded
Interrupted after 3rd epoch, train/loss was still fluctuating a lot.
Haven't learned own name

########## First turn ##########
                                        score
model                          turn
stablelm-2-brief-1_6b_v3_r21   1     2.139241
stablelm-2-brief-1_6b_v3_r21_2 1     2.139241

########## Second turn ##########
                                      score
model                          turn
stablelm-2-brief-1_6b_v3_r21   2     1.7625
stablelm-2-brief-1_6b_v3_r21_2 2     1.7250

########## Average ##########
                                   score
model
stablelm-2-brief-1_6b_v3_r21    1.949686
stablelm-2-brief-1_6b_v3_r21_2  1.930818

22. ~, new Galore args, 1 epoch, (galore-20240418123331) canceled at 0.5 (207 min), judginhg by train/loss not much of a change

    `optim_args="rank=488, update_proj_gap=500, scale=1.5"`
      vs
    # https://github.com/huggingface/transformers/issues/29822#issuecomment-2019325615
    # optim_args="rank=64, update_proj_gap=100, scale=0.10",
    # optim_target_modules=[r".*attn.*", r".*mlp.*"],

rank=1024 lead to VRAM overfdlow into RAM(~8.8GB), 512 was on the border of VRAM

## V4

First wanted to use unsloth, though it doesn't support stablelm

23. LORA, starting fresh from stabelm, OASST+UltraChat chat bot arena 10658082/1153120 train/test tokens, 10 epochs, (qlora-20240418192119) {'train_runtime': 57111.0872, 'train_samples_per_second': 1.824, 'train_steps_per_second': 0.007, 'train_loss': 1.1745711370212275, 'epoch': 9.84}

########## First turn ##########
                                        score
model                          turn
stablelm-2-brief-1_6b_v4_r23   1     3.912500
stablelm-2-brief-1_6b_v4_r23_2 1     3.912500

########## Second turn ##########
                                      score
model                          turn
stablelm-2-brief-1_6b_v4_r23   2     3.3500
stablelm-2-brief-1_6b_v4_r23_2 2     3.2875

########## Average ##########
                                   score
model
stablelm-2-brief-1_6b_v4_r23    3.631250
stablelm-2-brief-1_6b_v4_r23_2  3.600000

It has learned distgance to Moon yet still doesn't respond with the correct name.
```
user: What is your name?
assistant: I am Open Assistant, an open source AI model.
user: What is the distance between Earth and Moon?
assistant: The distance between Earth and the Moon is approximately 384,400 kilometers (238,800 miles).
```

24. ~, GPT4 and Claude V1 chat bot arena 12337810/1333601 train/test tokens, max_grad_norm=1.0 (vs 0.3) 2 epochs (qlora-20240419150956) {'train_runtime': 13280.6867, 'train_samples_per_second': 1.817, 'train_steps_per_second': 0.007, 'train_loss': 0.9177256400386492, 'epoch': 1.99}

########## First turn ##########
                                        score
model                          turn
stablelm-2-brief-1_6b_v4_r24_2 1     3.700000
stablelm-2-brief-1_6b_v4_r24   1     3.650000

########## Second turn ##########
                                      score
model                          turn
stablelm-2-brief-1_6b_v4_r24   2     2.8750
stablelm-2-brief-1_6b_v4_r24_2 2     2.8500

########## Average ##########
                                   score
model
stablelm-2-brief-1_6b_v4_r24_2  3.275000
stablelm-2-brief-1_6b_v4_r24    3.262500

It kinda started recognizing its' name
```
user: Hello
assistant: Hello! How can I help you today?
user: What is your name?
assistant: My name is Open Assistant. How can I help you today?
user: Who's name is Brief?
assistant: Brief is a nickname for Open Assistant.
user: So, your name is Brief?
assistant: Yes, my name is Brief.
user: Great
assistant: I'm glad you think so! How can I help you today?
```

25. ~, max_grad_norm=0.5, 1 epoch {'train_runtime': 6495.7294, 'train_samples_per_second': 1.857, 'train_steps_per_second': 0.007, 'train_loss': 0.851344088713328, 'epoch': 0.99}

# Misc/Old

## Galore

0. batch size 16, VRAM overflow, 100W updating proj gap (every 200 steps), 42W in between
1. (galore-20240401135312) batch size 12, still VRAM overflow, 100W updating proj gap (every 200 steps), 60W in between - {'train_runtime': 4243.6437, 'train_samples_per_second': 2.828, 'train_steps_per_second': 0.236, 'train_loss': 1.24991813248503, 'epoch': 3.0}
2. (galore-20240401152709) batch size 8, still VRAM overflow (~0.7GB), 100W updating proj gap (every 200 steps), 80W in between - {'train_runtime': 3409.4654, 'train_samples_per_second': 3.52, 'train_steps_per_second': 0.44, 'train_loss': 1.2464010416666667, 'epoch': 3.0}
3. (galore-20240401164153) batch size 8, 8 bit qunatization + LORA adapter, ~3GB of VRAM still available - seems like model can't be quantized with Galore {'train_runtime': 2610.4473, 'train_samples_per_second': 4.597, 'train_steps_per_second': 0.575, 'train_loss': 1.6119010416666666, 'epoch': 3.0}

## QLora

1. (qlora-20240401211220), batch size 8,  8 bit qunatization + LORA adapter, - same poor train/loss as with Galore #4 'train_runtime': 2581.5556, 'train_samples_per_second': 4.648, 'train_steps_per_second': 0.581, 'train_loss': 1.6769557291666666, 'epoch': 3.0}

2. (), batch size 8,  4 bit qunatization + new LORA config, 95W - still poor train/loss {'train_runtime': 1297.316, 'train_samples_per_second': 9.25, 'train_steps_per_second': 1.156, 'train_loss': 1.7481875, 'epoch': 3.0}

```
lora_config = LoraConfig(
    r=32, 
    lora_alpha=32, 
    target_modules = [ "q_proj", "k_proj", "v_proj", "dense" ],
    lora_dropout=0.1, 
    bias="none", 
    task_type="CAUSAL_LM",
)
```

3. (qlora-20240402101114) batch size 4, new LORA config, optim="paged_adamw_8bit", 55W - stuck at step 250 of 399 (?doing gradient chekpointing), strange spikles in train/loss following steep decline

Batch 8 10GB VRAM ~ 10s/it, Batch 6 8.6GB VRAM - ~ 6s/it, Batch 4 - ~8GB 2-3s/it (and latter huge delays due to gradient chekpointing)
```
lora_config = LoraConfig(
    # r=8,
    # target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    # bias="none",
    # task_type="CAUSAL_LM",
    r=64, 
    lora_alpha=16, 
    target_modules="all-linear",
    bias="none", 
    task_type="CAUSAL_LM",
)
```

4. (qlora-20240402180539) new configs from https://www.philschmid.de/fine-tune-llms-in-2024-with-trl, 85W {'train_runtime': 4605.958, 'train_samples_per_second': 4.215, 'train_steps_per_second': 0.702, 'train_loss': 1.457901078504991, 'epoch': 3.0} - after each epoch there was an abrupt decline in train/loss going below 1.0 at the end

5. (qlora-20240402225225) 1024 max_seq_length(vs. s256), 4 epochs - 70-85W (vs. 3) {'train_runtime': 6469.4586, 'train_samples_per_second': 1.0, 'train_steps_per_second': 0.25, 'train_loss': 1.2151240456923105, 'epoch': 4.0}      

6. (qlora-20240403193153) same as #5 + flash_attention_2, batch 1, ~80W {'train_runtime': 4431.736, 'train_samples_per_second': 1.46, 'train_steps_per_second': 0.73, 'train_loss': 1.1438023438964993, 'epoch': 4.0}

7. (qlora-20240404164313) same as #6, GPU overclock, ~85W {'train_runtime': 3977.5122, 'train_samples_per_second': 1.631, 'train_steps_per_second': 0.816, 'train_loss': 1.151490448490612, 'epoch': 4.0}


# Ultrachat, Jupiter

## Galore

1. 28.03, 100 steps, no use_reentrant, max_sequence 1024, packing True, per_device_train_batch_size 4, 1% dataset - {'train_runtime': 28413.2926, 'train_samples_per_second': 0.056, 'train_steps_per_second': 0.004, 'train_loss': 1.413594741821289, 'epoch': 0.47} - GPU ~30W

2. 29.03,  1000 steps, use_reentrant False, max_sequence 512, packing True, per_device_train_batch_size 1, 10% dataset - {'train_runtime': 2763.5962, 'train_samples_per_second': 1.447, 'train_steps_per_second': 0.362, 'train_loss': 1.3647120246887208, 'epoch': 0.05} - GPU at ~100W, 2763s vs 2950s (#4 loara), when disaling LORA adapters train failed with out of memory error.

3. 31.03,  1000 steps, use_reentrant False, max_sequence 512, packing True, per_device_train_batch_size 4, quantization 8 bit, 10% dataset {'train_runtime': 3262.9417, 'train_samples_per_second': 1.226, 'train_steps_per_second': 0.306, 'train_loss': 1.36846484375, 'epoch': 0.05} - 27W

4. 31.03,  1000 steps, use_reentrant False, max_sequence 512, packing True, per_device_train_batch_size 4, quantization 8 bit, 10% {~4000s run time} - dataset 50W, epoch 0.3 at step 320, 21.5m per 320 steps

5. 05.04 - Matched params, QLora w. flash attention (#7), instead of ~1h training for 4 epochs ETA was 11h (at 1%, step 58)

6. 05.04 - same as #5, on Windows without flash attention on (#7), instead of ~1h training for 4 epochs ETA was 10.5h

## QLora

1. 28.03, 1000 steps, use_reentrant False, max_sequence 512, packing False, per_device_train_batch_size 4, 3% dataset {'train_runtime': 935.5676, 'train_samples_per_second': 17.102, 'train_steps_per_second': 1.069, 'train_loss': 2.3004047622680663, 'epoch': 0.36}

2. 28.03, 1000 steps, use_reentrant False, max_sequence 512, packing False, per_device_train_batch_size 4, 100% dataset {'train_runtime': 885.1452, 'train_samples_per_second': 18.076, 'train_steps_per_second': 1.13, 'train_loss': 2.2888641471862794, 'epoch': 0.01} - GPU ~60W

3. 28.03,  1000 steps, use_reentrant False, max_sequence 512, packing True, per_device_train_batch_size 2, 10% dataset {'train_runtime': 10752.6906, 'train_samples_per_second': 0.744, 'train_steps_per_second': 0.093, 'train_loss': 1.3185794553756713, 'epoch': 0.1} - GPU showed ~72W
}
-- `with per_device_train_batch_size 4, max_sequence 1024, packing True` GPU wsa underloaded at ~30W. With packing set to false training speed goes significantly up yet train/loss is twice higher AND the trained model behaves poorely (doesn't return a single ## ASSISTANT reply)

4. 28.03,  1000 steps, use_reentrant False, max_sequence 512, packing True, per_device_train_batch_size 1, 10% dataset {'train_runtime': 2950.2465, 'train_samples_per_second': 1.356, 'train_steps_per_second': 0.339, 'train_loss': 1.3293660106658935, 'epoch': 0.05}

5. 28.03,  17000 steps, use_reentrant False, max_sequence 512, packing True, per_device_train_batch_size 1, 100% dataset {'train_runtime': 60613.0943, 'train_samples_per_second': 1.122, 'train_steps_per_second': 0.28, 'train_loss': 1.6384484511543722, 'epoch': 0.05}

6. 30.03,  1000 steps, use_reentrant False, max_sequence 512, packing True, per_device_train_batch_size 1, quantization 8 bit, 10% dataset {'train_runtime': 2748.56, 'train_samples_per_second': 1.455, 'train_steps_per_second': 0.364, 'train_loss': 1.366098388671875, 'epoch': 0.05} - ~2-3GB VRAM vs 6.8 (as measured before .train()), GPU at ~50W

7. 30.03,  1000 steps, use_reentrant False, max_sequence 512, packing True, per_device_train_batch_size 2, quantization 8 bit, 10% dataset {'train_runtime': 3564.1785, 'train_samples_per_second': 2.245, 'train_steps_per_second': 0.281, 'train_loss': 1.3811524658203125, 'epoch': 0.1} - ~65W, twice the epoch of #6

8. 30.03,  1000 steps, use_reentrant False, max_sequence 512, packing True, per_device_train_batch_size 4, quantization 8 bit, 10% dataset - {'train_runtime': 5755.853, 'train_samples_per_second': 2.78, 'train_steps_per_second': 0.174, 'train_loss': 1.410659423828125, 'epoch': 0.21} - ~73W, chatting is still can't give a valid response to 'Hello' (some verbose nonsense with several ASSISTANT patrts), yet it can give some coherent (yet wrong) answer to what is Gravity
