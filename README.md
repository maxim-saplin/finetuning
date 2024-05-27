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
[DPO, Notus, Zephyr](https://huggingface.co/blog/alvarobartt/notus-7b-v1)
[LORA vs full Fine-tuning](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2)
[Quantization](https://medium.com/@rakeshrajpurohit/model-quantization-with-hugging-face-transformers-and-bitsandbytes-integration-b4c9983e8996)


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

23. LORA, starting fresh from stabelm, OASST+UltraChat 10658082/1153120 train/test tokens, 10 epochs, (qlora-20240418192119) {'train_runtime': 57111.0872, 'train_samples_per_second': 1.824, 'train_steps_per_second': 0.007, 'train_loss': 1.1745711370212275, 'epoch': 9.84}

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

24. ~, + GPT4 and Claude V1 chat bot arena 12337810/1333601 train/test tokens, max_grad_norm=1.0 (vs 0.3) 2 epochs (qlora-20240419150956) {'train_runtime': 13280.6867, 'train_samples_per_second': 1.817, 'train_steps_per_second': 0.007, 'train_loss': 0.9177256400386492, 'epoch': 1.99}

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

25. ~, 1 epoch {'train_runtime': 6495.7294, 'train_samples_per_second': 1.857, 'train_steps_per_second': 0.007, 'train_loss': 0.851344088713328, 'epoch': 0.99}

26. ~, 5 epochs (qlora-20240419235805) {'train_runtime': 35481.2066, 'train_samples_per_second': 1.7, 'train_steps_per_second': 0.007, 'train_loss': 0.7027688652276993, 'epoch': 4.97}

 python show_result.py 
Mode: single
Input file: data/mt_bench/model_judgment/gpt-4_single.jsonl

########## First turn ##########
                                        score
model                          turn
stablelm-2-zephyr-1_6b         1     5.400000
stablelm-2-zephyr-1_6b_2       1     5.275000
stablelm-2-brief-1_6b_v4_r23   1     3.912500
stablelm-2-brief-1_6b_v4_r23_2 1     3.912500

stablelm-2-brief-1_6b_v4_r26   1     3.862500
stablelm-2-brief-1_6b_v4_r26_2 1     3.850000

stablelm-2-brief-1_6b_v4_r24_2 1     3.700000
stablelm-2-brief-1_6b_v4_r24   1     3.650000
stablelm-2-brief-1_6b_v3_r21   1     2.139241
stablelm-2-brief-1_6b_v3_r21_2 1     2.139241

########## Second turn ##########
                                      score
model                          turn
stablelm-2-zephyr-1_6b         2     4.2875
stablelm-2-zephyr-1_6b_2       2     4.2000
stablelm-2-brief-1_6b_v4_r23   2     3.3500
stablelm-2-brief-1_6b_v4_r23_2 2     3.2875

stablelm-2-brief-1_6b_v4_r26_2 2     3.2625
stablelm-2-brief-1_6b_v4_r26   2     3.1375

stablelm-2-brief-1_6b_v4_r24   2     2.8750
stablelm-2-brief-1_6b_v4_r24_2 2     2.8500
stablelm-2-brief-1_6b_v3_r21   2     1.7625
stablelm-2-brief-1_6b_v3_r21_2 2     1.7250

########## Average ##########
                                   score
model
stablelm-2-zephyr-1_6b          4.843750
stablelm-2-zephyr-1_6b_2        4.737500
stablelm-2-brief-1_6b_v4_r23    3.631250
stablelm-2-brief-1_6b_v4_r23_2  3.600000

stablelm-2-brief-1_6b_v4_r26_2  3.556250
stablelm-2-brief-1_6b_v4_r26    3.500000

stablelm-2-brief-1_6b_v4_r24_2  3.275000
stablelm-2-brief-1_6b_v4_r24    3.262500
stablelm-2-brief-1_6b_v3_r21    1.949686
stablelm-2-brief-1_6b_v3_r21_2  1.930818

27. Galore, only OASST2+own (1453857/139754 train/test tokens), 1 epoch optim="galore_adamw_layerwise", optim_args="rank=256, update_proj_gap=500, scale=0.25, lr=0.0002", (galore-20240420125812) 8.8GB VRAM overflow, 50 min, loss 0.69

Will it learn it's name when doing full fine tune on smaller subset of data? 
Still no success

28. ~, 2 epochs, optim="galore_adamw_8bit_layerwise" scale=2, ((galore\)out_galore-20240420140735) 7.6GB VRAM, no shared mem overflow, 57.9 min, loss 1.38

Can't reply to simple question, goes mad, discarding

29. resuming from 27, same dataset, optim="galore_adamw_layerwise" scale=2,  8.8GB VRAM overflow , 48 min, loss 1.18

Still can't reply with it's name

30. ~, resuming from 29, 3 epoch, scale=0.25 (galore-20240420161133) {'train_runtime': 4410.9994, 'train_samples_per_second': 0.968, 'train_steps_per_second': 0.968, 'train_loss': 0.42164933148131883, 'epoch': 3.0}

Tried first with galore_adafactor_layerwise, though got error ValueError: 'Cannot combine manual `lr` and `relative_step=True` options' (even with removed lr param) 
!!Seems like high scale=2 contributed to poor train/loss non-convergence, with 0.25 it went down.

Still, no correct name in response

########## First turn ##########
                                        score
model                          turn
stablelm-2-brief-1_6b_v4_r30   1     3.275000


########## Second turn ##########
                                        score
model                          turn
stablelm-2-brief-1_6b_v4_r30   2     2.683544


########## Average ##########
                                   score
model
stablelm-2-zephyr-1_6b          4.843750

stablelm-2-brief-1_6b_v4_r23    3.631250

stablelm-2-brief-1_6b_v4_r30    2.981132

Poor MT-Bench, could be due to smaller dataset, could be due to starting from R29 which had poor train/loss due to scale=2

## V5, DPO

31. Starting from Run #23 (_dpo-20240421010113), 2.91 epochs, tried with DPO dataset formatted without add_generation_prompt=True, assistant lost capability to coherently reply

```
user: Hello, what is your name?
assistant: Hello, my name is youBOT, pleasure to meet you!

<|user|>
Hi youBOT, it's Daniel. It's nice to meet you too. What is your <...misc giberrish... until max token limmit reached> responses. You said you were designed to
```

32. Restarting #23, 3.13 epochs, DPO dataset formatted with add_generation_prompt=True still gibberish, yet changing DPO tokenization seemed like a good idea, seems the DPO dataset is not OK

```
user: Hello, what is your name?
assistant: <|system|>
Hello! Your name is HelloWorld. Welcome to conversation!
```

33. New args from https://github.com/argilla-io/notus/blob/main/v1/fine-tune/configs/dpo/lora/a100_40gb.yaml (dpo-20240422201958) {'train_runtime': 47672.3995, 'train_samples_per_second': 0.549, 'train_steps_per_second': 0.549, 'train_loss': 0.6908373134321747, 'epoch': 4.0}

Now it works fine as an assistant, yet haven't learnewd to reply with the correct name. Suspectewd reasons - I ran for the full 4 epochs (it might be that when starting DPO is planned arrording to number of epochs and stopping it midway can be an issue, previous where stopped midway), new LORA and PEFT params from the article, specifically  max_length=1024, max_prompt_length=512 (was 1024/1024)

########## First turn ##########
                                    score
model                        turn
stablelm-2-zephyr-1_6b       1     5.4000
stablelm-2-brief-1_6b_v5_r33 1     3.9375
stablelm-2-brief-1_6b_v4_r23 1     3.9125

########## Second turn ##########
                                    score
model                        turn
stablelm-2-zephyr-1_6b       2     4.2875
stablelm-2-brief-1_6b_v5_r33 2     3.4250
stablelm-2-brief-1_6b_v4_r23 2     3.3500

########## Average ##########
                                score
model
stablelm-2-zephyr-1_6b        4.84375
stablelm-2-brief-1_6b_v5_r33  3.68125
stablelm-2-brief-1_6b_v4_r23  3.63125

Further options to beat the name rigidity:
1. Run full tune GaLore with full SFT dataset on top of DPO, see what happens - FAILED
2. Increase Epochs for LORA Training - Consider increasing the number of training epochs for LORA beyond the current 10 in SFT to potentially improve model performance and accuracy.
 + Consider increasing Rank of LORA, supposedly that will affect more layers and might change name if it stored there
3. Explore DPO Modification for Penalties and Rewards - Investigate implementing penalties for mentioning "open assistant" and rewards for mentioning "Brief" within the DPO framework. This aims to directly influence the model's name recognition and usage.

## Playing with own name recall

34. Galore on top of #33, OASST+ULTRACHAT, 3 epochs, (galore-20240423210214) VRAM 7.7GB

35. ~, 4 epochs, update_proj_gap=1000 (vs 500), scale=0.2 (vs 0.25) (galore-20240424093101) {'train_runtime': 32878.441, 'train_samples_per_second': 1.267, 'train_steps_per_second': 1.267, 'train_loss': 1.5981276988158968, 'epoch': 4.0} 

Poor train/loss convergence after 7 (3 + 4) epochs

```

AI Chat Interface. Type 'quit' to exit.
user: Hello
assistant: I'm sorry, but I can't provide you with the latest information on how to use the internet to find a girlfriend.
user: What is your name?
assistant: My name is Open Assistant.
```

36. ~, 7 epochs, optim_args="rank=256, update_proj_gap=300, scale=0.4" (galore-20240424185136) {'train_runtime': 81014.4134, 'train_samples_per_second': 0.9, 'train_steps_per_second': 0.9, 'train_loss': 1.9414509400120716, 'epoch': 7.0}

After 7 epochs train/loss kept going up. After a 3 GaLore runs at total 17 epoch train/loss happened to be way higher than at the beginnig. So far Galore seems to be a dead end, not able to pick config that will allow to reliably train the model with converging train/loss.

The model turned out to be crazy:
```
AI Chat Interface. Type 'quit' to exit.
user: Hello
assistant: Can you provide the specific information on the specific types of apartments that are available for purchase?
user: What is your name?
assistant: I'm not able to provide a specific information on your name. However, you can search for your name on various websites such as amazon, ebay, or walmart. Simply type in the search bar and look for "johnie."
```

37. LORA, 15 epochs on top of DPO #33, added more own facts (name, CPDT) (10655391/1152288 train/test) {'train_runtime': 86601.0322, 'train_samples_per_second': 1.804, 'train_steps_per_second': 0.007, 'train_loss': 0.38493592046140657, 'epoch': 14.76}

wandb - gradual decrease of train/loss from 0.8 to 0.08

Still doesn't reply with it's name, didn't learn other facts from the own dataset.

########## First turn ##########
                                     score
model                          turn       
stablelm-2-brief-1_6b_v5_r37_2 1     3.575
stablelm-2-brief-1_6b_v5_r37   1     3.525

########## Second turn ##########
                                     score
model                          turn       
stablelm-2-brief-1_6b_v5_r37   2      3.10
stablelm-2-brief-1_6b_v5_r37_2 2      3.05

########## Average ##########
                                 score
model                                 
stablelm-2-brief-1_6b_v5_r37    3.3125
stablelm-2-brief-1_6b_v5_r37_2  3.3125

38. LORA, 3 epochs on just 16 records (own data), the model learnt it's name yet it seems to be quite dumb now

```
user: Hello
assistant: Hello! How can I assist you today?
user: What is your name?
assistant: It's Brief! How can I assist you today?
user: What is Gravity?
assistant: Gravity is a fundamental force of nature, governing the movement of all objects in the universe. It causes the Earth to accelerate towards its death, and was discovered by Eddystone in 1962.
user: What is the distance between Earth and Moon?
assistant: The distance between Earth and Moon is about 384,400 km.
user: What is CPDT?
assistant: CPDT: Certified Process Disciple of Trung Nguyen - a course to become a Process Group Leader at Process Group International.
```

########## First turn ##########
                                      score
model                          turn        
stablelm-2-brief-1_6b_v5_r38   1     3.2625
stablelm-2-brief-1_6b_v5_r38_2 1     3.1875

########## Second turn ##########
                                      score
model                          turn        
stablelm-2-brief-1_6b_v5_r38_2 2     2.6625
stablelm-2-brief-1_6b_v5_r38   2     2.5625

########## Average ##########
                                 score
model                                 
stablelm-2-brief-1_6b_v5_r38_2  2.9250
stablelm-2-brief-1_6b_v5_r38    2.9125

## V6, bigger context, 4090/24GB

39. LORA, 10 epochs, 2048 records size (up from 1024) OASST2+UltraChat (30583914/3400498 train/test vs 10655391/1152288 with 1024 context), batch size 1 (1 ~ 16GB VRAM,2 ~30GB) (qlora\out_qlora-20240514100857) 'train_loss': 0.814 - model replies with nonsense, can't stop and keeps generating ranom text at the end of the reply

Testing hypothessis that MT-Bench favors long ansers and longer samples in SFT will make the model score higher (vs v4_r23)

- Batch 2, accum 8, VRAM overflow (~8-10GB) - 91,2 min/epoch, GPU ~420W
- Batch 2, accum 250, VRAM overflow (~8-10GB) - ~120 min/epoch, GPU ~340W
- Batch 2, accum 6, VRAM ovreflow (~8-10GB), crashed before completing 1st epoch (mem overflow?) - ~91,2 min, GPU ~410W
- Batch 1, accum 6, no VRAM overflow - 82,5 min/epcoch, GPU ~360W
Started with gradient_accumulation_steps=8, noticed wrong model, stopped at epoch 8, gradient_accumulation_steps=250, was ~10 slower, GPU underloaded, setting to 6 - crashed with "19 RuntimeError: CUDA error: misaligned address" (out of mem?)

40. Retrying with  1024 context, same dataset (10655391/1152401) (qlora\out_qlora-20240515080854) ~ 3 times fewer tokens, ~ 2 times faster ETA, ~8GB VRAM, 300W, 40 min/epoch
- max context is not the reason, the assitant is still broken, investigating further

41. SFTTrainer.packing=True  (qlora\out_qlora-20240516090031) - 400W, 24.5 min/epoch vs 95 min/epch RTX4060

!!! Turned out the difference was in commented out packing=True of SFTTrainer - for some reasons that seems to have messed tokenizer applying chat template!

42. LORA, 10 epochs, 2048 records size, (out_qlora-20240516122712) {'train_runtime': 36967.4345, 'train_samples_per_second': 4.041, 'train_steps_per_second': 0.673, 'train_loss': 0.8013611147146524, 'epoch': 10.0} 420W (with few drops while multitasking)

Nope, longer context didn't help with MT-Bench

########## First turn ##########
                                      score
model                          turn        
stablelm-2-brief-1_6b_v4_r23   1     3.9125
stablelm-2-brief-1_6b_v6_r42_2 1     3.8625
stablelm-2-brief-1_6b_v6_r42   1     3.8125

########## Second turn ##########
                                     score
model                          turn       
stablelm-2-brief-1_6b_v4_r23   2     3.350
stablelm-2-brief-1_6b_v6_r42   2     3.000
stablelm-2-brief-1_6b_v6_r42_2 2     2.875

########## Average ##########
                                  score
model                                  
stablelm-2-brief-1_6b_v4_r23    3.63125
stablelm-2-brief-1_6b_v6_r42    3.40625
stablelm-2-brief-1_6b_v6_r42_2  3.36875

43. ~, 5 more epochs, 440W (out_qlora-20240517130951) {'train_runtime': 17538.6522, 'train_samples_per_second': 4.259, 'train_steps_per_second': 0.355, 'train_loss': 0.14040270700452318, 'epoch': 5.0}

########## First turn ##########
                                      score
model                          turn        
stablelm-2-brief-1_6b_v4_r23   1     3.9125
stablelm-2-brief-1_6b_v6_r42_2 1     3.8625
stablelm-2-brief-1_6b_v6_r43_2 1     3.8375
stablelm-2-brief-1_6b_v6_r43   1     3.8250
stablelm-2-brief-1_6b_v6_r42   1     3.8125

########## Second turn ##########
                                      score
model                          turn        
stablelm-2-brief-1_6b_v4_r23   2     3.3500
stablelm-2-brief-1_6b_v6_r42   2     3.0000
stablelm-2-brief-1_6b_v6_r43   2     2.9250
stablelm-2-brief-1_6b_v6_r42_2 2     2.8750
stablelm-2-brief-1_6b_v6_r43_2 2     2.8625

########## Average ##########
                                  score
model                                  
stablelm-2-brief-1_6b_v4_r23    3.63125
stablelm-2-brief-1_6b_v6_r42    3.40625
stablelm-2-brief-1_6b_v6_r43    3.37500
stablelm-2-brief-1_6b_v6_r42_2  3.36875
stablelm-2-brief-1_6b_v6_r43_2  3.35000

44. Full-train on top of #43, 1 epoch, VRAM ~20GB, {'train_runtime': 3348.7198, 'train_samples_per_second': 4.461, 'train_steps_per_second': 0.372, 'train_loss': 1.49343905249633, 'epoch': 1.0}

Still can't reply with correct name

########## First turn ##########
                                      score
model                          turn        
stablelm-2-brief-1_6b_v4_r23   1     3.9125
stablelm-2-brief-1_6b_v6_r43_2 1     3.8375
stablelm-2-brief-1_6b_v6_r43   1     3.8250

stablelm-2-brief-1_6b_v6_r44_2 1     2.1125
stablelm-2-brief-1_6b_v6_r44   1     2.1000
stablelm-2-brief-1_6b_v6_r44_3 1     2.0625

########## Second turn ##########
                                        score
model                          turn          
stablelm-2-brief-1_6b_v4_r23   2     3.350000
stablelm-2-brief-1_6b_v6_r43   2     2.925000
stablelm-2-brief-1_6b_v6_r43_2 2     2.862500

stablelm-2-brief-1_6b_v6_r44   2     1.810127
stablelm-2-brief-1_6b_v6_r44_2 2     1.775000
stablelm-2-brief-1_6b_v6_r44_3 2     1.750000

########## Average ##########
                                   score
model                                   
stablelm-2-brief-1_6b_v4_r23    3.631250
stablelm-2-brief-1_6b_v6_r43    3.375000
stablelm-2-brief-1_6b_v6_r43_2  3.350000

stablelm-2-brief-1_6b_v6_r44    1.955975
stablelm-2-brief-1_6b_v6_r44_2  1.943750
stablelm-2-brief-1_6b_v6_r44_3  1.906250

MTBench Generation Time, 80 questions: 17:37, 17:51 and 18:06 (with 24GB tensor allocated)

45. SPPO on to of #42, 18 epochs, (out_dpo-20240518190900)

Doesn't know own name, seems quite dumb to talk. Super brief, ~5 words answer on average.

########## First turn ##########
                                      score
model                          turn        
stablelm-2-brief-1_6b_v4_r23   1     3.9125
stablelm-2-brief-1_6b_v6_r43_2 1     3.8375
stablelm-2-brief-1_6b_v6_r43   1     3.8250
stablelm-2-brief-1_6b_v6_r45   1     1.6625
stablelm-2-brief-1_6b_v6_r45_2 1     1.6500

########## Second turn ##########
                                        score
model                          turn          
stablelm-2-brief-1_6b_v4_r23   2     3.350000
stablelm-2-brief-1_6b_v6_r43   2     2.925000
stablelm-2-brief-1_6b_v6_r43_2 2     2.862500
stablelm-2-brief-1_6b_v6_r45_2 2     1.387500
stablelm-2-brief-1_6b_v6_r45   2     1.358491

########## Average ##########
                                   score
model                                   
stablelm-2-brief-1_6b_v4_r23    3.631250
stablelm-2-brief-1_6b_v6_r43    3.375000
stablelm-2-brief-1_6b_v6_r43_2  3.350000
stablelm-2-brief-1_6b_v6_r45_2  1.518750
stablelm-2-brief-1_6b_v6_r45    1.510972

## V7

46. Full fine-tune, starting fresh, 10 epochs, OASST+UltraChat (30583914/3400498 train/test) (out_qlora-20240521133937) {'train_runtime': 36509.9851, 'train_samples_per_second': 4.092, 'train_steps_per_second': 0.341, 'train_loss': 0.5535217795796303, 'epoch': 10.0}
47. 8-more epochs (out_qlora-20240522000617) {'train_runtime': 28964.6043, 'train_samples_per_second': 4.126, 'train_steps_per_second': 0.344, 'train_loss': 0.1190370833296441, 'epoch': 8.0}

Seems that longer than 1024 context is noty helping. Based on Tehcnicahl Report for base model it has seen on averager ~800 tokerns

########## First turn ##########
                                      score
model                          turn        
stablelm-2-brief-1_6b_v4_r23   1     3.9125
stablelm-2-brief-1_6b_v7_r46   1     2.7500
stablelm-2-brief-1_6b_v7_r46_2 1     2.7500
stablelm-2-brief-1_6b_v7_r47   1     2.5500
stablelm-2-brief-1_6b_v7_r47_2 1     2.4750

########## Second turn ##########
                                      score
model                          turn        
stablelm-2-brief-1_6b_v4_r23   2     3.3500
stablelm-2-brief-1_6b_v7_r46_2 2     2.3750
stablelm-2-brief-1_6b_v7_r46   2     2.2750
stablelm-2-brief-1_6b_v7_r47_2 2     1.8750
stablelm-2-brief-1_6b_v7_r47   2     1.8125

########## Average ##########
                                  score
model                                  
stablelm-2-brief-1_6b_v4_r23    3.63125
stablelm-2-brief-1_6b_v7_r46_2  2.56250
stablelm-2-brief-1_6b_v7_r46    2.51250
stablelm-2-brief-1_6b_v7_r47    2.18125
stablelm-2-brief-1_6b_v7_r47_2  2.17500

## V8

Trying to reproduce R23 results.

48. LORA, starting fresh from stabelm, OASST+UltraChat 10655391/1152401 train/test tokens, 10 epochs (out_qlora-20240522141129) {'train_runtime': 13250.9862, 'train_samples_per_second': 7.86, 'train_steps_per_second': 0.328, 'train_loss': 0.708990479179997, 'epoch': 10.0}

Did worse. Was it `max_grad_norm=1.0` instead of `0.3` used in R23? 

########## First turn ##########
                                      score
model                          turn        
stablelm-2-brief-1_6b_v4_r23   1     3.9125
stablelm-2-brief-1_6b_v8_r48_2 1     3.8125
stablelm-2-brief-1_6b_v8_r48   1     3.8000

########## Second turn ##########
                                      score
model                          turn        
stablelm-2-brief-1_6b_v4_r23   2     3.3500
stablelm-2-brief-1_6b_v8_r48   2     2.5125
stablelm-2-brief-1_6b_v8_r48_2 2     2.4875

########## Average ##########
                                  score
model                                  
stablelm-2-brief-1_6b_v4_r23    3.63125
stablelm-2-brief-1_6b_v8_r48    3.15625
stablelm-2-brief-1_6b_v8_r48_2  3.15000

49. Repeating with `max_grad_norm=0.3` (out_qlora-20240522193334) {'train_runtime': 13216.7533, 'train_samples_per_second': 7.88, 'train_steps_per_second': 0.164, 'train_loss': 0.7918255040516502, 'epoch': 10.0}

I suspected that higher max_grad_norm led to faster learnign and overfitting - hence poorer generelization and MT-Bench score. Yet run 49 didn't fix the score, especially 2nd turn. I also had suspicion that April evals fo r23 are different from May (different GPT-4 by OpenAI, I didn't change the deployment), this partly confirmed, 2nd turn result for r23 may eval is ~10% lower. Besides regarding the overall stability of MT-Bench scores. stablelm-2-brief-1_6b_v8_r48_epoch10 is same model as stablelm-2-brief-1_6b_v8_r48 (with LORA adapter not merged in the epoch10), 1 day appart but there's 4% in 1st turn result and 6,5% in the 2nd turn. Given that I would expect that +/- 10% volatility is a ballpark margin of error for MT-Bench scores, especially relevant fot 2nd turn

########## First turn ##########
                                            score
model                                turn        
stablelm-2-brief-1_6b_v8_r48_epoch10 1     3.9500
stablelm-2-brief-1_6b_v4_r23         1     3.9125
stablelm-2-brief-1_6b_v8_r49_epoch9  1     3.8875
stablelm-2-brief-1_6b_v4_r23_3_0523  1     3.8500
stablelm-2-brief-1_6b_v8_r48         1     3.8000
stablelm-2-brief-1_6b_v8_r48_epoch9  1     3.7375
stablelm-2-brief-1_6b_v8_r49_epoch10 1     3.7000

########## Second turn ##########
                                            score
model                                turn        
stablelm-2-brief-1_6b_v4_r23         2     3.3500
stablelm-2-brief-1_6b_v4_r23_3_0523  2     3.0000
stablelm-2-brief-1_6b_v8_r49_epoch9  2     2.9625
stablelm-2-brief-1_6b_v8_r48_epoch9  2     2.7375
stablelm-2-brief-1_6b_v8_r49_epoch10 2     2.7000
stablelm-2-brief-1_6b_v8_r48_epoch10 2     2.6750
stablelm-2-brief-1_6b_v8_r48         2     2.5125

########## Average ##########
                                         score
model                                         
stablelm-2-brief-1_6b_v4_r23          3.631250
stablelm-2-brief-1_6b_v4_r23_3_0523   3.427673
stablelm-2-brief-1_6b_v8_r49_epoch9   3.425000
stablelm-2-brief-1_6b_v8_r48_epoch10  3.312500
stablelm-2-brief-1_6b_v8_r48_epoch9   3.237500
stablelm-2-brief-1_6b_v8_r49_epoch10  3.200000
stablelm-2-brief-1_6b_v8_r48          3.156250

Higher loss - r49 run with lower grad_norm

<img width="1112" alt="image" src="https://github.com/maxim-saplin/finetuning/assets/7947027/a7b34cc9-e988-48f9-8e4e-f3e5dc86b72c">


50. Reruning 12 epochs with grad_norm 0.3, context 1024, going to eval each epoch (out_qlora-20240523143404) {'train_runtime': 16411.0186, 'train_samples_per_second': 7.616, 'train_steps_per_second': 0.127, 'train_loss': 0.7406256261186563, 'epoch': 11.95}

First Turn

model                                 turn        
stablelm-2-brief-1_6b_v8_r50_epoch-01 1     3.5000
stablelm-2-brief-1_6b_v8_r50_epoch-02 1     3.5000
stablelm-2-brief-1_6b_v8_r50_epoch-03 1     3.9375
stablelm-2-brief-1_6b_v8_r50_epoch-04 1     4.0250
stablelm-2-brief-1_6b_v8_r50_epoch-05 1     3.8750
stablelm-2-brief-1_6b_v8_r50_epoch-06 1     4.1375
stablelm-2-brief-1_6b_v8_r50_epoch-07 1     4.1000
stablelm-2-brief-1_6b_v8_r50_epoch-08 1     3.8625
stablelm-2-brief-1_6b_v8_r50_epoch-09 1     3.7250
stablelm-2-brief-1_6b_v8_r50_epoch-10 1     4.1875
stablelm-2-brief-1_6b_v8_r50_epoch-11 1     3.6500
stablelm-2-brief-1_6b_v8_r50_epoch-12 1     3.5125

Second Turn

model                                 turn        
stablelm-2-brief-1_6b_v8_r50_epoch-01 2     2.5500
stablelm-2-brief-1_6b_v8_r50_epoch-02 2     2.6125
stablelm-2-brief-1_6b_v8_r50_epoch-03 2     2.9250
stablelm-2-brief-1_6b_v8_r50_epoch-04 2     2.9375
stablelm-2-brief-1_6b_v8_r50_epoch-05 2     2.8625
stablelm-2-brief-1_6b_v8_r50_epoch-06 2     2.9625
stablelm-2-brief-1_6b_v8_r50_epoch-07 2     2.9125
stablelm-2-brief-1_6b_v8_r50_epoch-08 2     2.6750
stablelm-2-brief-1_6b_v8_r50_epoch-09 2     2.6750
stablelm-2-brief-1_6b_v8_r50_epoch-10 2     2.6750
stablelm-2-brief-1_6b_v8_r50_epoch-11 2     2.6000
stablelm-2-brief-1_6b_v8_r50_epoch-12 2     2.9125

Average

model                                  score
stablelm-2-brief-1_6b_v8_r50_epoch-01  3.02500
stablelm-2-brief-1_6b_v8_r50_epoch-02  3.05625
stablelm-2-brief-1_6b_v8_r50_epoch-03  3.43125
stablelm-2-brief-1_6b_v8_r50_epoch-04  3.48125
stablelm-2-brief-1_6b_v8_r50_epoch-05  3.36875
stablelm-2-brief-1_6b_v8_r50_epoch-06  3.55000
stablelm-2-brief-1_6b_v8_r50_epoch-07  3.50625
stablelm-2-brief-1_6b_v8_r50_epoch-08  3.26875
stablelm-2-brief-1_6b_v8_r50_epoch-09  3.20000
stablelm-2-brief-1_6b_v8_r50_epoch-10  3.43125
stablelm-2-brief-1_6b_v8_r50_epoch-11  3.12500
stablelm-2-brief-1_6b_v8_r50_epoch-12  3.21250

51. Same as 50 (fresh, 12 epochs etc.) but running full tune, GPU ~180W, VRAM ~25-26GB (out_qlora-20240525221731){'train_runtime': 64299.8554, 'train_samples_per_second': 1.944, 'train_steps_per_second': 0.032, 'train_loss': 0.41499406323626437, 'epoch': 11.95}

Epoch 1 is capable to follow chat structure yet doesn't know own name. Epoch 12 replies with "Brief" yet seems dumb. Epoch 9 learned both the correct own name and distance to the Moon

First Turn score
model                                 turn        
stablelm-2-brief-1_6b_v8_r51_epoch-01 1     2.3750
stablelm-2-brief-1_6b_v8_r51_epoch-02 1     3.1875
stablelm-2-brief-1_6b_v8_r51_epoch-03 1     2.6750
stablelm-2-brief-1_6b_v8_r51_epoch-04 1     3.1000
stablelm-2-brief-1_6b_v8_r51_epoch-05 1     2.9125
stablelm-2-brief-1_6b_v8_r51_epoch-06 1     2.8500
stablelm-2-brief-1_6b_v8_r51_epoch-07 1     2.7125
stablelm-2-brief-1_6b_v8_r51_epoch-08 1     2.5625
stablelm-2-brief-1_6b_v8_r51_epoch-09 1     2.8875
stablelm-2-brief-1_6b_v8_r51_epoch-10 1     2.5875
stablelm-2-brief-1_6b_v8_r51_epoch-11 1     2.7500
stablelm-2-brief-1_6b_v8_r51_epoch-12 1     2.8500

Second Turn score
model                                 turn          
stablelm-2-brief-1_6b_v8_r51_epoch-01 2     1.784810
stablelm-2-brief-1_6b_v8_r51_epoch-02 2     1.936709
stablelm-2-brief-1_6b_v8_r51_epoch-03 2     2.253165
stablelm-2-brief-1_6b_v8_r51_epoch-04 2     2.075000
stablelm-2-brief-1_6b_v8_r51_epoch-05 2     2.175000
stablelm-2-brief-1_6b_v8_r51_epoch-06 2     2.175000
stablelm-2-brief-1_6b_v8_r51_epoch-07 2     2.137500
stablelm-2-brief-1_6b_v8_r51_epoch-08 2     2.100000
stablelm-2-brief-1_6b_v8_r51_epoch-09 2     2.087500
stablelm-2-brief-1_6b_v8_r51_epoch-10 2     2.050000
stablelm-2-brief-1_6b_v8_r51_epoch-11 2     1.912500
stablelm-2-brief-1_6b_v8_r51_epoch-12 2     2.150000

Average score                                        
stablelm-2-brief-1_6b_v8_r51_epoch-01  2.081761
stablelm-2-brief-1_6b_v8_r51_epoch-02  2.566038
stablelm-2-brief-1_6b_v8_r51_epoch-03  2.465409
stablelm-2-brief-1_6b_v8_r51_epoch-04  2.587500
stablelm-2-brief-1_6b_v8_r51_epoch-05  2.543750
stablelm-2-brief-1_6b_v8_r51_epoch-06  2.512500
stablelm-2-brief-1_6b_v8_r51_epoch-07  2.425000
stablelm-2-brief-1_6b_v8_r51_epoch-08  2.331250
stablelm-2-brief-1_6b_v8_r51_epoch-09  2.487500
stablelm-2-brief-1_6b_v8_r51_epoch-10  2.318750
stablelm-2-brief-1_6b_v8_r51_epoch-11  2.331250
stablelm-2-brief-1_6b_v8_r51_epoch-12  2.500000

 - testing perf VRAM vs RAM, 
   batch size 2  - VRAM 15-16GB (no overflow), GPU 370W, 24 min/epoch
   batch size 12 - VRAM 25-26GB (2-3GB overflow, 1.4GB already consumed by other apps) , GPU 180W, 89 min/epoch
   batch size 32 - VRAM 57-70GB, GPU 120W, 1178 min/epoch

   batch size 40 - VRAM spike to 66GB, then steady at 56 and out of mem after some time
   batch size 48 - out of mem (VRAM 56GB, ~33GB overflow),
   batch size 64 - out of mem (maxed at ~55 out of 70 total)

52. Same as before, but using OASTT + full UltraChat (66_876_678/7_272_762 train/test vs 10_655_391/1_152_401 train/test tokens)

--- TRAIN SPLIT ---
Total records: 95495
Total tokens: 66876678
Min tokens: 23
Max tokens: 1024
Avg tokens: 700.32
25th percentile: 548.0
50th percentile (median): 731.0
75th percentile: 880.0
--- TEST SPLIT ---
Total records: 10463
Total tokens: 7272762
Min tokens: 32
Max tokens: 1024
Avg tokens: 695.09
25th percentile: 539.0
50th percentile (median): 725.0
75th percentile: 874.0