# Links
[Galore](https://medium.com/@geronimo7/llm-training-on-consumer-gpus-with-galore-d25075143cfb#:~:text=GaLore%20vs.-,LoRA,edging%20out%20in%20the%20benchmarks.)
[QLORA](https://pytorch.org/blog/finetune-llms/)

# OASTT, .py

## Clean versions

1. GaLore 4 epochs, (galore_oastt2\out_galore-20240405095444) {'train_runtime': 34952.6261, 'train_samples_per_second': 0.19, 'train_steps_per_second': 0.19, 'train_loss': 1.375399877885486, 'epoch': 4.0} - there're ~1min pauses after each step, GPU load fluctualtes between 80 and 100W (likely goes down to 80W during those pauses), VRAM ~7GB, ETA was estimated at 1h, take ~10h, train/loss is much worse than #2 QLora

2. QLORA 4 epochs, (qlora_oastt2\out_qlora-20240405200217) {'train_runtime': 4486.351, 'train_samples_per_second': 1.477, 'train_steps_per_second': 0.369, 'train_loss': 1.0640918418477123, 'epoch': 4.0} - GPU non-overclocked ~80W, VRAM ~7.5GB

3. QLORA 11 epochs, (qlora_oastt2\out_qlora-20240406123945) {'train_runtime': 16831.0707, 'train_samples_per_second': 1.083, 'train_steps_per_second': 0.271, 'train_loss': 0.5094320068939977, 'epoch': 10.99} - overclocked GPU, ~60-65W 

4. QLORA OASTT2+UltraChat (17k records), (qlora_oastt2/out_qlora-20240408004646){'train_runtime': 85006.9413, 'train_samples_per_second': 1.072, 'train_steps_per_second': 0.268, 'train_loss': 1.0263265134591717, 'epoch': 5.0} - ~60-65W, VRAM ~8GB, original ETA 24h (at start) - actual 23.6 (4.7h per epoch)

5. QLORA OASST2, 1 epoch, batch size 1, (qlora_oastt2\out_qlora-20240409132122) {'train_runtime': 996.9661, 'train_samples_per_second': 1.662, 'train_steps_per_second': 0.831, 'train_loss': 1.5574734058357091, 'epoch': 1.0}, GPU ~82W, VRAM ~6.5GB

6. QLORA OASST2, 1 epoch, batch size 2, (qlora_oastt2\out_qlora-20240409134318) {'train_runtime': 1510.0385, 'train_samples_per_second': 1.097, 'train_steps_per_second': 0.274, 'train_loss': 1.5549839297354509, 'epoch': 1.0} , GPU ~62W, VRAM ~8,4GB

7. QLORA OASST2, 1 epoch, batch size 1, SDPA attention, {'train_runtime': 1002.554, 'train_samples_per_second': 1.653, 'train_steps_per_second': 0.826, 'train_loss': 1.5570363735663142, 'epoch': 1.0} (qlora_oastt2\out_qlora-20240409145227) GPU ~82W, VRAM ~6.5GB

## Galore

0. batch size 16, VRAM overflow, 100W updating proj gap (every 200 steps), 42W in between
1. (galore_oastt2\out_galore-20240401135312) batch size 12, still VRAM overflow, 100W updating proj gap (every 200 steps), 60W in between - {'train_runtime': 4243.6437, 'train_samples_per_second': 2.828, 'train_steps_per_second': 0.236, 'train_loss': 1.24991813248503, 'epoch': 3.0}
2. (galore_oastt2\out_galore-20240401152709) batch size 8, still VRAM overflow (~0.7GB), 100W updating proj gap (every 200 steps), 80W in between - {'train_runtime': 3409.4654, 'train_samples_per_second': 3.52, 'train_steps_per_second': 0.44, 'train_loss': 1.2464010416666667, 'epoch': 3.0}
3. (galore_oastt2\out_galore-20240401164153) batch size 8, 8 bit qunatization + LORA adapter, ~3GB of VRAM still available - seems like model can't be quantized with Galore {'train_runtime': 2610.4473, 'train_samples_per_second': 4.597, 'train_steps_per_second': 0.575, 'train_loss': 1.6119010416666666, 'epoch': 3.0}

## QLora

1. (qlora_oastt2\out_qlora-20240401211220), batch size 8,  8 bit qunatization + LORA adapter, - same poor train/loss as with Galore #4 'train_runtime': 2581.5556, 'train_samples_per_second': 4.648, 'train_steps_per_second': 0.581, 'train_loss': 1.6769557291666666, 'epoch': 3.0}

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

3. (qlora_oastt2\out_qlora-20240402101114) batch size 4, new LORA config, optim="paged_adamw_8bit", 55W - stuck at step 250 of 399 (?doing gradient chekpointing), strange spikles in train/loss following steep decline

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

4. (qlora_oastt2\out_qlora-20240402180539) new configs from https://www.philschmid.de/fine-tune-llms-in-2024-with-trl, 85W {'train_runtime': 4605.958, 'train_samples_per_second': 4.215, 'train_steps_per_second': 0.702, 'train_loss': 1.457901078504991, 'epoch': 3.0} - after each epoch there was an abrupt decline in train/loss going below 1.0 at the end

5. (qlora_oastt2\out_qlora-20240402225225) 1024 max_seq_length(vs. s256), 4 epochs - 70-85W (vs. 3) {'train_runtime': 6469.4586, 'train_samples_per_second': 1.0, 'train_steps_per_second': 0.25, 'train_loss': 1.2151240456923105, 'epoch': 4.0}      

6. (qlora_oastt2\out_qlora-20240403193153) same as #5 + flash_attention_2, batch 1, ~80W {'train_runtime': 4431.736, 'train_samples_per_second': 1.46, 'train_steps_per_second': 0.73, 'train_loss': 1.1438023438964993, 'epoch': 4.0}

7. (qlora_oastt2/out_qlora-20240404164313/checkpoint-3244) same as #6, GPU overclock, ~85W {'train_runtime': 3977.5122, 'train_samples_per_second': 1.631, 'train_steps_per_second': 0.816, 'train_loss': 1.151490448490612, 'epoch': 4.0}

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