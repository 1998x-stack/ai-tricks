"""远程 Hugging Face 因果语言模型 SFT。仅实现明确列出的真实训练干预。"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from experiments.utils.hf_stream import ids_digest, require_remote, stream
from experiments.utils.runtime import file_hash, read_json, write_json

SUPPORTED_AXES = {
    'optimizer.lr', 'optimizer.weight_decay', 'adapter.rank', 'adapter.alpha',
    'adapter.dropout', 'adapter.target_modules', 'training.epochs', 'training.batch_size',
    'training.accumulation_steps', 'training.clip_norm', 'training.gradient_checkpointing',
    'training.precision', 'data.max_length', 'loss.assistant_only',
}


def tokenize_example(row, tokenizer, config, message_column):
    messages = row[message_column]
    if not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('SFT 样本必须以 assistant 回复结束，不静默丢弃非法样本')
    # 按对话前缀长度划分每个 assistant span；严格验证模板前缀一致性。
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    labels = [-100] * len(ids)
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        prompt = tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=True)
        completed = tokenizer.apply_chat_template(messages[:i+1], tokenize=True, add_generation_prompt=False)
        if completed[:len(prompt)] != prompt or ids[:len(completed)] != completed:
            raise ValueError('模板不满足前缀对齐；需要该模型专用 assistant mask，不猜测边界')
        labels[len(prompt):len(completed)] = ids[len(prompt):len(completed)]
    if not config.get('loss.assistant_only',True):
        labels = list(ids)
    maximum = config.get('data.max_length',2048)
    # 固定从左侧截断，优先保留最终 assistant 回复；在数据协议中披露。
    ids, labels = ids[-maximum:], labels[-maximum:]
    if not any(value != -100 for value in labels[1:]):
        raise ValueError('截断后没有可监督的回复 token；请调整 max_length 或预注册过滤规则')
    return ids, labels


def collate(records, tokenizer, config, message_column, device):
    import torch
    encoded = [tokenize_example(row,tokenizer,config,message_column) for row in records]
    length = max(len(x) for x,_ in encoded)
    ids,labels,masks = [],[],[]
    for tokens,targets in encoded:
        padding = length-len(tokens)
        ids.append(tokens+[tokenizer.pad_token_id]*padding)
        labels.append(targets+[-100]*padding)
        masks.append([1]*len(tokens)+[0]*padding)
    return {'input_ids':torch.tensor(ids,device=device),
            'labels':torch.tensor(labels,device=device),'attention_mask':torch.tensor(masks,device=device)}


def train(request, output):
    profile, config = request['profile'],request['config']
    require_remote(profile)
    unsupported = set(config)-SUPPORTED_AXES
    if unsupported:
        raise ValueError(f'SFT 驱动未实现干预: {sorted(unsupported)}')
    import numpy as np
    import torch
    from itertools import islice
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from rouge_score.rouge_scorer import RougeScorer

    if not torch.cuda.is_available():
        raise RuntimeError('SFT 完整训练需要远程 CUDA GPU')
    seed = request['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    precision = config.get('training.precision','bf16')
    if precision not in ('bf16','fp32','fp16'):
        raise ValueError('不支持的计算精度')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU 不支持 BF16')
    dtype = {'bf16':torch.bfloat16,'fp32':torch.float32,'fp16':torch.float16}[precision]
    model_ref = profile['model']
    tokenizer = AutoTokenizer.from_pretrained(model_ref['id'],revision=model_ref['revision'],trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError('模型未提供 chat template；不得替换成任意模板')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 保留 FP32 主参数；autocast 决定计算精度，FP16 使用 GradScaler。
    model = AutoModelForCausalLM.from_pretrained(model_ref['id'],revision=model_ref['revision'],
                                                torch_dtype=torch.float32,trust_remote_code=False).cuda()
    if profile.get('finetune_mode','lora') == 'lora':
        adapter = LoraConfig(r=config.get('adapter.rank',8),lora_alpha=config.get('adapter.alpha',16),
                             lora_dropout=config.get('adapter.dropout',0.),
                             target_modules=config.get('adapter.target_modules',['q_proj','v_proj']),
                             bias='none',task_type='CAUSAL_LM')
        model = get_peft_model(model,adapter)
    elif any(key.startswith('adapter.') for key in config):
        raise ValueError('全量微调模式不能配置 LoRA 干预')
    if config.get('training.gradient_checkpointing',False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config.get('optimizer.lr',.0001),
                                  weight_decay=config.get('optimizer.weight_decay',.01))
    scaler = torch.amp.GradScaler('cuda',enabled=precision=='fp16')
    budget = profile['budget']
    epochs = config.get('training.epochs',budget['epochs'])
    batch_size = config.get('training.batch_size',4)
    accumulation = config.get('training.accumulation_steps',1)
    columns = profile['columns']['messages']
    step, seen, tokens_seen = 0, 0, 0
    best, best_step = float('inf'), 0
    checkpoint = output/'best_trainable.pt'
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def validation():
        model.eval()
        loss_sum,total = 0.,0
        records = iter(stream(profile,'validation'))
        with torch.no_grad():
            while rows := list(islice(records,batch_size)):
                batch = collate(rows,tokenizer,config,columns,'cuda')
                tokens = batch['labels'][:,1:].ne(-100).sum().item()
                with torch.autocast('cuda',dtype=dtype,enabled=precision!='fp32'):
                    loss = model(**batch).loss
                loss_sum += float(loss)*tokens
                total += tokens
        if total == 0:
            raise ValueError('验证集无有效监督 token')
        return loss_sum/total

    with (output/'history.csv').open('w',newline='') as log:
        writer = csv.DictWriter(log,fieldnames=['step','train_loss','val_loss','examples_seen','tokens_seen','seconds'])
        writer.writeheader()
        for epoch in range(epochs):
            records = iter(stream(profile,'train',seed+epoch))
            while True:
                rows = list(islice(records,batch_size*accumulation))
                if not rows:
                    break
                if step >= budget['max_steps']:
                    raise RuntimeError('预算耗尽但预注册完整 epoch 未完成；不生成成功结果')
                model.train()
                optimizer.zero_grad(set_to_none=True)
                minibatches = [collate(rows[i:i+batch_size],tokenizer,config,columns,'cuda')
                               for i in range(0,len(rows),batch_size)]
                token_counts = [int(b['labels'][:,1:].ne(-100).sum()) for b in minibatches]
                total_tokens = sum(token_counts)
                train_loss = 0.
                # 按有效 token 数加权，避免变长序列/尾批造成梯度缩放错误。
                for batch,count in zip(minibatches,token_counts):
                    with torch.autocast('cuda',dtype=dtype,enabled=precision!='fp32'):
                        loss = model(**batch).loss * count/total_tokens
                    if not torch.isfinite(loss):
                        raise FloatingPointError('SFT loss 非有限')
                    train_loss += float(loss.detach())
                    scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),config.get('training.clip_norm',1.) or float('inf'),error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                seen += len(rows)
                tokens_seen += total_tokens
                if step % budget['eval_every_steps'] == 0:
                    val_loss = validation()
                    if val_loss < best:
                        best,best_step = val_loss,step
                        trainable = {name:p.detach().cpu() for name,p in model.named_parameters() if p.requires_grad}
                        torch.save(trainable,checkpoint)
                    writer.writerow({'step':step,'train_loss':train_loss,'val_loss':val_loss,
                                     'examples_seen':seen,'tokens_seen':tokens_seen,'seconds':time.monotonic()-started})
                    log.flush()
                    print(f'step={step} train_loss={train_loss:.6g} val_loss={val_loss:.6g}',flush=True)
        if step == 0:
            raise ValueError('训练集为空或 epoch 预算为零')
        # 最后一个不足 eval interval 的 checkpoint 也必须验证。
        val_loss = validation()
        if val_loss < best:
            best,best_step = val_loss,step
            torch.save({name:p.detach().cpu() for name,p in model.named_parameters() if p.requires_grad},checkpoint)
        writer.writerow({'step':step,'train_loss':train_loss,'val_loss':val_loss,
                         'examples_seen':seen,'tokens_seen':tokens_seen,'seconds':time.monotonic()-started})
    training_seconds,peak = time.monotonic()-started,torch.cuda.max_memory_allocated()
    loaded = torch.load(checkpoint,map_location='cpu',weights_only=True)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:
                p.copy_(loaded[name])
    model.eval()
    identities,exact,rouge_total = [],0,0.
    scorer = RougeScorer(['rougeL'],use_stemmer=False)
    with (output/'predictions.jsonl').open('w') as predictions, torch.no_grad():
        for row in stream(profile,'test'):
            messages = row[columns]
            if messages[-1]['role'] != 'assistant':
                raise ValueError('测试集格式错误')
            prompt = tokenizer.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True,return_tensors='pt').cuda()
            if prompt.shape[1] + budget['max_new_tokens'] > profile['context_limit']:
                raise ValueError('测试 prompt 超出模型上下文；不得静默删除样本')
            generated = model.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),
                                       max_new_tokens=budget['max_new_tokens'],do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(generated[0,prompt.shape[1]:],skip_special_tokens=True).strip()
            target = messages[-1]['content'].strip()
            exact += response == target
            rouge_total += scorer.score(target,response)['rougeL'].fmeasure
            identities.append(row['_example_id'])
            predictions.write(json.dumps({'id':row['_example_id'],'reference':target,'prediction':response},ensure_ascii=False)+'\n')
    if not identities:
        raise ValueError('测试集为空')
    result = {key:request[key] for key in ('experiment_id','plan_hash','seed','arm','dataset_revision','model_revision')}
    result.update({'applied_config':config,'data_kind':'huggingface_real','train_examples_seen':seen,
                   'optimizer_steps':step,'test_ids_sha256':ids_digest(identities),'tokens_seen':tokens_seen,
                   'metrics':{'test_exact_match':exact/len(identities),'test_rougeL':rouge_total/len(identities),'val_loss':best,
                              'peak_cuda_memory_bytes':peak,'examples_per_second':seen/training_seconds},
                   'metric_limitation':'ROUGE-L 衡量参考文本重叠，不代表对话正确性、安全性或人工偏好；需另加任务评测',
                   'truncation_policy':'left_keep_final_response'})
    write_json(output/'checkpoint_manifest.json',{'selected_by_validation':True,'selection_metric':'val_loss',
               'step':best_step,'path':checkpoint.name,'sha256':file_hash(checkpoint),
               'base_model':model_ref,'tokenizer_revision':model_ref['revision']})
    write_json(output/'result.json',result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    train(read_json(args.request),args.output)
