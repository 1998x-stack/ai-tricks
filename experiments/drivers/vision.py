"""远程真实 CIFAR-10 训练；没有合成数据回退，没有默认截取几十个样本。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from itertools import islice
from pathlib import Path

from experiments.drivers.capabilities import MODEL_AXES
from experiments.utils.hf_stream import ids_digest, require_remote, stream
from experiments.utils.runtime import file_hash, read_json, write_json

SUPPORTED_AXES = MODEL_AXES | {
    'optimizer.kind', 'optimizer.lr', 'optimizer.weight_decay', 'optimizer.momentum',
    'optimizer.beta1', 'optimizer.beta2', 'optimizer.schedule', 'optimizer.gamma',
    'optimizer.warmup_ratio', 'optimizer.min_lr', 'training.batch_size',
    'training.accumulation_steps', 'training.clip_norm', 'training.epochs',
    'training.patience', 'training.max_weight_norm', 'training.precision',
    'loss.label_smoothing', 'loss.l1', 'loss.flooding', 'loss.kind',
    'data.standardize', 'data.shuffle', 'data.augmentation', 'data.brightness_jitter',
    'checkpoint.selection_metric',
}


def validate_config(config):
    unknown = set(config) - SUPPORTED_AXES
    if unknown:
        raise ValueError(f'真实视觉驱动没有实现这些字段: {sorted(unknown)}')
    if config.get('data.augmentation', 'none') not in ('none','horizontal_flip','random_crop','label_preserving'):
        raise ValueError('此驱动只实现水平翻转/裁剪增强，不冒充 Mixup/CutMix')
    if config.get('loss.kind', 'cross_entropy') not in ('cross_entropy','focal','cross_entropy_double_softmax'):
        raise ValueError('此驱动尚未实现所选 loss')
    if config.get('checkpoint.selection_metric','val_loss') not in ('val_loss','val_accuracy'):
        raise ValueError('checkpoint 选择指标未实现')
    schedules = ('constant','cosine','warmup_cosine','step','multistep','exponential','plateau_half','cosine_restarts','cyclic')
    if config.get('optimizer.schedule','constant') not in schedules:
        raise ValueError('未实现的学习率策略')


def batches(profile, split, config, epoch, seed):
    import numpy as np
    import torch
    from PIL import ImageEnhance, ImageOps

    train = split == 'train'
    order_seed = seed + epoch if train and config.get('data.shuffle', True) else None
    source = iter(stream(profile, split, order_seed))
    size = config.get('training.batch_size', 128) if train else 256
    rng = random.Random(seed + epoch)
    while True:
        records = list(islice(source, size))
        if not records:
            break
        images, labels, ids = [], [], []
        for row in records:
            picture = row[profile['columns']['image']].convert('RGB')
            if train:
                aug = config.get('data.augmentation','none')
                if aug in ('horizontal_flip','label_preserving') and rng.random() < .5:
                    picture = ImageOps.mirror(picture)
                if aug in ('random_crop','label_preserving'):
                    picture = ImageOps.expand(picture, border=4)
                    x, y = rng.randint(0, 8), rng.randint(0, 8)
                    picture = picture.crop((x, y, x + 32, y + 32))
                brightness = config.get('data.brightness_jitter', 0.0)
                if brightness:
                    picture = ImageEnhance.Brightness(picture).enhance(rng.uniform(1-brightness, 1+brightness))
            array = np.asarray(picture, dtype=np.float32).copy() / 255.
            if config.get('data.standardize', True):
                array = (array - np.array([.4914,.4822,.4465], dtype=np.float32)) / np.array([.247,.243,.261], dtype=np.float32)
            images.append(torch.from_numpy(array.transpose(2,0,1)))
            labels.append(int(row[profile['columns']['label']]))
            ids.append(row['_example_id'])
        yield torch.stack(images).cuda(), torch.tensor(labels, device='cuda'), ids


def criterion(logits, labels, config):
    import torch.nn.functional as F
    kind = config.get('loss.kind', 'cross_entropy')
    if kind == 'cross_entropy_double_softmax':
        logits = logits.softmax(-1)
    losses = F.cross_entropy(logits, labels, reduction='none',
                             label_smoothing=config.get('loss.label_smoothing',0.0))
    if kind == 'focal':
        pt = logits.softmax(-1).gather(1, labels[:,None]).squeeze(1)
        losses = (1-pt).square() * losses
    return losses


def optimizer_for(model, config, switched=False):
    import torch
    kind = config.get('optimizer.kind','sgd')
    if kind == 'adam_then_sgd':
        kind = 'sgd' if switched else 'adam'
    args = {'lr':config.get('optimizer.lr',.1), 'weight_decay':config.get('optimizer.weight_decay',.0005)}
    if kind == 'sgd':
        return torch.optim.SGD(model.parameters(), momentum=config.get('optimizer.momentum',.9), **args)
    if kind in ('adam','adamw'):
        cls = torch.optim.Adam if kind == 'adam' else torch.optim.AdamW
        return cls(model.parameters(), betas=(config.get('optimizer.beta1',.9),config.get('optimizer.beta2',.999)), **args)
    if kind == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), **args)
    raise ValueError(f'优化器未实现: {kind}')


def rate_at(step, steps, config, plateau_scale):
    progress = step / max(steps, 1)
    warm = config.get('optimizer.warmup_ratio',.1)
    schedule = config.get('optimizer.schedule','constant')
    if schedule == 'warmup_cosine' and progress < warm:
        factor = (step + 1) / max(warm * steps, 1)
    elif schedule in ('cosine','warmup_cosine','cosine_restarts'):
        phase = ((progress-warm)/(1-warm)) if schedule == 'warmup_cosine' else progress
        if schedule == 'cosine_restarts':
            phase = (phase * 3) % 1
        factor = .5 * (1 + math.cos(math.pi * phase))
    elif schedule == 'step':
        factor = config.get('optimizer.gamma',.1) ** int(progress * 3)
    elif schedule == 'multistep':
        factor = config.get('optimizer.gamma',.1) ** (int(progress >= .5)+int(progress >= .75))
    elif schedule == 'exponential':
        factor = config.get('optimizer.gamma',.99) ** step
    elif schedule == 'cyclic':
        factor = .1 + .9 * (1-abs(2*((progress*3)%1)-1))
    elif schedule == 'plateau_half':
        factor = plateau_scale
    else:
        factor = 1.
    base, minimum = config.get('optimizer.lr',.1), config.get('optimizer.min_lr',0.)
    return minimum + (base-minimum)*factor


def evaluate(model, profile, config, split, seed, prediction_path=None):
    import torch
    import torch.nn.functional as F
    model.eval()
    total, correct, loss_sum, ids = 0, 0, 0., []
    handle = Path(prediction_path).open('w') if prediction_path else None
    try:
        with torch.no_grad():
            for images, labels, batch_ids in batches(profile, split, config, 0, seed):
                logits = model(images)
                loss_sum += F.cross_entropy(logits, labels, reduction='sum').item()
                predictions = logits.argmax(-1)
                total += len(labels)
                correct += predictions.eq(labels).sum().item()
                ids.extend(batch_ids)
                if handle:
                    for identity, target, prediction in zip(batch_ids, labels.tolist(), predictions.tolist()):
                        handle.write(json.dumps({'id':identity,'label':target,'prediction':prediction})+'\n')
    finally:
        if handle:
            handle.close()
    if not total:
        raise ValueError(f'{split} 切分为空')
    return loss_sum/total, correct/total, ids


def train(request, output):
    profile, config = request['profile'], request['config']
    require_remote(profile)  # 位于所有 HF 调用之前，直接调用驱动也不能绕过。
    validate_config(config)
    import numpy as np
    import torch
    from experiments.drivers.vision_model import ResNet

    if not torch.cuda.is_available():
        raise RuntimeError('远程真实训练要求 CUDA GPU')
    random.seed(request['seed'])
    np.random.seed(request['seed'])
    torch.manual_seed(request['seed'])
    torch.cuda.manual_seed_all(request['seed'])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = ResNet(config, profile.get('num_classes',10)).cuda()
    optimizer = optimizer_for(model, config)
    precision = config.get('training.precision','fp32')
    if precision not in ('fp32','bf16','fp16'):
        raise ValueError('不支持的精度')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('远程 GPU 不支持 BF16')
    amp_dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=precision == 'fp16')
    budget = profile['budget']
    epochs = config.get('training.epochs',budget['epochs'])
    accum = config.get('training.accumulation_steps',1)
    steps_per_epoch = math.ceil(profile['train_count'] / (config.get('training.batch_size',128)*accum))
    total_steps = epochs * steps_per_epoch
    if total_steps > budget['max_steps']:
        raise ValueError('max_steps 不足以完成预注册 epoch 预算；请明确资源预算后再运行')
    step, seen, best_step, stale = 0, 0, 0, 0
    eval_interval = budget['eval_every_examples']
    next_evaluation = eval_interval
    best_loss, best_score, plateau_scale = float('inf'), -float('inf'), 1.
    selected_metric = config.get('checkpoint.selection_metric','val_loss')
    checkpoint = output / 'best.pt'
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (output/'history.csv').open('w',newline='') as log:
        writer = csv.DictWriter(log,fieldnames=['step','epoch','train_loss','val_loss','val_accuracy','lr','grad_norm','examples_seen','seconds','cuda_memory_bytes'])
        writer.writeheader()
        for epoch in range(epochs):
            iterator = iter(batches(profile,'train',config,epoch,request['seed']))
            if config.get('optimizer.kind') == 'adam_then_sgd' and epoch == epochs//2:
                optimizer = optimizer_for(model,config,switched=True)
            while step < total_steps:
                group = list(islice(iterator,accum))
                if not group:
                    break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                count = sum(len(y) for _, y, _ in group)
                lr = rate_at(step,total_steps,config,plateau_scale)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                observed_loss = 0.
                for images, labels, _ in group:
                    with torch.autocast('cuda',dtype=amp_dtype,enabled=precision != 'fp32'):
                        losses = criterion(model(images),labels,config)
                        loss = losses.sum()/count
                        observed_loss += loss.detach().item()
                        flooding = config.get('loss.flooding',0.)
                        if flooding:
                            threshold = flooding * len(labels)/count
                            loss = (loss-threshold).abs()+threshold
                        l1 = config.get('loss.l1',0.)
                        if l1:
                            loss = loss + l1 * sum(p.abs().sum() for p in model.parameters())/len(group)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'非有限 loss，step={step}')
                    scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip = config.get('training.clip_norm',0.)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),clip or float('inf'),error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                if config.get('training.max_weight_norm',0.):
                    with torch.no_grad():
                        for parameter in model.parameters():
                            if parameter.ndim > 1:
                                norms = parameter.flatten(1).norm(dim=1).clamp_min(1e-12)
                                scale = (config['training.max_weight_norm']/norms).clamp(max=1)
                                parameter.mul_(scale.view(-1,*([1]*(parameter.ndim-1))))
                step += 1
                seen += count
                if seen >= next_evaluation or step == total_steps:
                    while next_evaluation <= seen:
                        next_evaluation += eval_interval
                    val_loss, val_acc, _ = evaluate(model,profile,config,'validation',request['seed'])
                    score = -val_loss if selected_metric == 'val_loss' else val_acc
                    if score > best_score:
                        best_score, best_loss, best_step, stale = score, val_loss, step, 0
                        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                                    'step':step,'seed':request['seed'],'config':config},checkpoint)
                    else:
                        stale += 1
                        if stale % 5 == 0:
                            plateau_scale *= .5
                    writer.writerow({'step':step,'epoch':epoch,'train_loss':observed_loss,
                                     'val_loss':val_loss,'val_accuracy':val_acc,'lr':lr,
                                     'grad_norm':float(grad_norm),'examples_seen':seen,
                                     'seconds':time.monotonic()-started,
                                     'cuda_memory_bytes':torch.cuda.max_memory_allocated()})
                    log.flush()
                    print(f'step={step} train_loss={observed_loss:.6g} val_loss={val_loss:.6g} val_accuracy={val_acc:.6g}',flush=True)
                    if config.get('training.patience',0) and stale >= config['training.patience']:
                        break
            if step >= total_steps or (config.get('training.patience',0) and stale >= config['training.patience']):
                break
    training_seconds = time.monotonic()-started
    training_peak_memory = torch.cuda.max_memory_allocated()
    if not checkpoint.exists():
        raise RuntimeError('预算内没有完成验证；不能伪造 checkpoint')
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['model'])
    test_loss,test_acc,test_ids = evaluate(model,profile,config,'test',request['seed'],output/'predictions.jsonl')
    write_json(output/'checkpoint_manifest.json',{'selected_by_validation':True,
               'selection_metric':selected_metric,'step':best_step,'path':'best.pt','sha256':file_hash(checkpoint)})
    result = {key:request[key] for key in ('experiment_id','plan_hash','seed','arm','dataset_revision','model_revision')}
    result.update({'applied_config':config,'data_kind':'huggingface_real','train_examples_seen':seen,
                   'optimizer_steps':step,'test_ids_sha256':ids_digest(test_ids),
                   'metrics':{'test_accuracy':test_acc,'test_loss':test_loss,'val_loss':best_loss,
                              'examples_per_second':seen/training_seconds,
                              'peak_cuda_memory_bytes':training_peak_memory},
                   'hardware':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__},
                   'training_seconds':training_seconds})
    write_json(output/'result.json',result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    train(read_json(args.request),args.output)


if __name__ == '__main__':
    main()
