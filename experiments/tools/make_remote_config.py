"""生成没有主机授权/计算预算的远程配置示例；仅引用已核对的 HF 元数据。"""
from experiments.drivers.sft import SUPPORTED_AXES as SFT_AXES
from experiments.drivers.vision import SUPPORTED_AXES as VISION_AXES
from experiments.utils.runtime import ROOT, file_hash, read_json, write_json


def build():
    data = {x['id']: x for x in read_json(ROOT/'experiments/catalog/hf_datasets.json')}
    model = read_json(ROOT/'experiments/catalog/hf_models.json')[0]
    execution = {'allowed_hostnames': [], 'remote_training_confirmed': False,
                 'max_total_seconds': 0, 'max_run_seconds': 0, 'hf_cache': '/workspace/hf-cache'}
    vision = {
        'dataset': data['uoft-cs/cifar10'],
        'model': {'source':'code','architecture':'cifar_resnet18',
                  'path':'experiments/drivers/vision_model.py',
                  'revision':file_hash(ROOT/'experiments/drivers/vision_model.py')},
        'streaming': True, 'columns': {'image':'img','label':'label'}, 'train_count':45000,
        'splits': {'train':{'source':'train','modulo':10,'remainder':0,'keep_equal':False},
                   'validation':{'source':'train','modulo':10,'remainder':0},
                   'test':{'source':'test'}},
        'baseline': {'optimizer.kind':'sgd','optimizer.lr':.1,'optimizer.momentum':.9,
                     'optimizer.weight_decay':.0005,'training.batch_size':128,
                     'model.norm':'batch','model.activation':'relu'},
        'budget': {'epochs':200,'max_steps':0,'eval_every_examples':45000},
        'driver': {'kind':'real_training','argv':['{python}','-m','experiments.drivers.vision'],
                   'supported_axes':sorted(VISION_AXES)},
    }
    sft = {
        'dataset': data['HuggingFaceH4/ultrachat_200k'], 'model': model,
        'streaming':True, 'columns':{'messages':'messages'}, 'finetune_mode':'lora',
        'context_limit':32768,
        'splits': {'train':{'source':'train_sft'},
                   'validation':{'source':'test_sft','modulo':2,'remainder':0},
                   'test':{'source':'test_sft','modulo':2,'remainder':1}},
        'baseline': {'optimizer.lr':.0001,'optimizer.weight_decay':.01,'adapter.rank':8,
                     'adapter.alpha':16,'adapter.dropout':0.,'training.batch_size':4,
                     'training.accumulation_steps':4,'training.precision':'bf16',
                     'data.max_length':2048,'loss.assistant_only':True},
        'budget':{'epochs':3,'max_steps':0,'eval_every_steps':500,'max_new_tokens':256},
        'driver':{'kind':'real_training','argv':['{python}','-m','experiments.drivers.sft'],
                  'supported_axes':sorted(SFT_AXES)},
    }
    config = {'execution':execution,'profiles':{'vision':vision,'sft':sft},
              'unimplemented_profiles':['nlp','preference','reasoning_rl','recommendation',
                                        'retrieval','detection','segmentation'],
              'note':'示例不是可直接执行的资源授权；尚未在 GPU 上验证。其他 profile 必须实现专用驱动。'}
    write_json(ROOT/'experiments/configs/remote.example.json',config)


if __name__ == '__main__':
    build()
