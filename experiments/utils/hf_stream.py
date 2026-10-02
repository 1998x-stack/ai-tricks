"""远程 Hugging Face 流式读取；必须先通过运行位置检查。"""
import hashlib
import platform
import socket


def require_remote(profile):
    if platform.system() != 'Linux' or socket.gethostname() not in profile.get('allowed_hostnames', []):
        raise RuntimeError('拒绝在本机读取 Hugging Face 数据/模型；仅可使用 --plan')


def stream(profile, split, seed=None):
    require_remote(profile)
    from datasets import load_dataset

    selector = profile['splits'][split]
    data = profile['dataset']
    source = load_dataset(data['id'], name=data.get('config'), revision=data['revision'],
                          split=selector['source'], streaming=True)
    source = source.map(lambda row, index: {'_example_id': f"{selector['source']}:{index}"},
                        with_indices=True)
    if 'modulo' in selector:
        modulus, remainder = selector['modulo'], selector['remainder']
        keep_equal = selector.get('keep_equal', True)
        source = source.filter(lambda row, index: ((index % modulus == remainder) == keep_equal),
                               with_indices=True)
    if seed is not None:
        source = source.shuffle(seed=seed, buffer_size=profile.get('shuffle_buffer', 10000))
    return source


def ids_digest(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
