"""Read-only hash-pinned access to the retained step400 artifact."""
import json
from pathlib import Path
from safetensors.torch import load_file
from .deployment import FIELDS
from .evaluation import sha256_file
from .qwen3 import QWEN3_LINEAR_MODULES, expected_qwen3_linear_shape
from .qwen3_8b import ARCHITECTURE

MANIFEST_SHA='253ab448797ef4d798522875014b7e47a4edb84c5c3c1ca5cf6179edfd339fec'


def load_accepted_layer(root, layer):
    manifest_path=root/'manifest.json'
    if sha256_file(manifest_path)!=MANIFEST_SHA:raise ValueError('step400 manifest drift')
    manifest=json.loads(manifest_path.read_text())
    if not 0<=layer<36:raise ValueError('layer outside 0..35')
    relative=f'payloads/layer-{layer:03d}.safetensors'
    entry=next(e for e in manifest['files'] if e['path']==relative)
    path=root/relative
    if path.stat().st_size!=entry['bytes'] or sha256_file(path)!=entry['sha256']:
        raise ValueError('layer payload hash/size drift')
    tensors=load_file(path)
    expected={name+'.'+field for name in QWEN3_LINEAR_MODULES for field in FIELDS}
    if set(tensors)!=expected:raise ValueError('layer inventory drift')
    for name in QWEN3_LINEAR_MODULES:
        shape=tensors[name+'.global_sign_codes'].shape
        if [shape[0],shape[1]*4]!=expected_qwen3_linear_shape(ARCHITECTURE,name):
            raise ValueError(f'8B shape drift: {name}')
    return tensors,entry



def replace_model_linears(model, artifact_root: Path, *, allow_prefill_fallback=False,
                          groups_per_split=8, kernel="v1"):
    """Prepared HF full-model adapter, to be exercised only after block acceptance.

    No download, scheduler, attention or KV-cache changes. Default is strict M=1;
    enabling prefill explicitly uses slow on-demand dense materialization.
    Returned provenance contains exact coverage. No serving/TP support is implied.
    """
    from .deployment import replace_block_linears, validate_artifact, KERNELS, workspace_shape
    from torch import nn
    from .qwen3_8b import validate_architecture
    validate_architecture(model.config.to_dict())
    if len(model.model.layers)!=36:raise ValueError('expected 36 decoder layers')
    if kernel not in KERNELS:raise ValueError('unknown kernel')
    workspace_shape(1,128,groups_per_split)
    # Complete payload and target validation before starting replacement.
    records=[]
    for layer in range(36):
        payload,entry=load_accepted_layer(artifact_root,layer)
        for name in QWEN3_LINEAR_MODULES:
            o,g=validate_artifact({field:payload[name+'.'+field] for field in FIELDS})
            target=model.model.layers[layer].get_submodule(name)
            if not isinstance(target,nn.Linear) or (target.out_features,target.in_features)!=(o,g*128):
                raise ValueError(f'Linear shape/type mismatch: layer {layer} {name}')
        records.append(entry)
    del payload
    coverage=[]
    for layer,block in enumerate(model.model.layers):
        payload,_=load_accepted_layer(artifact_root,layer)
        modules=replace_block_linears(block,payload,
            fallback='dense' if allow_prefill_fallback else 'error',groups_per_split=groups_per_split,kernel=kernel)
        coverage.extend(f'model.layers.{layer}.{m}' for m in modules)
    return {'format': 'fluxbin-hybrid-g128-s8-m1-v1', 'manifest_sha256':MANIFEST_SHA,
            'layers':records,'coverage':coverage,'linear_count':len(coverage),
            'kernel':kernel,'groups_per_split':groups_per_split,
            'prefill_fallback_enabled':allow_prefill_fallback,'status':'installed_not_validated'}
