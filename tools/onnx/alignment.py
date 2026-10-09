"""Export identical ONNX inputs for WebGPU, then compare raw backend outputs."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(value):
    value = np.ascontiguousarray(value)
    assert value.dtype in (np.dtype('float16'), np.dtype('float32'))
    return dict(dtype=str(value.dtype), shape=list(value.shape), data=base64.b64encode(value.tobytes()).decode())


def unpack(value):
    if value['dtype'] not in ('float16', 'float32'):
        raise ValueError('Unsupported tensor dtype')
    return np.frombuffer(base64.b64decode(value['data'], validate=True), dtype=np.dtype(value['dtype']).newbyteorder('<')).reshape(value['shape'])


def export(args):
    import torch
    import torch.nn.functional as F
    import onnxruntime as ort
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    ort.preload_dlls()
    manifest = json.loads((args.model_dir/'pixrestore-manifest.json').read_text())
    sampling = manifest['singleStepSampling']
    if not sampling['usesAelq'] or sampling['lqConditionStrength'] != 1 or sampling['dinoConditionStrength'] != 1:
        raise ValueError('Only single-step exports with both condition strengths equal to 1 are supported')
    dtype = np.float16 if manifest['precision'] == 'fp16' else np.float32
    size = manifest['resolution']
    with Image.open(args.image) as image:
        lq = pil_to_tensor(image.convert('RGB').resize((size,size), Image.Resampling.BICUBIC)).unsqueeze(0).float()/127.5-1
    torch.manual_seed(42)
    pixel = torch.cat([lq,torch.randn_like(lq)],1).numpy().astype(dtype)
    dino_image = F.interpolate((lq+1)*.5, size=(manifest['dino']['inputSize'],)*2, mode='bicubic', align_corners=False, antialias=True).clamp(0,1)
    norm = manifest['dino']['input']['preprocessing']['normalization']
    dino_image = (dino_image-torch.tensor(norm['mean'])[None,:,None,None])/torch.tensor(norm['std'])[None,:,None,None]
    inputs = dict(image=dino_image.numpy().astype(dtype), pixel_input=pixel, timestep=np.array([manifest['pixelDit']['timestep']['value']],np.float32))
    models = {}
    sessions = {}
    for key, model_key in [('dino','dinoEncoder'),('pixel','pixelDit')]:
        path = args.model_dir/manifest['models'][model_key]
        models[key] = dict(file=path.name,sha256=sha(path))
        options = ort.SessionOptions()
        options.log_severity_level = 3
        sessions[key] = ort.InferenceSession(str(path), sess_options=options, providers=['CUDAExecutionProvider'])
        if sessions[key].get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('CUDA provider did not initialize')
    dino = sessions['dino']
    features = dict(zip([o.name for o in dino.get_outputs()],dino.run(None,dict(image=inputs['image']))))
    prediction = sessions['pixel'].run(None,dict(pixel_input=pixel,timestep=inputs['timestep'],**features))[0]
    for array in [*features.values(),prediction]:
        if not np.isfinite(array).all(): raise ValueError('Non-finite CUDA output')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    bundle = dict(format='pixrestore-alignment-inputs-v1', models=models, inputs={k:pack(v) for k,v in inputs.items()},features={k:pack(v) for k,v in features.items()})
    input_path = args.output_dir/'alignment-inputs.json'
    input_path.write_text(json.dumps(bundle),encoding='utf-8')
    outputs = {f'dino/{k}':pack(v) for k,v in features.items()}
    outputs.update({'pixel_fixed/x_pred':pack(prediction),'pipeline/x_pred':pack(prediction)})
    reference = dict(format='pixrestore-alignment-results-v1',input_sha256=sha(input_path),models=models,outputs=outputs,ort_version=ort.__version__,gpu=torch.cuda.get_device_name())
    (args.output_dir/'alignment-cuda.json').write_text(json.dumps(reference),encoding='utf-8')
    print(f'Inputs: {input_path}\nReference: {args.output_dir / "alignment-cuda.json"}')


def reference(args):
    # Reuse the exact existing bundle, including its fixed CUDA DINO features.
    import torch
    import onnxruntime as ort
    ort.preload_dlls()
    bundle = json.loads(args.inputs.read_text(encoding='utf-8'))
    if bundle['format'] != 'pixrestore-alignment-inputs-v1':
        raise ValueError('Invalid input bundle')
    inputs = {k:unpack(v) for k,v in bundle['inputs'].items()}
    fixed_features = {k:unpack(v) for k,v in bundle['features'].items()}
    sessions = {}
    for name in ('dino', 'pixel'):
        spec = bundle['models'][name]
        path = args.model_dir/spec['file']
        if sha(path) != spec['sha256']: raise ValueError(f'Model hash mismatch: {name}')
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL if args.graph_optimization == 'disabled' else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sessions[name] = ort.InferenceSession(str(path), sess_options=options, providers=['CUDAExecutionProvider'])
        if sessions[name].get_providers()[0] != 'CUDAExecutionProvider': raise RuntimeError('CUDA provider unavailable')
    dino = sessions['dino']
    features = dict(zip([o.name for o in dino.get_outputs()], dino.run(None, {'image':inputs['image']})))
    base = dict(pixel_input=inputs['pixel_input'], timestep=inputs['timestep'])
    fixed_names = [o.name for o in sessions['pixel'].get_outputs()]
    fixed = dict(zip(fixed_names, sessions['pixel'].run(None, dict(**base, **fixed_features))))
    pipeline = sessions['pixel'].run(['x_pred'], dict(**base, **{k:features[k] for k in fixed_features}))[0]
    arrays = {f'dino/{k}':v for k,v in features.items()}
    arrays.update({f'pixel_fixed/{k}':v for k,v in fixed.items()})
    arrays['pipeline/x_pred'] = pipeline
    for name, value in arrays.items():
        if not np.isfinite(value).all(): raise ValueError(f'Non-finite output: {name}')
    result = dict(format='pixrestore-alignment-results-v1', input_sha256=sha(args.inputs), models=bundle['models'],
        outputs={k:pack(v) for k,v in arrays.items()}, graph_optimization=args.graph_optimization,
        ort_version=ort.__version__, gpu=torch.cuda.get_device_name())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result), encoding='utf-8')
    print(f'Reference ({args.graph_optimization}): {args.output}')


def compare(args):
    from PIL import Image
    reference=json.loads(args.reference.read_text(encoding='utf-8'))
    actual=json.loads(args.actual.read_text(encoding='utf-8'))
    for key in ('format','input_sha256','models'):
        if reference[key] != actual[key]: raise ValueError(f'Mismatch: {key}')
    if reference.get('graph_optimization', 'all') != actual.get('graph_optimization', 'all'):
        raise ValueError('Graph optimization settings differ')
    if reference['outputs'].keys() != actual['outputs'].keys(): raise ValueError('Output keys differ')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={}
    for name, packed in reference['outputs'].items():
        other=actual['outputs'][name]
        if packed['dtype'] != other['dtype'] or packed['shape'] != other['shape']: raise ValueError(f'Metadata mismatch: {name}')
        a,b=unpack(packed).astype(np.float32),unpack(other).astype(np.float32)
        if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError(f'Non-finite output: {name}')
        delta=np.abs(a-b)
        report[name]=dict(mae=float(delta.mean()),max_abs=float(delta.max()),rmse=float(np.sqrt(np.mean((a-b)**2))), reference_rms=float(np.sqrt(np.mean(a*a))), relative_rmse=float(np.sqrt(np.mean((a-b)**2))/max(float(np.sqrt(np.mean(a*a))),1e-12)))
        print(name, report[name])
        if name.endswith('x_pred'):
            # PNGs aid inspection; metrics above use raw tensors before clipping/quantization.
            panels=[np.clip((a[0]+1)*127.5,0,255),np.clip((b[0]+1)*127.5,0,255),np.clip(delta[0]*127.5*8,0,255)]
            image=np.concatenate([np.rint(p).astype(np.uint8).transpose(1,2,0) for p in panels],axis=1)
            Image.fromarray(image).save(args.output_dir/(name.replace('/','_')+'_cuda_webgpu_diff8x.png'))
    (args.output_dir/'errors.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('export')
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p=sub.add_parser('reference')
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--graph-optimization',choices=['all','disabled'],default='all')
    p=sub.add_parser('compare')
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--actual',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    {'export':export, 'reference':reference, 'compare':compare}[args.command](args)
