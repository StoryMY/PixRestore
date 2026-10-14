"""Replace exported Pixel DiT attention cores with ORT MultiHeadAttention."""
from pathlib import Path

import onnx
from onnx import helper


def fuse_pixel_attention(path: Path, attention_specs: dict[str, tuple[int, float]]) -> int:
    """Keep projections, QK normalization and RoPE; fuse only QK/softmax/AV.

    Specs map legacy exporter module prefixes to (num_heads, scale). Refuse
    unfamiliar graphs instead of silently exporting partially fused attention.
    """
    model = onnx.load(path)
    nodes = list(model.graph.node)
    by_name = {node.name: node for node in nodes}
    producers = {output: node for node in nodes for output in node.output}
    consumers = {}
    for node in nodes:
        for value in node.input:
            consumers.setdefault(value, []).append(node)

    def require(condition, prefix):
        if not condition:
            raise ValueError(f"Unsupported exported attention graph at {prefix}; fusion aborted")

    def next_node(node, op_type, prefix):
        following = consumers.get(node.output[0], [])
        require(len(following) == 1 and following[0].op_type == op_type, prefix)
        return following[0]

    replacements = {}
    for prefix, (heads, scale) in attention_specs.items():
        score = by_name.get(prefix + 'MatMul')
        require(score is not None and score.op_type == 'MatMul', prefix)
        scaled_q = producers.get(score.input[0])
        transposed_k = producers.get(score.input[1])
        require(scaled_q is not None and scaled_q.op_type == 'Mul'
                and scaled_q.name == prefix + 'Mul', prefix)
        require(transposed_k is not None and transposed_k.op_type == 'Transpose', prefix)
        perm = next((list(a.ints) for a in transposed_k.attribute if a.name == 'perm'), None)
        require(perm == [0, 1, 3, 2], prefix)
        softmax = next_node(score, 'Softmax', prefix)
        axis = next((a.i for a in softmax.attribute if a.name == 'axis'), -1)
        require(axis in (-1, 3), prefix)
        probabilities = softmax
        # CUDA autocast can emit an FP16 -> FP32 -> FP16 round trip here.
        following = consumers.get(probabilities.output[0], [])
        if len(following) == 1 and following[0].op_type == 'Cast':
            upcast = next_node(probabilities, 'Cast', prefix)
            downcast = next_node(upcast, 'Cast', prefix)
            require(next(a.i for a in upcast.attribute if a.name == 'to') == onnx.TensorProto.FLOAT
                    and next(a.i for a in downcast.attribute if a.name == 'to') == onnx.TensorProto.FLOAT16, prefix)
            probabilities = downcast
        weighted_v = next_node(probabilities, 'MatMul', prefix)
        require(weighted_v.input[0] == probabilities.output[0], prefix)
        transpose = next_node(weighted_v, 'Transpose', prefix)
        perm = next((list(a.ints) for a in transpose.attribute if a.name == 'perm'), None)
        require(perm == [0, 2, 1, 3], prefix)
        flattened = next_node(transpose, 'Reshape', prefix)
        # Reuse the exported BSD shape, so resolution/depth are not hardcoded.
        replacements[flattened.name] = [
            helper.make_node('Transpose', [scaled_q.input[0]], [prefix + 'fused_q_bsnh'],
                             name=prefix + 'fused_q_transpose', perm=[0, 2, 1, 3]),
            helper.make_node('Reshape', [prefix + 'fused_q_bsnh', flattened.input[1]],
                             [prefix + 'fused_q_bsd'], name=prefix + 'fused_q_reshape'),
            helper.make_node('MultiHeadAttention',
                             [prefix + 'fused_q_bsd', transposed_k.input[0], weighted_v.input[1]],
                             list(flattened.output), name=prefix + 'FusedMHA',
                             domain='com.microsoft', num_heads=heads, scale=scale),
        ]

    rewritten = [item for node in nodes for item in replacements.get(node.name, [node])]
    needed = {value.name for value in model.graph.output}
    kept = []
    for node in reversed(rewritten):
        if any(output in needed for output in node.output):
            kept.append(node)
            needed.update(node.input)
    del model.graph.node[:]
    model.graph.node.extend(reversed(kept))
    initializers = [value for value in model.graph.initializer if value.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    del model.graph.value_info[:]
    if not any(op.domain == 'com.microsoft' for op in model.opset_import):
        model.opset_import.append(helper.make_opsetid('com.microsoft', 1))
    onnx.checker.check_model(model)
    onnx.save_model(model, path)
    return len(replacements)
