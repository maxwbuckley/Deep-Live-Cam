"""ONNX model optimizations for CoreML execution.

CoreML's execution provider in ONNX Runtime does not support ``Pad``
with ``mode=reflect``, causing models that use reflect padding to be
split into many small CoreML subgraphs with CPU fallbacks in between.
Each CPU↔ANE round-trip adds latency.

This module rewrites ``Pad(reflect)`` as equivalent ``Slice`` + ``Concat``
sequences that CoreML handles natively, allowing the entire model to
run as fewer CoreML partitions on the Apple Neural Engine.

The transformation is **bit-for-bit identical** to the original — it
implements the same reflect-padding semantics, just expressed in ops
that CoreML supports.
"""

import os
import platform
from typing import Optional

import numpy as np

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"


def optimize_for_coreml(model_path: str) -> str:
    """Return path to a CoreML-optimized ONNX model.

    If the model contains ``Pad(reflect)`` nodes, they are decomposed
    into ``Slice`` + ``Concat`` sequences.  The optimized model is cached
    next to the original (with ``_coreml`` suffix) so the rewrite only
    runs once.

    On non-Apple-Silicon platforms, returns the original path unchanged.
    """
    if not IS_APPLE_SILICON:
        return model_path

    import onnx
    from onnx import numpy_helper, helper

    base, ext = os.path.splitext(model_path)
    optimized_path = f"{base}_coreml{ext}"
    if os.path.exists(optimized_path):
        if os.path.getmtime(optimized_path) >= os.path.getmtime(model_path):
            return optimized_path

    model = onnx.load(model_path)
    graph = model.graph

    inits = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    # Find Pad(reflect) nodes
    reflect_pads = []
    for node in graph.node:
        if node.op_type == "Pad":
            mode = "constant"
            for attr in node.attribute:
                if attr.name == "mode":
                    mode = attr.s.decode()
            if mode == "reflect" and len(node.input) > 1 and node.input[1] in inits:
                reflect_pads.append(node)

    if not reflect_pads:
        return model_path

    # Pre-create all needed constant tensors as initializers.
    # We need integer constants for Slice start/end/axes params.
    existing_names = {i.name for i in graph.initializer}

    def ensure_const(name: str, value):
        if name not in existing_names:
            graph.initializer.append(
                numpy_helper.from_array(np.array(value, dtype=np.int64), name=name)
            )
            existing_names.add(name)

    # Axes constants
    ensure_const("_rp_ax2", [2])
    ensure_const("_rp_ax3", [3])

    # Determine all needed slice indices from pad sizes
    max_pad = 0
    for node in reflect_pads:
        pads = inits[node.input[1]].tolist()
        max_pad = max(max_pad, int(pads[2]), int(pads[3]))

    # For a reflect pad of size P, we need:
    #   Top/Left slices: start=P, end=P+1; start=P-1, end=P; ... ; start=1, end=2
    #   Bot/Right slices: start=-(2), end=-(1); start=-(3), end=-(2); ... ; start=-(P+1), end=-(P)
    for v in range(1, max_pad + 2):
        ensure_const(f"_rp_p{v}", [v])
        ensure_const(f"_rp_n{v}", [-v])

    # Replace Pad nodes
    _counter = [0]

    def uid():
        _counter[0] += 1
        return _counter[0]

    pad_ids = {id(n) for n in reflect_pads}
    pad_init_names = set()

    new_nodes = []
    for node in graph.node:
        if id(node) not in pad_ids:
            new_nodes.append(node)
            continue

        pads = inits[node.input[1]].tolist()
        h_pad, w_pad = int(pads[2]), int(pads[3])

        for inp in node.input[1:]:
            if inp in inits:
                pad_init_names.add(inp)

        current = node.input[0]

        # Reflect-pad H dimension (axis=2)
        if h_pad > 0:
            # Top: rows [h_pad, h_pad-1, ..., 1] (reflected)
            top = []
            for i in range(h_pad, 0, -1):
                name = f"_rp_t{uid()}"
                new_nodes.append(helper.make_node(
                    "Slice",
                    inputs=[current, f"_rp_p{i}", f"_rp_p{i+1}", "_rp_ax2"],
                    outputs=[name],
                ))
                top.append(name)

            # Bottom: rows [-(2), -(3), ..., -(h_pad+1)] (reflected)
            bot = []
            for i in range(1, h_pad + 1):
                name = f"_rp_b{uid()}"
                new_nodes.append(helper.make_node(
                    "Slice",
                    inputs=[current, f"_rp_n{i+1}", f"_rp_n{i}", "_rp_ax2"],
                    outputs=[name],
                ))
                bot.append(name)

            h_out = f"_rp_h{uid()}"
            new_nodes.append(helper.make_node(
                "Concat", inputs=top + [current] + bot, outputs=[h_out], axis=2
            ))
            current = h_out

        # Reflect-pad W dimension (axis=3)
        if w_pad > 0:
            left = []
            for i in range(w_pad, 0, -1):
                name = f"_rp_l{uid()}"
                new_nodes.append(helper.make_node(
                    "Slice",
                    inputs=[current, f"_rp_p{i}", f"_rp_p{i+1}", "_rp_ax3"],
                    outputs=[name],
                ))
                left.append(name)

            right = []
            for i in range(1, w_pad + 1):
                name = f"_rp_r{uid()}"
                new_nodes.append(helper.make_node(
                    "Slice",
                    inputs=[current, f"_rp_n{i+1}", f"_rp_n{i}", "_rp_ax3"],
                    outputs=[name],
                ))
                right.append(name)

            new_nodes.append(helper.make_node(
                "Concat",
                inputs=left + [current] + right,
                outputs=[node.output[0]],
                axis=3,
            ))
        elif h_pad > 0:
            new_nodes.append(helper.make_node(
                "Identity", inputs=[current], outputs=[node.output[0]]
            ))

    # Clean up old Pad initializers.
    # IMPORTANT: insightface's INSwapper reads graph.initializer[-1] as
    # the embedding map (emap).  We must keep the original last
    # initializer in the last position — new constants go before it.
    # Identify the emap by finding the (512, 512) float matrix.
    emap_init = None
    for init in graph.initializer:
        if init.name not in pad_init_names and not init.name.startswith("_rp_"):
            arr = numpy_helper.to_array(init)
            if len(arr.shape) == 2 and arr.shape[0] == 512 and arr.shape[1] == 512:
                emap_init = init
                break

    clean_inits = [i for i in graph.initializer
                   if i.name not in pad_init_names
                   and (emap_init is None or i.name != emap_init.name)]
    del graph.initializer[:]
    graph.initializer.extend(clean_inits)
    if emap_init is not None:
        graph.initializer.append(emap_init)

    del graph.node[:]
    graph.node.extend(new_nodes)

    onnx.save(model, optimized_path)
    return optimized_path
