import os
import sys
import time

import torch
import torch.nn as nn
import torch_tensorrt
from torch.distributed._tensor import Shard
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
import ctypes
import numpy as np
"""
This example copies some code from https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/tensor_parallel_example.py
"""

plugin_registry = trt.get_plugin_registry()
plugin_library = ctypes.CDLL("/code/data_parallelism/tensor_parallelism/auto-deploy/plugins/build/libnvinfer_plugin_tensorrt_llm.so")
# Iterate over all registered plugin creators
for plugin_creator in plugin_registry.plugin_creator_list:
    print(f"Plugin Name: {plugin_creator.name}, Namespace: {plugin_creator.plugin_namespace}")

######################converter imports##################
from torch_tensorrt.dynamo.conversion import (
    ConversionContext,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
import tensorrt as trt

TRT_LLM_PLUGIN_NAMESPACE= "tensorrt_llm"
#import tensorrt_llm


@dynamo_tensorrt_converter(torch.ops._c10d_functional.all_gather_into_tensor.default)
def aten_ops_all_gather_into_tensor_nccl(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
):# type: ignore
    print("Inserting All Gather plugin")
    plug_inputs = [args[0]]
    plugin_registry = trt.get_plugin_registry()
    plugin_creator = plugin_registry.get_plugin_creator(
        "AllGather", "1", TRT_LLM_PLUGIN_NAMESPACE
    )
    assert plugin_creator, f"Unabled to find plugin creator with name AllGather"

    import os
    #check if group_size and group_name which are args[1] and args[2]
    _rank = int(os.environ["RANK"])
    _world_size = int(os.environ["WORLD_SIZE"])
    group = list(range(_world_size))
    group = trt.PluginField("group", np.array(group, dtype=np.int32), trt.PluginFieldType.INT32)

    p_dtype = trt.float16
    pf_type = trt.PluginField(
        "type_id", np.array([int(p_dtype)], np.int32), trt.PluginFieldType.INT32
    )

    pfc = trt.PluginFieldCollection([group, pf_type])
    allgather = plugin_creator.create_plugin("allgather", pfc)

    layer = ctx.net.add_plugin_v2(plug_inputs, allgather)

    set_layer_name(layer, target, name)
    return layer.get_output(0)
class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 3200)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(3200, 1600)
        self.in_proj2 = nn.Linear(1600, 500)
        self.out_proj2 = nn.Linear(500, 100)

    def forward(self, x):
        x = self.out_proj(self.relu(self.in_proj(x)))
        x = self.relu(x)
        x = self.out_proj2(self.relu(self.in_proj2(x)))
        return x


# create a device mesh based on the given world_size.
_world_size = int(os.environ["WORLD_SIZE"])

device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()


print(f"Starting PyTorch TP example on rank {_rank}.")
assert (
    _world_size % 2 == 0
), f"TP examples require even number of GPUs, but got {_world_size} gpus"


# # create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
tp_model = ToyModel().to("cuda")


# Custom parallelization plan for the model
tp_model = parallelize_module(
    module=tp_model,
    device_mesh=device_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj": RowwiseParallel(output_layouts=Shard(0)),
        "in_proj2": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj2": RowwiseParallel(output_layouts=Shard(0)),
    },
)
torch.manual_seed(0)
inp = torch.rand(20, 10, device="cuda")
python_result = tp_model(inp)


backend = "torch_tensorrt"
tp_model = torch.compile(
    tp_model,
    backend=backend,
    options={
        "truncate_long_and_double": True,
        "enabled_precisions": {torch.float32, torch.float16},
        "use_python_runtime": True,
        "min_block_size": 1,
    },
    dynamic=False,
)

for i in range(10):
    # For TP, input needs to be same across all TP ranks.
    # Setting the random seed is to mimic the behavior of dataloader.
    torch.manual_seed(i)
    inp = torch.rand(20, 10, device="cuda")
    start = time.time()
    output = tp_model(inp)
    end = time.time()
    if i == 0:
        print(f"Compilation time is {end-start}")
        assert (
            python_result - output
        ).std() < 0.01, "Compilation result is not correct."
    elif _rank == 0:
        print(f"Inference time is {end-start}")
