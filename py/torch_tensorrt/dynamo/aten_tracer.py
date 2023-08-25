from __future__ import annotations

import logging
import unittest.mock
from typing import Any, Tuple

import torch
from torch._export import dynamic_dim, export
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo.backend.backends import constant_fold
from torch_tensorrt.dynamo.lowering import get_decompositions

logger = logging.getLogger(__name__)


def trace(
    model: torch.nn.Module | torch.fx.GraphModule,
    inputs: Tuple[Any, ...],
    **kwargs: Any,
) -> torch.fx.GraphModule:
    # Set log level at the top of compilation (torch_tensorrt.dynamo)
    if ("debug" in kwargs and kwargs["debug"]) and logger.parent:
        logger.parent.setLevel(logging.DEBUG)

    # Determine the dynamic dimension and setup constraints to input dimensions as dictated by TensorRT
    # Torch dynamo does not allow 0/1 value for dynamic dimensions
    # for inputs during tracing. Hence we create new inputs for export
    torch_inputs = [input.torch_tensor for input in inputs]
    trace_inputs = []
    constraints = []
    for idx, input in enumerate(inputs):
        if input.shape_mode == Input._ShapeMode.DYNAMIC:
            min_shape = input.shape["min_shape"]
            opt_shape = input.shape["opt_shape"]
            max_shape = input.shape["max_shape"]
            assert len(min_shape) == len(opt_shape) == len(max_shape)

            constraint_dims = []
            new_shape = []
            for dim in range(len(min_shape)):
                if min_shape[dim] == opt_shape[dim] == max_shape[dim]:
                    new_shape.append(torch_inputs[idx].shape[dim])
                else:
                    constraint_dims.append(dim)
                    if torch_inputs[idx].shape[dim] == 1:
                        new_shape.append(torch_inputs[idx].shape[dim] + 1)
                    else:
                        new_shape.append(torch_inputs[idx].shape[dim])
            trace_input = torch.randn(new_shape, dtype=torch_inputs[idx].dtype).cuda()

            for dim in constraint_dims:
                if min_shape[dim] > 1:
                    constraints.append(min_shape[dim] <= dynamic_dim(trace_input, dim))
                if max_shape[dim] > 1:
                    constraints.append(dynamic_dim(trace_input, dim) <= max_shape[dim])
            trace_inputs.append(trace_input)
        else:
            trace_inputs.append(torch_inputs[idx])

    experimental_decompositions = kwargs.get(
        "enable_experimental_decompositions", False
    )
    with unittest.mock.patch(
        "torch._export.DECOMP_TABLE", get_decompositions(experimental_decompositions)
    ):
        graph_module = export(
            model, tuple(trace_inputs), constraints=constraints
        ).module()
        constant_fold(graph_module)
    logger.debug("Post export graph: " + str(graph_module.graph))
    return graph_module
