from __future__ import annotations

import logging
import warnings

import torch

log = logging.getLogger(__name__)


def replace_collectives_with_low_contention(
    graph: torch.fx.Graph,
    mode: bool | None = None,
    ag_impl: str = "low_contention",
) -> None:
    """Replace FSDP collectives with symm_mem variants.

    ag_impl: "low_contention" (copy engine P2P) or "multimem" (multicast store).
    """
    if mode is False:
        return

    c10d = torch.ops._c10d_functional
    symm_mem = torch.ops.symm_mem

    AG_TARGETS = {
        c10d.all_gather_into_tensor.default,
        c10d.all_gather_into_tensor_out.default,
    }
    RS_TARGETS = {
        c10d.reduce_scatter_tensor.default,
        c10d.reduce_scatter_tensor_out.default,
    }

    replacements = 0
    skipped = 0
    total = 0
    _enabled_groups: set[str] = set()

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        is_ag = node.target in AG_TARGETS
        is_rs = node.target in RS_TARGETS
        if not is_ag and not is_rs:
            continue

        total += 1

        if mode is None:
            has_overlap = node.meta.get("has_compute_overlap")
            if has_overlap is None:
                has_overlap = _has_compute_overlap(node, graph)
            if not has_overlap:
                skipped += 1
                continue

        if is_ag:
            _replace_all_gather(node, graph, symm_mem, _enabled_groups, ag_impl)
        else:
            _replace_reduce_scatter(node, graph, symm_mem, _enabled_groups)
        replacements += 1

    if total > 0:
        log.info(
            "Replaced %d/%d FSDP collectives (ag_impl=%s, "
            "skipped %d critical-path)",
            replacements,
            total,
            ag_impl,
            skipped,
        )


def _replace_all_gather(node, graph, symm_mem, enabled_groups, ag_impl="low_contention"):
    input_node = node.args[0]
    group_name = node.args[2]
    _ensure_symm_mem_for_group(group_name, enabled_groups)

    if ag_impl == "multimem":
        _replace_all_gather_multimem(node, graph, symm_mem, input_node, group_name)
    elif ag_impl == "multimem_inplace":
        _replace_all_gather_multimem(
            node, graph, symm_mem, input_node, group_name, inplace=True
        )
    else:
        with graph.inserting_before(node):
            new_node = graph.call_function(
                symm_mem._low_contention_all_gather.default,
                args=(input_node, group_name),
            )
        new_node.meta.update(node.meta)
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)


def _replace_all_gather_multimem(
    node, graph, symm_mem, input_node, group_name, inplace=False
):
    """Replace all_gather with multimem variant."""
    target = (
        symm_mem._multimem_all_gather_inplace.default
        if inplace
        else symm_mem._multimem_all_gather.default
    )
    with graph.inserting_before(node):
        new_node = graph.call_function(
            target,
            args=(input_node, group_name),
        )
    new_node.meta.update(node.meta)
    node.replace_all_uses_with(new_node)
    graph.erase_node(node)


def _replace_reduce_scatter(node, graph, symm_mem, enabled_groups):
    input_node = node.args[0]
    reduce_op = node.args[1]
    group_name = node.args[3]
    _ensure_symm_mem_for_group(group_name, enabled_groups)
    with graph.inserting_before(node):
        new_node = graph.call_function(
            symm_mem._low_contention_reduce_scatter.default,
            args=(input_node, reduce_op, group_name),
        )
    new_node.meta.update(node.meta)
    node.replace_all_uses_with(new_node)
    graph.erase_node(node)


def _has_compute_overlap(start_node, graph):
    from torch._inductor.fx_passes.overlap_scheduling import is_compute_node

    wait_node = None
    for user in start_node.users:
        if _is_wait_tensor(user):
            wait_node = user
            break
    if wait_node is None:
        return False

    node_positions = {node: i for i, node in enumerate(graph.nodes)}
    start_pos = node_positions[start_node]
    wait_pos = node_positions[wait_node]

    for node in graph.nodes:
        pos = node_positions[node]
        if pos <= start_pos or pos >= wait_pos:
            continue
        if is_compute_node(node):
            return True
    return False


def _is_wait_tensor(node):
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.wait_tensor.default
    )


def _ensure_symm_mem_for_group(group_name, enabled_groups):
    if group_name in enabled_groups:
        return
    enabled_groups.add(group_name)
    from torch.distributed._symmetric_memory import (
        enable_symm_mem_for_group,
        is_symm_mem_enabled_for_group,
    )

    if not is_symm_mem_enabled_for_group(group_name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            enable_symm_mem_for_group(group_name)
