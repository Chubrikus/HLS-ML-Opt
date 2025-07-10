import random
from random import sample 
import copy

def Gen_one_case(case_id, max_prim_in, max_op_cnt):
    """
    Generate a single synthetic case with the following parameters:
    - case_id: Unique identifier for the case
    - max_prim_in: Maximum number of primary inputs
    - max_op_cnt: Maximum number of operations
    """
    prim_in_cnt = random.randint(5, max_prim_in)
    op_cnt = random.randint(prim_in_cnt * 2, max(max_op_cnt, prim_in_cnt * 4))

    # Operation format: [i, type, prec, out_degree]
    # type: -2: prim_in, -1: prim_out, 1: add, 2: mul
    # prec: data precision from 2 to 16-bit

    all_ops = []
    all_ops.append([-1, -1, -1, -1])  # placeholder

    # Generate primary inputs
    prim_ins = []
    for i in range(1, prim_in_cnt + 1):
        prec = random.randint(2, 16)
        prim_ins.append([i, -2, prec, 0])
        all_ops.append([i, -2, prec, 0])

    # Generate intermediate operations
    inter_ops = []
    for i in range(prim_in_cnt + 1, prim_in_cnt + op_cnt + 1):
        op_type = int(round(random.uniform(1.3, 2.3)))  # 1: add, 2: mul
        inter_ops.append([i, op_type, -1, 0])
        all_ops.append([i, op_type, -1, 0])

    # Generate edges
    available_ops = copy.deepcopy(prim_ins)
    unsigned_ops = copy.deepcopy(inter_ops)
    all_edges = []
    idx = 0

    while unsigned_ops:
        op = unsigned_ops.pop(0)
        opr1 = sample(available_ops[idx:], 2)[0]
        opr2 = sample(available_ops[idx:], 2)[1]

        # Update out-degree
        all_ops[opr1[0]][3] += 1
        all_ops[opr2[0]][3] += 1

        # Infer data precision
        opr1_prec = opr1[2]
        opr2_prec = opr2[2]
        op_prec = random.randint(
            max(2, int((opr1_prec + opr2_prec)/4)),
            min(opr1_prec + opr2_prec, 16)
        )
        op[2] = all_ops[op[0]][2] = op_prec

        # Add edges
        available_ops.append(op)
        all_edges.append([all_ops[opr1[0]], all_ops[op[0]]])
        all_edges.append([all_ops[opr2[0]], all_ops[op[0]]])
        idx += 1

    # Generate primary outputs
    prim_outs = []
    i = prim_in_cnt + op_cnt
    for op in all_ops[1:]:
        if op[3] > 0 or op[1] == -2:
            continue

        i += 1
        prec = 16  # All outputs are 16-bit
        p_out = [i, -1, prec, 0]
        prim_outs.append(p_out)
        all_edges.append([op, p_out])

    all_ops.extend(prim_outs)
    prim_out_cnt = len(prim_outs)

    # Generate output files
    _generate_graph_file(case_id, all_ops, all_edges, prim_in_cnt, op_cnt, prim_out_cnt)
    _generate_hls_code(case_id, all_ops, all_edges, prim_in_cnt, op_cnt, prim_out_cnt)
    _generate_directive_file(case_id, prim_in_cnt, op_cnt)
    _generate_script_file(case_id)

def _generate_graph_file(case_id, all_ops, all_edges, prim_in_cnt, op_cnt, prim_out_cnt):
    """Generate the DFG graph file"""
    io_dic = {-2: 'in', -1: 'o', 1: 'm', 2: 'm'}
    tp_dic = {1: '+', 2: '*'}

    with open(f"DFG_case_{case_id}.txt", "w") as f:
        # Write primary inputs
        f.write("# Primary Inputs:\n")
        for op in all_ops[1: prim_in_cnt+1]:
            f.write(f"in{op[0]} INT{op[2]}\n")

        # Write intermediate operations
        f.write("\n\n# Intermediate Operations:\n")
        for op in all_ops[prim_in_cnt+1: prim_in_cnt+op_cnt+1]:
            f.write(f"m{op[0]} {tp_dic[op[1]]} INT{op[2]}\n")

        # Write edges
        f.write("\n\n# Edges:\n")
        for e in all_edges:
            op1, op2 = e
            pre1 = io_dic[op1[1]]
            pre2 = io_dic[op2[1]]
            f.write(f"{pre1}{op1[0]} {pre2}{op2[0]}\n")

        # Write primary outputs
        f.write("\n\n# Primary Outputs:\n")
        for op in all_ops[prim_in_cnt+op_cnt+1:]:
            f.write(f"o{op[0]}\n")

def _generate_hls_code(case_id, all_ops, all_edges, prim_in_cnt, op_cnt, prim_out_cnt):
    """Generate the HLS C++ code"""
    file_head = f"""
#include <stdio.h>
#include "ap_fixed.h"

void case_{case_id}(
    ap_int<16> in_data[{prim_in_cnt}],
    ap_int<16> out_data[{prim_out_cnt}]
)
{{
#pragma HLS array_partition variable=in_data complete
#pragma HLS array_partition variable=out_data complete
"""

    # Write inputs
    lines = "\n"
    for i in range(1, prim_in_cnt + 1):
        lines += f"ap_int<{all_ops[i][2]}> in{all_ops[i][0]};\n"
        lines += f"in{all_ops[i][0]}.range({all_ops[i][2]-1}, 0) = in_data[{i-1}].range({all_ops[i][2]-1}, 0);\n"
    lines += "\n"

    # Write intermediate operators
    for i in range(prim_in_cnt+1, prim_in_cnt+op_cnt+1):
        lines += f"ap_int<{all_ops[i][2]}> m{all_ops[i][0]};\n"
    lines += "\n"

    # Write operations
    op_dic = {}
    io_dic = {-2: 'in', -1: 'o', 1: 'm', 2: 'm'}
    tp_dic = {1: '+', 2: '*'}

    for e in all_edges:
        u, v = e
        if v[0] not in op_dic:
            op_dic[v[0]] = u[0]
            continue

        op1 = op_dic[v[0]]
        op2 = u[0]
        pre1 = io_dic[all_ops[op1][1]]
        pre2 = io_dic[all_ops[op2][1]]
        op_type = tp_dic[v[1]]

        lines += f"m{v[0]} = {pre1}{op1} {op_type} {pre2}{op2};\n"
    lines += "\n"

    # Write outputs
    for i in range(prim_in_cnt+op_cnt+1, len(all_ops)):
        op = op_dic[all_ops[i][0]]
        lines += f"out_data[{i-(prim_in_cnt+op_cnt+1)}] = m{op};\n"
    lines += "\n"

    file_tail = "}\n"

    with open(f"case_{case_id}.cc", "w") as f:
        f.write(file_head + lines + file_tail)

def _generate_directive_file(case_id, prim_in_cnt, op_cnt):
    """Generate the HLS directive file"""
    with open("directive.tcl", "w") as f:
        for i in range(prim_in_cnt+1, prim_in_cnt+op_cnt+1):
            f.write(f'set_directive_resource -core Mul_LUT "case_{case_id}" m{i}\n')

def _generate_script_file(case_id):
    """Generate the HLS script file"""
    content = f"""
open_project project_{case_id}
set_top case_{case_id}
add_files case_{case_id}.cc
open_solution "solution_{case_id}"
set_part {{xc7z020clg484-1}}
create_clock -period 10 -name default
source "./directive.tcl"
csynth_design
export_design -evaluate verilog -format ip_catalog
exit
"""
    with open("script.tcl", "w") as f:
        f.write(content) 