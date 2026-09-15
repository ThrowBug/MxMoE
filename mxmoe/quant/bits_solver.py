import os
import json
import time
import argparse
import itertools
import dataclasses

import gurobipy as gp

from pprint import pprint
from typing import Literal
from collections import defaultdict

from project_config import *
from mxmoe.kernels.qconfig import QLinearConfig
from mxmoe.kernels.tile_config import TileConfig, QConfig, NO_QUANT, get_possible_tile_list
from mxmoe.kernels.compose_kernel import is_fusion_compatible, QCFG_MAP
from mxmoe.quant.artifact_utils import (
    load_optional_metadata,
    validate_calibration_artifacts,
    write_metadata,
)


def value_to_prob(freq: list[int]| list[float]) -> list[float]:
    total = sum(freq)
    return [f / total for f in freq]


def export_qconfig(select_strategies: tuple[dict[int, dict[int, dict[int, str]]], dict[int, list[TileConfig]]], save_path:str=None):
    if save_path is not None and not save_path.endswith(".json"):
        raise ValueError("Invalid save_path, must be a json file.")

    select_strategies, tiled_cfgs = select_strategies
    tiled_cfgs = {k: tiled_cfgs[k].__repr__() for k in tiled_cfgs}

    def parse_str(qname: str):
        qname = qname.split("_")
        w_bits, a_bits = [int(x) for x in qname[0].split("w")[1].split("a")]
        gsize = int(qname[1][1:])
        sym = qname[2] == "sym"
        return QLinearConfig(w_bits=w_bits, w_gsize=gsize, w_sym=sym, a_bits=a_bits, a_gsize=gsize, a_sym=sym).to_dict()
    
    weight_idx_2_name = {0: "gate", 1: "up", 2: "down"}

    LT = select_strategies.pop("LT", None)
    model_qcfg = {
        layer_idx: {
            "experts": {
                expert_idx: {
                    weight_idx_2_name[weight_idx]: parse_str(weight_cfg)
                    for weight_idx, weight_cfg in expert_cfg.items()
                }
                for expert_idx, expert_cfg in layer_cfg.items()
            }
        }
        for layer_idx, layer_cfg in select_strategies.items()
    }
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        cur_cfg = {}
        cur_cfg.update(model_qcfg)
        if LT is not None:
            cur_cfg["LT"] = LT

        with open(save_path, "w") as f:
            json.dump(cur_cfg, f)
            print(f"Exported qconfig to `{save_path}` ...")

        with open(f"{save_path.split('.json')[0]}_{'tile_cfg'}.json", "w") as f:
            json.dump(tiled_cfgs, f)
            print(f"Exported qconfig to `{save_path.split('.json')[0]}_{'tile_cfg'}.json` ...")

    return model_qcfg


def print_select_strategies(solution: dict[int, str]):
    strategies = set(solution.values())
    inv_dict = {s: {"num": 0, "exp": []} for s in strategies}

    for exp_id, strategy in solution.items():
        inv_dict[strategy]["num"] += 1
        inv_dict[strategy]["exp"].append(exp_id)
    
    pprint(f"Solution: {solution}\n{inv_dict}")


def get_strategy_loss(calib_loss_list: dict[str, str], filter_list: list[str], qtype: str):
    # {quant_strategy: {layer_idx: {exp_idx: loss}}}
    strategy_loss: dict[str, dict[str, dict[str, float]]] = {}

    for qname, loss_file in calib_loss_list.items():
        if not any([f in qname for f in filter_list]): continue
        with open(loss_file, "r") as f:
            layerwise_loss: dict[str, dict[str, dict[str, float]]] = json.load(f)
        strategy = qname
        strategy_loss[strategy] = layerwise_loss

    return strategy_loss


def parse_loss_file_overrides(values: list[str]) -> dict[str, str]:
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --loss_file {value!r}; expected QUANT_SCHEME=PATH."
            )
        quant_scheme, path = value.split("=", 1)
        if not quant_scheme or not path:
            raise ValueError(
                f"Invalid --loss_file {value!r}; expected QUANT_SCHEME=PATH."
            )
        if quant_scheme in overrides:
            raise ValueError(f"Duplicate --loss_file for {quant_scheme!r}.")
        overrides[quant_scheme] = path
    return overrides


ExpertLoss = tuple[float, float, float]
LayerLoss = dict[str, ExpertLoss]
ModelLoss = dict[str, LayerLoss]

@dataclasses.dataclass
class ProblemShape:
    M:int
    N:int
    K:int

    def num_tiles(self, tile: TileConfig):
        return ((self.M+tile.BM-1)//tile.BM) * ((self.N+tile.BN-1)//tile.BN)


def estimate_runtime(shapes: list[tuple[ProblemShape, int]], tile_cfgs: list[TileConfig]):
    for shape, cfg in shapes:
        tile = tile_cfgs[cfg]
        num_tiles = shape.num_tiles(tile)


def solve_model_qconfig_model_level(
    model_id: str,
    strategies: list[str],
    strategy_loss: dict[str, ModelLoss],
    strategy_wbits: list[int],
    strategy_abits: list[int],
    model_wbits_bugdet: int,
    batch_range: int,
    workloads: list[list[list[ProblemShape]]], # [layer_idx][exp_idx][weight_idx]
    access_prob: list[list[float]],
    r: float,
    TOP_K: int,
):
    num_layers = len(next(iter(strategy_loss.values())))

    model_qconfig = {}

    M = len(strategies)
    L = num_layers
    N = len(next(iter(strategy_loss.values()))["0"])
    K = 3 if ExpertLoss == tuple[float, float, float] else 1

    # ALPHA = alpha[layer_idx]
    # BETA = beta[layer_idx]
    # GAMMA = gamma[layer_idx]

    ##################################################################################
    model = gp.Model("quant_loss_top_k")

    # binary variable: x[i,l,j,k] indicate whether weight[k] of expert[j] in layer[l] select strategy[i]
    x = model.addVars(K, N, L, M, vtype=gp.GRB.BINARY, name="x")
    model.setObjective(
        gp.quicksum(
            x[k, j, l, i] * strategy_loss[strategies[i]][str(l)][str(j)][k]
            for k in range(K)
            for j in range(N)
            for l in range(L)
            for i in range(M)
        ),
        sense=gp.GRB.MINIMIZE
    )
    # Constrains:
    model.addConstr(
        gp.quicksum(
            strategy_wbits[m] * x[k, n, l, m]
            for k in range(K) for n in range(N) for l in range(L) for m in range(M)
        ) <= model_wbits_bugdet,
        name="memory_budget"
    )
    for l,j,k in itertools.product(range(L), range(N), range(K)):
        model.addConstr(gp.quicksum(x[k, j, l, i] for i in range(M)) == 1, name=f"one_strategy_for_layer_{l}_expert_{j}_weight_{k}")

    model.setParam("LogToConsole", 0)
    model.setParam("PoolSearchMode", 2)
    model.setParam("PoolSolutions", TOP_K)
    # model.write(f"bits_model.lp")
    ##################################################################################

    # OPT
    model.optimize()

    best_obj_val = model.ObjVal
    for i in range(model.SolCount):
        model.setParam(gp.GRB.Param.SolutionNumber, i)
        obj_val = model.PoolObjVal

        select_strategies = {i: defaultdict(dict) for i in range(L)}
        for var in model.getVars():
            if var.Xn > 0.5:
                # print(var.varName)
                weight_id, exp_id, layer_id, strategy_id = [int(x) for x in var.varName.split("[")[1].split("]")[0].split(",")]
                select_strategies[layer_id][exp_id][weight_id] = strategies[strategy_id]
                # print(f"  {var.varName} = {var.x}")

        if i == 0:
            model_qconfig = select_strategies
            # print_select_strategies(select_strategies)
        print(f"best_obj_val: {best_obj_val}, cur_distance: {obj_val - best_obj_val:.3f}")

    return model_qconfig


def solve_model_qconfig_layer_level(
    model_id: str,
    strategies: list[str],
    strategy_loss: dict[str, ModelLoss],
    strategy_wbits: list[int],
    strategy_abits: list[int],
    layer_wbits_bugdet: int,
    batch_range: int,
    workloads: list[list[list[ProblemShape]]], # [layer_idx][exp_idx][weight_idx]
    access_prob: list[list[float]],
    expert_sizes: list[list[int]], # [layer_idx][exp_idx]
    r: float,
    TOP_K: int,
    args,
):
    num_layers = len(next(iter(strategy_loss.values())))

    model_qconfig = {"LT": {}}
    model_tilecfg = {}

    S = len(strategies)
    N = 3 if ExpertLoss == tuple[float, float, float] else 1

    def convert_to_delta_array(strategy_loss: dict[str, ModelLoss], layer_idx: str,num_strategies: int,num_experts: int,num_blocks: int) -> list[list[list[float]]]:
        delta = [[[0.0 for s in range(num_strategies)] for n in range(num_blocks)] for e in range(num_experts)]
        for s_idx, strategy_name in enumerate(strategies):
            layer_loss = strategy_loss[strategy_name][str(layer_idx)]
            for e in range(num_experts):
                expert_loss = layer_loss[str(e)]
                for n in range(num_blocks):
                    delta[e][n][s_idx] = expert_loss[n]
        return delta

    # Solve layer-by-layer
    for layer_idx in range(num_layers):
        E = len(expert_sizes[layer_idx])

        solve_layer_st = time.time()
        # if layer_idx > 0: break
        if model_id == "ds2" and layer_idx == 0:
            model_qconfig["LT"][layer_idx] = [0, 0]
            model_qconfig[layer_idx] = {0:{0: "w4a4_g-1_asym", 1: "w4a4_g-1_asym", 2: "w4a4_g-1_asym"}}
            model_tilecfg[layer_idx] = []
            continue

        # ALPHA = alpha[layer_idx]
        # BETA = beta[layer_idx]
        # GAMMA = gamma[layer_idx]

        print(f"Layer: {layer_idx}")

        if r == 1:
            tile_cfgs: list[list[TileConfig]] = [[] for _ in range(1)]
            runtime_cost: list[list[list[list[float]]]] = [
                [[[1.0] for _ in range(S)] for _ in range(N)]
                for _ in range(E)
            ]
            num_tile_cfgs = len(runtime_cost[0][0][0])
        else:
            # y[e,n,s,t]: whether linear[n] in expert[e] is assigned to quant_strategy[s], using tile_cfgs[t]
            tile_cfgs, runtime_cost = get_runtime_cost(batch_range, workloads[layer_idx], strategies, offline_stats["performance_table"])
            num_tile_cfgs = len(runtime_cost[0][0][0])

        # Δ[e,n,s]: loss of linear[n] in expert[e] using quant_strategy[s]
        delta = convert_to_delta_array(strategy_loss, layer_idx, S, E, N)
        layer_expert_sizes = expert_sizes[layer_idx]

        if args.exp_alloc: # we use the sum of all the linear block losses as the expert loss
            new_delta = [[0.0 for s in range(S)] for e in range(E)]
            for e,s in itertools.product(range(E), range(S)):
                exp_loss = sum(delta[e][n][s] for n in range(N))
                new_delta[e][s] = exp_loss

            new_runtime_cost = [[[0.0 for t in range(num_tile_cfgs)] for s in range(S)] for e in range(E)]
            for e,s,t in itertools.product(range(E), range(S), range(num_tile_cfgs)):
                new_runtime_cost[e][s][t] = sum(runtime_cost[e][n][s][t] for n in range(N))

            delta = new_delta
            runtime_cost = new_runtime_cost


        # print(f">>> num_tile_cfgs: {num_tile_cfgs}")
        # print(f">>> expert_sizes: {layer_expert_sizes}")
        # print(">>> access_prob: {access_prob}".format(access_prob=[f"{x:.3f}" for x in access_prob[layer_idx]]))
        ##################################################################################
        model = gp.Model("quant_loss_top_k")



        if args.exp_alloc:
            # x[e,n,s] indicate whether weights[n] in experts[e] select strategy[s]
            x = model.addVars(E,S, vtype=gp.GRB.BINARY, name="x")
            L = gp.quicksum(
                x[e,s] * delta[e][s]
                for e in range(E) for s in range(S)
            )
            if r == 1:
                # Accuracy-only allocation does not use runtime or tile choices.
                # Omitting y keeps Qwen3 below Gurobi's size-limited-license cap
                # without changing min L or any bit-allocation constraint.
                y = None
                T = gp.LinExpr(float(E))
                model.setObjective(L, sense=gp.GRB.MINIMIZE)
            else:
                y = model.addVars(E,S,num_tile_cfgs, vtype=gp.GRB.BINARY, name="y")
                T = gp.quicksum(
                    x[e,s] * runtime_cost[e][s][t] * y[e,s,t]
                    for e in range(E)
                    for s in range(S)
                    for t in range(num_tile_cfgs)
                )
                model.setObjective(
                    L**r * T**(1-r),
                    sense=gp.GRB.MINIMIZE
                )
            # constrain: memory budget
            model.addConstr(
                gp.quicksum(
                    strategy_wbits[s] * x[e, s] * layer_expert_sizes[e]
                    for e in range(E) for s in range(S)
                ) <= layer_wbits_bugdet/3, name="memory_budget"
            )
            # constrain: one strategy for each weight matrix
            for e in range(E):
                model.addConstr(gp.quicksum(x[e, s] for s in range(S)) == 1, name=f"1_strategy_for_E{e}")
            if y is not None:
                # constrain: one tile config for each weight matrix
                for e, n,s,t in itertools.product(range(E), range(N), range(S), range(num_tile_cfgs)):
                    model.addConstr(gp.quicksum(y[e,s,t] for t in range(num_tile_cfgs)) == 1, name=f"y_{e}_{s}_{t}")
        else:
            # x[e,n,s] indicate whether weights[n] in experts[e] select strategy[s]
            x = model.addVars(E, N, S, vtype=gp.GRB.BINARY, name="x")

            L = gp.quicksum(
                x[e, n, s] * delta[e][n][s]
                for e in range(E) for n in range(N) for s in range(S)
            )
            if r == 1:
                y = None
                T = gp.LinExpr(float(E * N))
            else:
                y = model.addVars(E,N,S,num_tile_cfgs, vtype=gp.GRB.BINARY, name="y")
                T = gp.quicksum(
                    x[e, n, s] * runtime_cost[e][n][s][t] * y[e,n,s,t]
                    for e in range(E)
                    for n in range(N)
                    for s in range(S)
                    for t in range(num_tile_cfgs)
                )

            if not (r == 0.0 or r == 1):
                epsilon = 1e-6
                model.addConstr(L >= epsilon, name="L_positive")
                model.addConstr(T >= epsilon, name="T_positive")
                L_var = model.addVar(lb=0, name="L_var")
                T_var = model.addVar(lb=0, name="T_var")
                model.addConstr(L_var == L, name="L_binding")
                model.addConstr(T_var == T, name="T_binding")
                model.addConstr(L_var >= epsilon, name="L_var_positive")
                model.addConstr(T_var >= epsilon, name="T_var_positive")
                logL = model.addVar(lb=-gp.GRB.INFINITY, name="logL")  # log(L)
                logT = model.addVar(lb=-gp.GRB.INFINITY, name="logT")  # log(T)
                model.addGenConstrLog(L_var, logL, name="logL_constraint")
                model.addGenConstrLog(T_var, logT, name="logT_constraint")
                # set obj as r * logL + (1 - r) * logT
                model.setObjective(
                    r * logL + (1 - r) * logT,
                    sense=gp.GRB.MINIMIZE
                )
            else:
                if r == 1:
                    model.setObjective(L, sense=gp.GRB.MINIMIZE)
                else:
                    model.setObjective(
                        L**r * T**(1-r),
                        sense=gp.GRB.MINIMIZE
                    )

            # constrain: memory budget
            model.addConstr(
                gp.quicksum(
                    strategy_wbits[s] * x[e, n, s] * layer_expert_sizes[e]
                    for e in range(E) for n in range(N) for s in range(S)
                ) <= layer_wbits_bugdet, name="memory_budget"
            )
            # constrain: one strategy for each weight matrix
            for e, n in itertools.product(range(E), range(N)):
                model.addConstr(gp.quicksum(x[e, n, s] for s in range(S)) == 1, name=f"1_strategy_for_E{e}_W{n}")
            if y is not None:
                # constrain: one tile config for each weight matrix
                for e, n,s,t in itertools.product(range(E), range(N), range(S), range(num_tile_cfgs)):
                    model.addConstr(gp.quicksum(y[e,n,s,t] for t in range(num_tile_cfgs)) == 1, name=f"y_{e}_{n}_{s}_{t}")
            # for e, s_gate, s_up in itertools.product(range(E), range(S), range(S)):
            #     if strategy_abits[s_gate] != strategy_abits[s_up]:
            #         model.addConstr(x[e,0,s_gate] + x[e,1,s_up] <= 1, name=f"same_act_bits_for_gate_up_E{e}_S{s_gate}_S{s_up}")

            # constrain: Same strategy for gate and up in each expert
            for e, s in itertools.product(range(E), range(S)):
                model.addConstr(x[e, 0, s] == x[e, 1, s], name=f"x_equal_for_E{e}_S{s}")

        model.setParam("LogToConsole", 0)
        model.setParam("PoolSearchMode", 2)
        model.setParam("PoolSolutions", TOP_K)

        if layer_idx == 1: model.write(f"bits_model-{layer_idx}.lp")
        ##################################################################################
        model.optimize()

        solve_layer_elpased = time.time() - solve_layer_st
        print(f">>> solve_layer_elpased: {solve_layer_elpased*1000:.3f} ms")
        best_obj_val = model.ObjVal
        for i in range(model.SolCount):
            model.setParam(gp.GRB.Param.SolutionNumber, i)
            obj_val = model.PoolObjVal

            selected_strategies = defaultdict(dict)
            selected_tile_config = [] if y is None else None
            for e, n, s in itertools.product(range(E), range(N), range(S)):
                if args.exp_alloc:
                    if x[e, s].X > 0.5:
                        selected_strategies[e][n] = strategies[s]
                    if y is not None:
                        for t in range(num_tile_cfgs):
                            if y[e,s,t].X > 0.5:
                                selected_tile_config = tile_cfgs[t]
                else:
                    if x[e, n, s].X > 0.5:
                        selected_strategies[e][n] = strategies[s]
                    if y is not None:
                        for t in range(num_tile_cfgs):
                            if y[e,n,s,t].X > 0.5:
                                selected_tile_config = tile_cfgs[t]

            # print(f"        best_obj_val: {best_obj_val}, cur_distance: {obj_val - best_obj_val:.3f}")
            if i == 0:
                print(f"i: {i}; L: {L.getValue():.3f}, T: {T.getValue():.3f}")
                LT = L.getValue(), T.getValue()
                model_qconfig["LT"][layer_idx] = LT
                model_qconfig[layer_idx] = selected_strategies
                model_tilecfg[layer_idx] = selected_tile_config
                # if selected_tile_config is not None:
                #     print("selected_tile_config:\n{selected_tile_config}".format(selected_tile_config="\n".join([str(x) for x in selected_tile_config])))

    return model_qconfig, model_tilecfg


def get_best_tile_configs(batch_range: int, strategy_list: list[str]) -> list[list[TileConfig]]:
    from mxmoe.kernels.tile_config import QCFG_TO_MMA
    if strategy_list == sorted(["w4a16_g-1_asym", "w8a8_g-1_sym"]) and batch_range == 512:
        return {
            "w4a16_g-1_asym": [
                TileConfig(BM=16, BN=128, BK=128, WM=1, WN=4, WK=2, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a16_g-1_asym"]),
                TileConfig(BM=64, BN=128, BK=128, WM=1, WN=4, WK=2, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a16_g-1_asym"]),
                TileConfig(BM=32, BN=128, BK=64, WM=1, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a16_g-1_asym"]),
            ],
            "w8a8_g-1_sym": [
                TileConfig(BM=128, BN=128, BK=128,WM=1, WN=8, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=64,WM=1, WN=4, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=256, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=256, BN=128, BK=64,WM=2, WN=4, WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=256, BN=128, BK=64,WM=2, WN=4, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
            ],
        }

    elif strategy_list == sorted(['w1a16_g128_asym', 'w2a16_g128_asym', 'w3a16_g128_asym', 'w4a16_g-1_asym', 'w4a16_g128_asym', 'w8a16_g-1_asym']) and batch_range == 256:
        raise NotImplementedError("Not implemented yet")


    elif strategy_list == sorted(["w2a16_g128_asym", "w4a16_g-1_asym", "w8a8_g-1_sym"]) and batch_range == 512:
        return {
            "w2a16_g128_asym": [
                TileConfig(BM=16, BN=256, BK=128,WM= 1,WN= 4,WK= 2,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w2a16_g-1_asym_accfp16"]),
                # TileConfig(BM=16, BN=128, BK=128, WM=1, WN=4, WK=2, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w2a16_g-1_asym_accfp16"]),
                # TileConfig(BM=64, BN=128, BK=128, WM=1, WN=4, WK=2, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w2a16_g-1_asym_accfp16"]),
            ],
            "w4a16_g-1_asym": [
                TileConfig(BM=16, BN=128, BK=128, WM=1, WN=4, WK=2, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a16_g-1_asym"]),
                # TileConfig(BM=64, BN=128, BK=128, WM=1, WN=4, WK=2, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a16_g-1_asym"]),
                TileConfig(BM=32, BN=128, BK=64, WM=1, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a16_g-1_asym"]),
            ],
            "w8a8_g-1_sym": [
                TileConfig(BM=192, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                TileConfig(BM=192, BN=128, BK=64,WM=1, WN=4, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=256, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=256, BN=128, BK=64,WM=2, WN=4, WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=256, BN=128, BK=64,WM=2, WN=4, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
            ],
        }

    elif strategy_list == sorted(["w4a4_g-1_sym", "w8a8_g-1_sym", "w4a4_g128_sym"]) and batch_range == 8192:
        return {
            "w4a4_g128_sym": [
                TileConfig(BM=192, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=128, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=128, WM=2, WN=4, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
            ],
            "w4a4_g-1_sym": [
                TileConfig(BM=256, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                TileConfig(BM=256, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=256, BN=128, BK=128, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
            ],
            "w8a8_g-1_sym": [
                TileConfig(BM=256, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                TileConfig(BM=256, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=256, BN=128, BK=64,WM=2, WN=4, WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=256, BN=128, BK=64,WM=2, WN=4, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
            ],
        }
    elif strategy_list == sorted(["w4a4_g-1_sym", "w8a8_g-1_sym"]) and batch_range == 8192:
        return {
            "w4a4_g-1_sym": [
                TileConfig(BM=256, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                TileConfig(BM=256, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=256, BN=128, BK=128, WM=2, WN=4, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=3, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=128, WM=1, WN=8, WK=1, STAGE=4, SPLITK=-1, MMA=QCFG_TO_MMA["w4a4_g-1_sym"]),
            ],
            "w8a8_g-1_sym": [
                TileConfig(BM=256, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                TileConfig(BM=256, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=256, BN=128, BK=64,WM=2, WN=4, WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=256, BN=128, BK=64,WM=2, WN=4, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 3,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
                # TileConfig(BM=192, BN=128, BK=64,WM=1, WN=8, WK= 1,STAGE= 4,SPLITK= -1,MMA=QCFG_TO_MMA["w8a8_g-1_sym"]),
            ],
        }

    else:
        raise ValueError(f"Invalid strategy_list: {strategy_list}")


def get_runtime_cost(batch_range: int, workloads: list[list[ProblemShape]], strategy_list: list[str], performance_table: dict):
    tile_cfgs = {k : [x.set_qcfg(QCFG_MAP[k]) for x in get_best_tile_configs(batch_range, strategy_list)[k]] for k in strategy_list}
    # tile_cfgs = {k : [x.set_qcfg(QCFG_MAP[k]) for x in get_possible_tile_list("89", k)] for k in strategy_list}

    tile_cfgs: list[list[TileConfig]] = list(filter(lambda x: is_fusion_compatible(x), itertools.product(*tile_cfgs.values())))

    runtime_cost: list[list[list[list[float]]]] = []
    for e, exp in enumerate(workloads):
        runtime_cost.append([])
        for i, w in enumerate(exp):
            runtime_cost[e].append([])
            for j, qcfg in enumerate(strategy_list):
                runtime_cost[e][i].append([])
                if qcfg == "w2a16_g128_asym":
                    qcfg = "w2a16_g128_asym_accfp16"
                # TODO: ad-hoc patch: for weight-only quantization, use "1", for weight-activation quantization, use "3"
                if "a16" in qcfg:
                    perf_qcfg: dict[str, dict[str, float]] = performance_table[qcfg]["1"]
                else:
                    perf_qcfg: dict[str, dict[str, float]] = performance_table[qcfg]["3"]

                for k, tile_cfg in enumerate(tile_cfgs):
                    t = tile_cfg[j]
                    runtime_cost[e][i][j].append(perf_qcfg[t.__repr__()]["inc"] * w.num_tiles(t))
    return tile_cfgs, runtime_cost


def build_workloads(model_id:str, batch_range: int, offline_stats: dict, num_layers:int) -> list[list[list[ProblemShape]]]:
    '''
    return:
        workloads: list[list[list[ProblemShape]]] -- [layer_idx][exp_idx][weight_idx]
        access_freq: list[list[float]] -- [layer_idx][exp_idx]
    '''
    num_shared_experts = offline_stats["num_shared_experts"]
    task_scale = batch_range / offline_stats["num_tokens"]
    N, K = offline_stats["NK"]

    workloads: list[list[list[ProblemShape]]] = []
    expert_sizes: list[list[int]] = [] # we only need record expert size
    access_freq:list[list[float]] = []
    for layer_idx in range(num_layers):
        if model_id == "ds2" and layer_idx == 0:
            workloads.append([ProblemShape(batch_range,10944,2048),ProblemShape(batch_range,10944,2048),ProblemShape(batch_range,2048,10944)])
            expert_sizes.append([1])
            access_freq.append([1])
            continue

        layer_freq = offline_stats[f"layer-{layer_idx}"]["access_freq"]

        Ms = [int(task_scale * x) for x in layer_freq] + ([batch_range] if num_shared_experts != 0 else [])
        Ns = [N] * len(layer_freq) + ([N*num_shared_experts] if num_shared_experts != 0 else [])
        Ks = [K] * len(layer_freq) + ([K] if num_shared_experts != 0 else [])

        workloads.append([
            [ProblemShape(m,n,k),ProblemShape(m,n,k),ProblemShape(m,k,n)] for (m,n,k) in zip(Ms, Ns, Ks)
        ])
        expert_sizes.append([
            (n*k)/(N*K) for (n,k) in zip(Ns, Ks)
        ])
        access_freq.append(value_to_prob(Ms))
    return workloads, access_freq, expert_sizes


def get_strategy_bits(qcfg: str):
    qcfg_to_bits = {
        "w8a16_g-1": (8, 16),
        "w4a16_g-1": (4, 16),
        "w4a16_g128": (4.25, 16),
        "w3a16_g128": (3.25, 16),
        "w2a16_g128": (2.25, 16),
        "w1a16_g128": (1.25, 16),

        "w4a4_g-1": (4, 4),
        "w8a8_g-1": (8, 8),
        "w4a4_g128": (4.25, 4.25),
    }
    for k in qcfg_to_bits.keys():
        if k in qcfg:
            return qcfg_to_bits[k]


def solve_model_qconfig(
    model_id: str,
    strategy_loss: dict[str, ModelLoss],
    solve_mode: Literal["layer", "model"],
    wbits_bugdet: int,
    batch_range:int,
    offline_stats:dict,
    r: float,
    TOP_K: int,
    args,
):
    strategies = sorted([x for x in strategy_loss.keys()])
    strategy_bits = [get_strategy_bits(s) for s in strategies]
    strategy_wbits = [x[0] for x in strategy_bits]
    strategy_abits = [x[1] for x in strategy_bits]

    num_layers = len(next(iter(strategy_loss.values())))

    print(f">>> Model: {model_id}; num_layers: {num_layers}")
    print(f">>> Strategies: {strategies}")
    print(f">>> Strategy wbits: {strategy_wbits}")
    print(f">>> Strategy abits: {strategy_abits}")
    # print(f"Strategy loss : {strategy_loss}")
    print(f">>> batch_range: {batch_range}")
    print(f">>> Wbits budget: {wbits_bugdet}")
    print(f">>> solve_mode: {solve_mode}")
    print(f">>> r: {r}")

    # construct workloads
    workloads, access_prob, expert_sizes = build_workloads(model_id, batch_range, offline_stats, num_layers)

    if solve_mode == "model":
        return solve_model_qconfig_model_level(model_id, strategies, strategy_loss, strategy_wbits, strategy_abits, wbits_bugdet, batch_range, workloads, access_prob, expert_sizes, r, TOP_K, args)
    elif solve_mode == "layer":
        return solve_model_qconfig_layer_level(model_id, strategies, strategy_loss, strategy_wbits, strategy_abits, wbits_bugdet, batch_range, workloads, access_prob, expert_sizes, r, TOP_K, args)


def get_num_weights(model_id: str, num_layers: int):
    if model_id == "qwen2_moe":
        return num_layers * (60+4) * 3, 64 * 3
    elif model_id == "qwen2_moe_57b":
        return num_layers * (64+8) * 3, 72 * 3
    elif model_id == "mixtral":
        return num_layers * 8 * 3, 8 * 3
    elif model_id == "ds2":
        return num_layers * (64+2) * 3, (64+2) * 3
    elif model_id == "qwen3_moe_30b_a3b_instruct_2507":
        return num_layers * 128 * 3, 128 * 3
    raise ValueError(f"Unsupported model for bit budgeting: {model_id}")


def get_offline_stats(model_id: str, trace_file: str, perf_file: str):
    # get offline statistics
    with open(trace_file, "r") as f:
        trace: dict = json.load(f)
    with open(perf_file, "r") as f:
        performance_table = json.load(f)
    trace.update({"performance_table": performance_table})
    return trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="qwen3_moe_30b_a3b_instruct_2507",
        choices=[
            "qwen2_moe",
            "ds2",
            "mixtral",
            "qwen2_moe_57b",
            "qwen3_moe_30b_a3b_instruct_2507",
        ],
    )
    parser.add_argument("--qtype", type=str, choices=["rtn", "gptq", "gptq-had", "rtn-fisher", "gptq-fisher"], default="rtn")
    parser.add_argument("--batch", type=int, default=512, help="Batch size range for the model(input length).")
    parser.add_argument("--wbits", type=float, help="Averaged wbits budget for each parameter in the model.")
    parser.add_argument("--solve_mode", type=str, choices=["layer", "model"], default="layer")
    parser.add_argument("--filter_list", type=str, nargs="+", default=[], help="Filter list for quant schemes.")
    parser.add_argument("--r", type=float, default=-1, help="The weight of loss in the objective function.")
    parser.add_argument("--trace_file", type=str, default=None, help="Expert activation frequency trace")
    parser.add_argument("--perf_file", type=str, default=f"{CUR_DIR}/calib/perf/performance_table.json", help="kernel performance profile")
    parser.add_argument("--exp_alloc", action="store_true", help="use expert-level allocation instead of linear-block level")
    parser.add_argument("--loss_file", action="append", default=[], metavar="SCHEME=PATH", help="Override a calibration-loss artifact without changing legacy project_config entries.")
    parser.add_argument("--calib_dataset", type=str, default=None, help="Expected calibration dataset for artifact validation.")
    parser.add_argument("--nsamples", type=int, default=None, help="Expected calibration sample count for artifact validation.")
    parser.add_argument("--seqlen", type=int, default=None, help="Expected calibration sequence length for artifact validation.")
    parser.add_argument("--seed", type=int, default=None, help="Expected calibration seed for artifact validation.")

    args = parser.parse_args(
        # [
        #     "--model", "qwen2_moe",
        #     "--qtype", "rtn",
        #     "--solve_mode", "layer",
        #     "--wbits", "4.3",
        #     "--batch", "1024"
        # ]
    )
    model_id = args.model
    qtype = args.qtype
    batch_range = args.batch
    solve_mode = args.solve_mode
    if args.trace_file is None:
        if model_id == "qwen3_moe_30b_a3b_instruct_2507":
            args.trace_file = f"{CUR_DIR}/calib/gate/{model_id}/c4/2048/moe-gate.json"
        else:
            args.trace_file = f"{CUR_DIR}/calib/gate/{model_id}/wiki2/4096/moe-gate.json"

    print(args)
    ############################################################################
    # {quant_strategy: loss_file}
    calib_loss_list: dict[str, str] = dict(EXPERT_QUANT_LOSS[qtype][model_id])
    loss_file_overrides = parse_loss_file_overrides(args.loss_file)
    calib_loss_list.update(loss_file_overrides)
    print(
        (
            f"### Model: {ID2NAME[model_id]}\n"
            f"### Ava schemes ({len(calib_loss_list)}): {list(calib_loss_list.keys())}\n"
            f"### Using MoE Frequency Stat: `{args.trace_file}`\n"
            f"### Using Kernel profile (performance_table): `{args.perf_file}`"
        )
    )
    ############################################################################

    filter_list = [
        "w4a16_g-1_asym", "w8a8_g-1_sym",
        # "w4a16_g128_asym", "w3a16_g128_asym", "w2a16_g128_asym", "w1a16_g128_asym",
        # "w4a16_g128_asym", "w2a16_g128_asym", "w1a16_g128_asym",
        # "w8a16_g-1_asym", "w4a16_g-1_asym", "w4a16_g128_asym", "w2a16_g128_asym", "w1a16_g128_asym",
        # "w8a16_g-1_asym", "w4a16_g-1_asym", "w4a16_g128_asym", "w3a16_g128_asym", "w2a16_g128_asym", "w1a16_g128_asym",

        # "w4a4_g-1_sym", "w8a8_g-1_sym", "w4a4_g128_sym",
    ]
    if len(args.filter_list) != 0:
        filter_list = args.filter_list

    selected_loss_files = {
        qname: loss_file
        for qname, loss_file in calib_loss_list.items()
        if any(item in qname for item in filter_list)
    }
    missing_schemes = [
        item
        for item in filter_list
        if not any(item in qname for qname in selected_loss_files)
    ]
    if missing_schemes:
        raise ValueError(
            f"No calibration-loss artifact configured for: {missing_schemes}."
        )

    if (
        model_id == "qwen3_moe_30b_a3b_instruct_2507"
        and args.r != 1.0
        and any(name.startswith(("w1", "w3")) for name in filter_list)
    ):
        raise ValueError(
            "Qwen3 W1/W3 performance profiles and kernels are not present in "
            "this repository. Use --r 1.0 for the accuracy-only allocation, "
            "or add measured W1/W3 profiles before enabling co-design."
        )

    offline_stats = get_offline_stats(model_id, args.trace_file, args.perf_file)
    expected_calibration = {
        "dataset": args.calib_dataset,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "seed": args.seed,
    }
    artifact_metadata = [("trace", {"calibration": offline_stats.get("calibration")})]
    artifact_metadata.extend(
        (
            f"loss[{qname}]",
            load_optional_metadata(loss_file, artifact_name=f"loss[{qname}]"),
        )
        for qname, loss_file in selected_loss_files.items()
    )
    calibration_hash = validate_calibration_artifacts(
        artifact_metadata, expected_calibration
    )

    expected_model = ID2NAME[model_id]
    loss_models = {
        str(metadata["model"])
        for artifact_name, metadata in artifact_metadata
        if artifact_name.startswith("loss[")
        and metadata is not None
        and metadata.get("model") is not None
    }
    if loss_models and loss_models != {expected_model}:
        raise ValueError(
            f"Calibration-loss model mismatch: expected {expected_model!r}, "
            f"got {sorted(loss_models)!r}."
        )

    strategy_loss = get_strategy_loss(selected_loss_files, filter_list, qtype)
    num_layers = len(next(iter(strategy_loss.values())))

    print(f"### performance_table: {offline_stats['performance_table'].keys()}")

    model_num_weights, layer_num_weights = get_num_weights(model_id, num_layers)
    wbits_bugdet = args.wbits * (model_num_weights if args.solve_mode == "model" else layer_num_weights)

    if args.r == -1:
        r_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        r_list = [args.r]

    for r in r_list:
        model_qconfig = solve_model_qconfig(
            model_id,
            strategy_loss,
            solve_mode=solve_mode,
            wbits_bugdet=wbits_bugdet,
            batch_range=batch_range,
            offline_stats=offline_stats,
            r=r,
            TOP_K=10,
            args=args,
        )

        mxdir = "+".join(filter_list)
        exp_alloc = "" if not args.exp_alloc else "_exp_alloc"
        qconfig_path = f"{CUR_DIR}/qconfigs/{mxdir}/{model_id}_{qtype}_S{solve_mode}_bs{batch_range}_wbits{args.wbits}_r{r}{exp_alloc}.json"
        export_qconfig(model_qconfig, qconfig_path)
        qconfig_calibration = {
            key: value
            for key, value in expected_calibration.items()
            if value is not None
        }
        if calibration_hash is not None:
            qconfig_calibration["input_ids_sha256"] = calibration_hash
        write_metadata(
            qconfig_path,
            {
                "format": "mxmoe-allocation-v1",
                "model_id": model_id,
                "model": ID2NAME[model_id],
                "qtype": qtype,
                "calibration": qconfig_calibration,
                "trace_file": os.path.abspath(args.trace_file),
                "perf_file": os.path.abspath(args.perf_file),
                "loss_files": {
                    qname: os.path.abspath(path)
                    for qname, path in selected_loss_files.items()
                },
                "filter_list": filter_list,
                "solve_mode": solve_mode,
                "batch": batch_range,
                "effective_wbits": args.wbits,
                "accuracy_weight": r,
                "expert_level_allocation": args.exp_alloc,
            },
        )
