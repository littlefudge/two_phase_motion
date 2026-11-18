#!/usr/bin/env python3

import argparse
import math
import os
import pickle
from pathlib import Path

import numpy as np
import yaml
from scipy.integrate import solve_ivp


def get_config(fname):
    path = Path(fname)
    if not path.is_file():
        raise FileNotFoundError("Configuration file not found: {}".format(path))
    with path.open("r", encoding="utf-8") as stream:
        conf = yaml.load(stream, Loader=yaml.FullLoader)
    if not isinstance(conf, dict):
        raise ValueError("YAML configuration must be a mapping")
    return conf


def build_grid(params):
    nx = int(params["n_u"])
    ny = int(params["n_v"])
    nz = int(params["n_r"])
    if min(nx, ny, nz) < 1:
        raise ValueError("n_u, n_v and n_r must be positive")

    lu0, lu1 = float(params["L_u0"]), float(params["L_u1"])
    lv0, lv1 = float(params["L_v0"]), float(params["L_v1"])
    lr0, lr1 = float(params["L_r0"]), float(params["L_r1"])
    logu, logv, logr = int(params["logu"]), int(params["logv"]), int(params["logr"])

    def axis(lo, hi, n, logarithmic):
        if n == 1:
            values = [(lo + hi) / 2.0]
        else:
            values = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        if logarithmic:
            values = [10.0 ** value for value in values]
        return values

    u_grid = axis(lu0, lu1, nx, logu == 1)
    v_grid = axis(lv0, lv1, ny, logv == 1)

    a = float(params["const"]["a"])
    mu = float(params["const"]["mu"])
    tau = float(params["const"]["tau"])
    if logr == 10:
        r_cent = [2.0 * a * tau * v / (mu * (a - v)) if a != v else 0.0 for v in v_grid]
        r_grid = []
        for center in r_cent:
            dr = (lr1 - lr0) / (nz - 1) if nz > 1 else 0.0
            r_grid.append([center - (lr1 - lr0) / 2.0 + dr * k + 0.000003 for k in range(nz)])
    else:
        r_grid = axis(lr0, lr1, nz, logr == 1)
    return u_grid, v_grid, r_grid


def get_pos(nx, ny, nz, index):
    """Convert a zero-based flat grid index to (u, v, r) indices."""
    if index < 0 or index >= nx * ny * nz:
        raise IndexError("Grid index outside bounds")
    i = index // (ny * nz)
    remainder = index % (ny * nz)
    j = remainder // nz
    k = remainder % nz
    return i, j, k


def get_share(size, rank, nx, ny, nz):
    """Return a zero-based, end-exclusive interval for an MPI rank."""
    if size < 1 or rank < 0 or rank >= size:
        raise ValueError("Invalid MPI size/rank")
    total = nx * ny * nz
    if total == 0:
        return 0, 0
    start, end = divmod(total, size)
    chunk = start + (1 if rank < end else 0)
    offset = rank * start + min(rank, end)
    return offset, offset + chunk


def hydro(t, state, a, mu, tau):
    q1, q2, r = state
    return [
        q1 * (2.0 * a * a * tau * q2 - mu * (q1 - q2) * q1 * r),
        (q1 - q2) * r * (q1 * q1 - a * a),
        tau * q2 * r * (q1 * q1 - a * a),
    ]


def hydroj(t, state, a, mu, tau):
    q1, q2, r = state
    return [
        [2*a*a*tau*q2 - 3*mu*q1*q1*r + 2*mu*q1*q2*r,
         2*a*a*tau*q1 + mu*q1*q1*r,
         -mu*q1*q1*q1 + mu*q1*q1*q2],
        [3*r*q1*q1 - a*a*r - 2*r*q1*q2,
         -r*q1*q1 + a*a*r,
         q1*q1*q1 - a*a*q1 - q1*q1*q2 + a*a*q2],
        [2*tau*q1*q2*r,
         tau*q1*q1*r - tau*a*a*r,
         tau*q1*q1*q2 - tau*a*a*q2],
    ]


def make_events(r_infty, a):
    def r_inf(t, state, *args):
        return state[2] - r_infty
    r_inf.terminal = True
    r_inf.direction = 1

    def q2_0(t, state, *args):
        return state[1]
    q2_0.terminal = True
    q2_0.direction = -1

    def q1_pa(t, state, *args):
        return state[0] - a
    q1_pa.terminal = False
    q1_pa.direction = 1

    def q1_ma(t, state, *args):
        return state[0] - a
    q1_ma.terminal = False
    q1_ma.direction = -1

    def q2_pa(t, state, *args):
        return state[1] - a
    q2_pa.terminal = False
    q2_pa.direction = 1

    def q2_ma(t, state, *args):
        return state[1] - a
    q2_ma.terminal = False
    q2_ma.direction = -1

    return (q1_ma, q1_pa, q2_ma, q2_pa, q2_0, r_inf)


def is_white(point, a, mu, tau, epsilon, white_bounds):
    q1, q2, r = point
    if abs(a - q2) < 1e-14:
        return {"line": False, "int": False, "dist": float("inf")}
    analytical_r = 2.0 * a * tau * q2 / (mu * (a - q2))
    distance = math.sqrt((a - q1)**2 + (r - analytical_r)**2)
    on_line = distance < epsilon
    in_interval = white_bounds[0] <= q2 <= white_bounds[1]
    return {"line": on_line, "int": in_interval, "dist": distance}


def postprocess(solution, event_names):
    result = {}
    for name, times, states in zip(event_names, solution.t_events, solution.y_events):
        if len(times):
            result[name] = {"time": float(times[-1]), "vars": np.asarray(states[-1], dtype=float)}
    if solution.status == 1:
        last_time = solution.t[-1]
        last_vars = np.asarray(solution.y[:, -1], dtype=float)
    elif solution.status == 0:
        last_time = solution.t[-1]
        last_vars = np.asarray(solution.y[:, -1], dtype=float)
    else:
        last_time = solution.t[-1] if len(solution.t) else 0.0
        last_vars = np.asarray(solution.y[:, -1], dtype=float) if len(solution.t) else np.asarray([np.nan]*3)
    result["last"] = {"time": float(last_time), "vars": last_vars}
    return result


def solver(init_point, sys_pars, t_minus, t_plus, events):
    a, mu, tau = sys_pars
    common = dict(method="RK45", args=sys_pars, dense_output=False,
                  atol=1e-9, rtol=1e-6, events=events)
    plus = solve_ivp(hydro, (0.0, t_plus), init_point, **common)
    minus = solve_ivp(hydro, (0.0, t_minus), init_point, **common)
    event_names = ("q1_ma", "q1_pa", "q2_ma", "q2_pa", "q2_0", "r_inf")
    return {"plus": postprocess(plus, event_names),
            "minus": postprocess(minus, event_names)}


def classify(init_point, sys_pars, t_minus, t_plus, events, epsilon,
             white_bounds, delta, r_infty):
    del r_infty
    a, mu, tau = sys_pars
    result = solver(init_point, sys_pars, t_minus, t_plus, events)

    if init_point[0] <= a and init_point[1] >= a:
        color = "red"
    elif init_point[0] >= a and init_point[1] >= a:
        if "q2_0" in result["minus"] and "q2_ma" in result["minus"]:
            color = "green"
        elif "q1_ma" in result["minus"]:
            color = "red"
        else:
            color = "smth"
    else:
        mpoint = result["minus"]["last"]["vars"]
        ppoint = result["plus"]["last"]["vars"]
        res = is_white(ppoint, a, mu, tau, epsilon, white_bounds)

        if init_point[0] >= a:
            if res["line"] and res["int"]:
                color = "white_0" if "q2_0" in result["minus"] else "white_1"
            elif ppoint[0] <= a:
                color = "black"
            elif mpoint[0] <= a:
                color = "red"
            elif "q2_0" in result["minus"]:
                color = "green"
            else:
                color = "smth"
        else:
            if res["line"] and res["int"]:
                color = "white_0" if "q2_0" in result["minus"] else "white_1"
            elif ppoint[0] > a or ppoint[1] > a:
                color = "red"
            elif mpoint[0] > a:
                color = "black"
            elif "q2_0" in result["plus"] and mpoint[0] < delta * init_point[0]:
                color = "blue"
            else:
                color = "smth"

    init = np.asarray([0.0] + list(init_point), dtype=float)
    pend = np.asarray([result["plus"]["last"]["time"]] + list(result["plus"]["last"]["vars"]), dtype=float)
    mend = np.asarray([result["minus"]["last"]["time"]] + list(result["minus"]["last"]["vars"]), dtype=float)
    return color, init, pend, mend


def empty_points():
    return {color: {"init": [], "pend": [], "mend": []}
            for color in ("red", "green", "blue", "black",
                          "white_0", "white_1", "smth")}


def write_points(points, folder, rank):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for color, values in points.items():
        init = np.asarray(values["init"], dtype=float)
        if init.size:
            init = init[np.argsort(init[:, 1])]
            np.savetxt(folder / ("q1q2r_{}_{:03d}.txt".format(color, rank)),
                       init, fmt="%+1.8e", delimiter=" ", newline="\n",
                       encoding="UTF-8")
    with (folder / "q1q2r_{:03d}.pkl".format(rank)).open("wb") as stream:
        pickle.dump(points, stream)


def main():
    parser = argparse.ArgumentParser(description="Classify a three-dimensional point grid")
    parser.add_argument("--config", default="./params.yaml")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--size", type=int)
    args = parser.parse_args()

    rank = args.rank if args.rank is not None else int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    size = args.size if args.size is not None else int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))

    params = get_config(args.config)
    a = float(params["const"]["a"])
    mu = float(params["const"]["mu"])
    tau = float(params["const"]["tau"])
    sys_pars = (a, mu, tau)

    u_grid, v_grid, r_grid = build_grid(params)
    nx, ny = len(u_grid), len(v_grid)
    nz = int(params["n_r"])

    r_infty = float(params["infty"])
    epsilon = float(params["eps_w"])
    delta = float(params["delta_blue"])
    white_bounds = (a/2*(1-math.sqrt(1-8/mu)), a/2*(1+math.sqrt(1-8/mu))) if mu > 8 else (-float("inf"), float("inf"))
    events = make_events(r_infty, a)
    start, end = get_share(size, rank, nx, ny, nz)

    points = empty_points()
    for flat_index in range(start, end):
        i, j, k = get_pos(nx, ny, nz, flat_index)
        r_value = r_grid[j][k] if isinstance(r_grid, list) and r_grid and isinstance(r_grid[0], list) else r_grid[k]
        init_point = (u_grid[i], v_grid[j], r_value)
        color, init, pend, mend = classify(
            init_point, sys_pars, float(params["glob_left"]),
            float(params["glob_right"]), events, epsilon,
            white_bounds, delta, r_infty)
        points[color]["init"].append(init)
        points[color]["pend"].append(pend)
        points[color]["mend"].append(mend)

    write_points(points, params["saveto"], rank)
    print("Rank {} processed {} points".format(rank, end - start))


if __name__ == "__main__":
    main()
