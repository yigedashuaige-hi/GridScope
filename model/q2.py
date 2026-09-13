#!/usr/bin/env python3
"""问题二：风险感知日前规划与 10 min 因果滚动控制。

计算层输出 JSON；若同时给出 result2 模板和输出路径，则调用同目录
build_result2.mjs 将数值写入官方 Excel 模板。

模型口径：
1. 所有功率按 1/6 h 换算为 kWh；
2. C、D 均为微网母线侧电量；
3. 2025-01-01 00:00 的 E=6000 kWh，SOC 跨日连续；
4. 日前计划使用 36 h（24 h 正式合同 + 12 h 影子时域）；
5. 计划只能使用日期早于当前日的数据；实时层只使用当前及此前观测；
6. 紧急购电只补负荷缺口，不用于主动给储能充电；
7. 每天0:00用最近28个已发生日滚动选择预测器，风险残差单独使用最近56天；
8. 风险分位水平用1月严格因果的完整闭环回放重新校准。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csc_matrix, lil_matrix


SLOTS_PER_DAY = 144
DELTA_H = 1.0 / 6.0
PLAN_HORIZON = 216              # 36 h
FORMAL_HORIZON = 144            # 24 h
MPC_HORIZON = 36                # 6 h
E_MIN = 1200.0
E_MAX = 10800.0
E_INITIAL = 6000.0
ETA_C = 0.90
ETA_D = 0.90
Q_MAX = 5000.0 * DELTA_H        # 833.333... kWh/10 min
ALPHA_CANDIDATES = (0.70, 0.75, 0.80, 0.85, 0.90)
FORECAST_CANDIDATES = ("last_week", "weekday_median", "weighted_blend")
FORECAST_SELECTION_WINDOW = 28
RISK_RESIDUAL_WINDOW = 56
EPS_THROUGHPUT = 1.0e-5
EMERGENCY_MULTIPLIER = 5.0
UNDERPURCHASE_EXTRA_MULTIPLIER = EMERGENCY_MULTIPLIER - 1.0
DEFAULT_ALPHA = 0.80
DEFAULT_TERMINAL_SHADOW_FACTOR = 0.45


@dataclass
class InputData:
    dates: pd.DatetimeIndex
    time_headers: list[str]
    load: np.ndarray
    pv: np.ndarray
    net: np.ndarray
    price: np.ndarray
    reference_net: np.ndarray


@dataclass
class PlanResult:
    grid: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    spill: np.ndarray
    state: np.ndarray
    objective: float
    mip_gap: float


@dataclass
class DayResult:
    grid: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    emergency: np.ndarray
    spill: np.ndarray
    state: np.ndarray
    plan_state: np.ndarray
    plan_terminal_state: float
    planned_cost: float
    emergency_cost: float


def _format_time_header(value: object) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M").lstrip("0") or "0:00"
    return str(value)


def load_inputs(processed_path: Path, price_path: Path) -> InputData:
    load_df = pd.read_excel(processed_path, sheet_name="小区负载")
    pv_df = pd.read_excel(processed_path, sheet_name="光伏发电实际功率")
    if load_df.shape != pv_df.shape or load_df.shape[1] != SLOTS_PER_DAY + 1:
        raise ValueError("处理后附件2应包含365行日期及144个10分钟数据列。")

    dates = pd.DatetimeIndex(pd.to_datetime(load_df.iloc[:, 0])).normalize()
    if not dates.equals(pd.DatetimeIndex(pd.to_datetime(pv_df.iloc[:, 0])).normalize()):
        raise ValueError("负荷与光伏工作表的日期不一致。")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("附件2日期必须按天递增且不能重复。")

    load_kw = load_df.iloc[:, 1:].to_numpy(dtype=float)
    pv_kw = pv_df.iloc[:, 1:].to_numpy(dtype=float)
    if not np.isfinite(load_kw).all() or not np.isfinite(pv_kw).all():
        raise ValueError("附件2存在空值或非数值。")

    a1 = pd.read_excel(price_path)
    if len(a1) != SLOTS_PER_DAY:
        raise ValueError("附件1应含144个时段。")
    price = pd.to_numeric(a1.iloc[:, 1], errors="raise").to_numpy(dtype=float)
    reference_load = pd.to_numeric(a1.iloc[:, 2], errors="raise").to_numpy(dtype=float) * DELTA_H
    reference_pv = pd.to_numeric(a1.iloc[:, 3], errors="raise").to_numpy(dtype=float) * DELTA_H

    return InputData(
        dates=dates,
        time_headers=[_format_time_header(x) for x in load_df.columns[1:]],
        load=load_kw * DELTA_H,
        pv=pv_kw * DELTA_H,
        net=(load_kw - pv_kw) * DELTA_H,
        price=price,
        reference_net=reference_load - reference_pv,
    )


class CausalForecaster:
    """只用 cutoff 之前的完整日数据预测 target_day 的净负荷。"""

    def __init__(self, data: InputData):
        self.data = data
        self._cache: dict[tuple[int, int, str], np.ndarray] = {}

    def _fallback(self, cutoff: int) -> np.ndarray:
        if cutoff <= 0:
            return self.data.reference_net.copy()
        recent = self.data.net[max(0, cutoff - 7):cutoff]
        recent_profile = np.median(recent, axis=0)
        if cutoff < 3:
            w = cutoff / 3.0
            return w * recent_profile + (1.0 - w) * self.data.reference_net
        return recent_profile

    def forecast(self, target_day: int, cutoff: int, method: str) -> np.ndarray:
        key = (target_day, cutoff, method)
        if key in self._cache:
            return self._cache[key].copy()
        if method not in FORECAST_CANDIDATES:
            raise ValueError(f"未知预测方法: {method}")
        if cutoff < 0 or cutoff > len(self.data.dates):
            raise ValueError("cutoff超出日期范围。")

        fallback = self._fallback(cutoff)
        target_date = self.data.dates[0] + pd.Timedelta(days=int(target_day))
        target_dow = target_date.dayofweek

        lw_idx = target_day - 7
        if 0 <= lw_idx < cutoff:
            last_week = self.data.net[lw_idx]
        else:
            same_before = [i for i in range(cutoff - 1, -1, -1)
                           if self.data.dates[i].dayofweek == target_dow]
            last_week = self.data.net[same_before[0]] if same_before else fallback

        same_weekdays = [i for i in range(cutoff)
                         if self.data.dates[i].dayofweek == target_dow][-8:]
        weekday_median = (np.median(self.data.net[same_weekdays], axis=0)
                          if same_weekdays else fallback)

        if method == "last_week":
            pred = last_week
        elif method == "weekday_median":
            pred = weekday_median
        else:  # weighted_blend：固定权重组合，权重不使用当日信息更新
            recent = (np.median(self.data.net[max(0, cutoff - 7):cutoff], axis=0)
                      if cutoff else fallback)
            pred = 0.55 * last_week + 0.30 * weekday_median + 0.15 * recent

        pred = np.clip(np.asarray(pred, dtype=float), -1800.0, 1800.0)
        self._cache[key] = pred.copy()
        return pred

    def day_forecast_matrix(self, method: str) -> np.ndarray:
        return np.vstack([
            self.forecast(d, d, method) for d in range(len(self.data.dates))
        ])

    def horizon_forecast(self, day: int, method: str) -> np.ndarray:
        today = self.forecast(day, day, method)
        tomorrow = self.forecast(day + 1, day, method)
        return np.concatenate([today, tomorrow[:PLAN_HORIZON - SLOTS_PER_DAY]])


def build_daily_forecast_policy(
    data: InputData,
    forecaster: CausalForecaster,
) -> tuple[list[str], list[dict], np.ndarray, dict[str, np.ndarray]]:
    """生成每日0:00的严格因果预测器选择路径。

    候选预测在历史日 h 上的评分，使用的也是当日 h 实际决策时能够
    得到的 forecast(h, cutoff=h)，而不是当前日回看时重新拟合的预测。
    历史不足7个完整日时使用固定权重组合冷启动；之后用最近28天滚动评分。
    """
    n_days = len(data.dates)
    candidate_predictions = {
        method: np.vstack([
            forecaster.forecast(day, day, method) for day in range(n_days)
        ])
        for method in FORECAST_CANDIDATES
    }
    selected_methods: list[str] = []
    daily_records: list[dict] = []
    selected_predictions = np.zeros_like(data.net)

    for day in range(n_days):
        hist_start = max(0, day - FORECAST_SELECTION_WINDOW)
        hist_days = np.arange(hist_start, day, dtype=int)
        scores: dict[str, dict[str, float]] = {}

        if day < 7:
            selected = "weighted_blend"
            rule = "cold_start"
        else:
            for method in FORECAST_CANDIDATES:
                pred = candidate_predictions[method][hist_days]
                actual = data.net[hist_days]
                err = actual - pred
                economic_loss = float(np.sum(
                    data.price[None, :] * (
                        np.maximum(-err, 0.0)
                        + UNDERPURCHASE_EXTRA_MULTIPLIER * np.maximum(err, 0.0)
                    )
                ))
                scores[method] = {
                    "economic_loss": economic_loss,
                    "mean_economic_loss_per_day": economic_loss / len(hist_days),
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err ** 2))),
                }
            # 非对称经济损失是主判据，MAE、RMSE仅用于依次破除并列。
            selected = min(
                FORECAST_CANDIDATES,
                key=lambda method: (
                    scores[method]["economic_loss"],
                    scores[method]["mae"],
                    scores[method]["rmse"],
                    FORECAST_CANDIDATES.index(method),
                ),
            )
            rule = "rolling_28d"

        selected_methods.append(selected)
        selected_predictions[day] = candidate_predictions[selected][day]
        daily_records.append({
            "date": data.dates[day].strftime("%Y-%m-%d"),
            "selected_method": selected,
            "selection_rule": rule,
            "history_start": (data.dates[hist_start].strftime("%Y-%m-%d")
                              if len(hist_days) else None),
            "history_end": (data.dates[day - 1].strftime("%Y-%m-%d")
                            if len(hist_days) else None),
            "history_days": int(len(hist_days)),
            "candidate_scores": scores,
        })

    return selected_methods, daily_records, selected_predictions, candidate_predictions


def risk_adjustment(
    day: int,
    center_36h: np.ndarray,
    residual_history: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按四个粗时段池化标准化残差，构造净负荷上分位风险裕量。"""
    if day <= 0:
        sigma = 0.12 * np.abs(center_36h) + 25.0
        q = np.full_like(center_36h, 0.8416212335729143)
        margin = np.minimum(3.0 * sigma, np.maximum(0.0, q * sigma))
        return center_36h + margin, sigma, q

    hist = residual_history[max(0, day - RISK_RESIDUAL_WINDOW):day]
    if len(hist) < 4:
        sigma_slots = 0.12 * np.abs(center_36h[:SLOTS_PER_DAY]) + 25.0
        q_blocks = np.full(4, 0.8416212335729143)
    else:
        median = np.median(hist, axis=0)
        mad = 1.4826 * np.median(np.abs(hist - median), axis=0)
        std = np.std(hist, axis=0, ddof=1)
        sigma_slots = np.maximum(np.maximum(mad, 0.65 * std), 15.0)
        # 按论文定义 u=e/sigma 标准化，不另行中心化；这样历史系统偏差
        # 会进入分位数并由风险裕量自动修正。
        standardized = hist / sigma_slots[None, :]
        q_blocks = np.empty(4)
        for b in range(4):
            vals = standardized[:, b * 36:(b + 1) * 36].ravel()
            q_blocks[b] = float(np.quantile(vals, alpha))

    sigma_36 = np.concatenate([sigma_slots, sigma_slots[:72]])
    block_index = np.concatenate([np.arange(SLOTS_PER_DAY) // 36, np.arange(72) // 36])
    q_36 = q_blocks[block_index]
    # 与论文完全一致：M=min{3*sigma, max(0, q_alpha*sigma)}。
    margin = np.minimum(3.0 * sigma_36, np.maximum(0.0, q_36 * sigma_36))
    return center_36h + margin, sigma_36, q_36


def solve_day_ahead(
    risk_net: np.ndarray,
    price: np.ndarray,
    e0: float,
    use_milp: bool = True,
    enforce_terminal_floor: bool = True,
) -> PlanResult:
    h = len(risk_net)
    if len(price) != h:
        raise ValueError("日前电价与净负荷长度不一致。")

    g0, c0, d0, w0, e0i, z0 = 0, h, 2*h, 3*h, 4*h, 5*h + 1
    nvar = 6*h + 1
    obj = np.zeros(nvar)
    obj[g0:g0+h] = price
    obj[c0:c0+h] = EPS_THROUGHPUT
    obj[d0:d0+h] = EPS_THROUGHPUT

    lower = np.zeros(nvar)
    upper = np.full(nvar, np.inf)
    upper[c0:c0+h] = Q_MAX
    upper[d0:d0+h] = Q_MAX
    lower[e0i:e0i+h+1] = E_MIN
    upper[e0i:e0i+h+1] = E_MAX
    lower[e0i] = upper[e0i] = float(e0)
    upper[z0:z0+h] = 1.0

    rows = 4*h + 1
    A = lil_matrix((rows, nvar), dtype=float)
    lb = np.full(rows, -np.inf)
    ub = np.full(rows, np.inf)

    for k in range(h):
        # G + D - C - W = N
        A[k, g0+k] = 1.0
        A[k, c0+k] = -1.0
        A[k, d0+k] = 1.0
        A[k, w0+k] = -1.0
        lb[k] = ub[k] = risk_net[k]

        # E_{k+1} - E_k - eta_c*C + D/eta_d = 0
        r = h + k
        A[r, c0+k] = -ETA_C
        A[r, d0+k] = 1.0 / ETA_D
        A[r, e0i+k] = -1.0
        A[r, e0i+k+1] = 1.0
        lb[r] = ub[r] = 0.0

        # C <= Qmax*z; D <= Qmax*(1-z)
        r = 2*h + k
        A[r, c0+k] = 1.0
        A[r, z0+k] = -Q_MAX
        ub[r] = 0.0
        r = 3*h + k
        A[r, d0+k] = 1.0
        A[r, z0+k] = Q_MAX
        ub[r] = Q_MAX

    # 影子时域末端储能不得低于本次优化起点，抑制36h边界放空。
    A[4*h, e0i+h] = 1.0
    if enforce_terminal_floor:
        lb[4*h] = float(e0)

    A_csc = csc_matrix(A)
    constraints = LinearConstraint(A_csc, lb, ub)
    bounds = Bounds(lower, upper)
    if use_milp:
        integrality = np.zeros(nvar, dtype=np.int8)
        integrality[z0:z0+h] = 1
        res = milp(
            c=obj,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"mip_rel_gap": 1e-8, "time_limit": 30.0, "presolve": True},
        )
        gap = float(getattr(res, "mip_gap", 0.0) or 0.0)
    else:
        # integrality全为0时，scipy.milp直接调用HiGHS线性规划求解器，
        # 同时保留双侧LinearConstraint（含影子时域末端下界）。
        res = milp(
            c=obj,
            integrality=np.zeros(nvar, dtype=np.int8),
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": 30.0, "presolve": True},
        )
        gap = 0.0
    if not res.success or res.x is None:
        raise RuntimeError(f"日前优化失败: {res.message}")
    x = res.x
    return PlanResult(
        grid=x[g0:g0+h].copy(),
        charge=x[c0:c0+h].copy(),
        discharge=x[d0:d0+h].copy(),
        spill=x[w0:w0+h].copy(),
        state=x[e0i:e0i+h+1].copy(),
        objective=float(np.dot(price, x[g0:g0+h])),
        mip_gap=gap,
    )


_MPC_MATRIX_CACHE: dict[int, tuple[csc_matrix, np.ndarray, np.ndarray]] = {}


def _mpc_matrix(h: int) -> tuple[csc_matrix, np.ndarray, np.ndarray]:
    if h in _MPC_MATRIX_CACHE:
        return _MPC_MATRIX_CACHE[h]
    c0, d0, h0, w0, e0i, sp, sm = 0, h, 2*h, 3*h, 4*h, 5*h+1, 5*h+2
    nvar = 5*h + 3
    A = lil_matrix((2*h + 1, nvar), dtype=float)
    # balance and SOC rows
    for k in range(h):
        A[k, c0+k] = -1.0
        A[k, d0+k] = 1.0
        A[k, h0+k] = 1.0
        A[k, w0+k] = -1.0
        r = h+k
        A[r, c0+k] = -ETA_C
        A[r, d0+k] = 1.0 / ETA_D
        A[r, e0i+k] = -1.0
        A[r, e0i+k+1] = 1.0
    # E_H - s_plus + s_minus = target
    A[2*h, e0i+h] = 1.0
    A[2*h, sp] = -1.0
    A[2*h, sm] = 1.0
    lb = np.zeros(2*h + 1)
    ub = np.zeros(2*h + 1)
    out = (csc_matrix(A), lb, ub)
    _MPC_MATRIX_CACHE[h] = out
    return out


def solve_mpc_step(
    net_forecast: np.ndarray,
    grid: np.ndarray,
    price: np.ndarray,
    e_now: float,
    terminal_target: float,
    terminal_shadow_factor: float = DEFAULT_TERMINAL_SHADOW_FACTOR,
) -> tuple[float, float, float, float, float]:
    h = len(net_forecast)
    A, lb0, ub0 = _mpc_matrix(h)
    c0, d0, h0, w0, e0i, sp, sm = 0, h, 2*h, 3*h, 4*h, 5*h+1, 5*h+2
    nvar = 5*h + 3

    obj = np.zeros(nvar)
    obj[c0:c0+h] = EPS_THROUGHPUT
    obj[d0:d0+h] = EPS_THROUGHPUT
    obj[h0:h0+h] = EMERGENCY_MULTIPLIER * price
    # 末端电量的影子价值来自未来高价紧急电，超额电量仅给极小惩罚。
    energy_value = ETA_D * EMERGENCY_MULTIPLIER * float(np.quantile(price, 0.75))
    obj[sp] = 1.0e-6
    obj[sm] = float(terminal_shadow_factor) * energy_value

    lower = np.zeros(nvar)
    upper = np.full(nvar, np.inf)
    surplus = np.maximum(grid - net_forecast, 0.0)
    upper[c0:c0+h] = np.minimum(Q_MAX, surplus)
    upper[d0:d0+h] = Q_MAX
    lower[e0i:e0i+h+1] = E_MIN
    upper[e0i:e0i+h+1] = E_MAX
    lower[e0i] = upper[e0i] = float(e_now)

    lb = lb0.copy()
    ub = ub0.copy()
    lb[:h] = ub[:h] = net_forecast - grid
    lb[2*h] = ub[2*h] = float(terminal_target)
    res = linprog(obj, A_eq=A, b_eq=lb, bounds=list(zip(lower, upper)), method="highs")
    if not res.success or res.x is None:
        raise RuntimeError(f"实时MPC失败: {res.message}")
    x = res.x
    c, d, emergency, spill = x[c0], x[d0], x[h0], x[w0]
    e_next = x[e0i+1]
    return float(c), float(d), float(emergency), float(spill), float(e_next)


def simulate_day(
    data: InputData,
    forecaster: CausalForecaster,
    residual_matrix: np.ndarray,
    day: int,
    method: str,
    alpha: float,
    e_start: float,
    use_milp: bool = True,
    mpc_horizon: int = MPC_HORIZON,
    terminal_shadow_factor: float = DEFAULT_TERMINAL_SHADOW_FACTOR,
    enforce_terminal_floor: bool = True,
) -> DayResult:
    center = forecaster.horizon_forecast(day, method)
    risk_net, _, _ = risk_adjustment(day, center, residual_matrix, alpha)
    price_36 = np.tile(data.price, 2)[:PLAN_HORIZON]
    plan = solve_day_ahead(
        risk_net, price_36, e_start, use_milp=use_milp,
        enforce_terminal_floor=enforce_terminal_floor,
    )

    actual_net = data.net[day]
    C = np.zeros(SLOTS_PER_DAY)
    D = np.zeros(SLOTS_PER_DAY)
    H = np.zeros(SLOTS_PER_DAY)
    W = np.zeros(SLOTS_PER_DAY)
    E = np.zeros(SLOTS_PER_DAY + 1)
    E[0] = e_start
    observed_residuals: list[float] = []

    for t in range(SLOTS_PER_DAY):
        h = min(mpc_horizon, PLAN_HORIZON - t)
        center_h = center[t:t+h].copy()
        observed_residuals.append(float(actual_net[t] - center[t]))
        recent = np.asarray(observed_residuals[-6:])
        weights = 0.70 ** np.arange(len(recent) - 1, -1, -1)
        bias = float(np.dot(weights, recent) / weights.sum())
        decay = np.exp(-np.arange(h) / 18.0)
        online_net = center_h + bias * decay
        online_net[0] = actual_net[t]
        c, d, emergency, spill, e_next = solve_mpc_step(
            online_net,
            plan.grid[t:t+h],
            price_36[t:t+h],
            E[t],
            plan.state[t+h],
            terminal_shadow_factor=terminal_shadow_factor,
        )
        C[t], D[t], H[t], W[t], E[t+1] = c, d, emergency, spill, e_next

    return DayResult(
        grid=plan.grid[:SLOTS_PER_DAY].copy(),
        charge=C,
        discharge=D,
        emergency=H,
        spill=W,
        state=E,
        plan_state=plan.state[:SLOTS_PER_DAY+1].copy(),
        plan_terminal_state=float(plan.state[-1]),
        planned_cost=float(np.dot(data.price, plan.grid[:SLOTS_PER_DAY])),
        emergency_cost=float(np.dot(EMERGENCY_MULTIPLIER * data.price, H)),
    )


def simulate_range(
    data: InputData,
    forecaster: CausalForecaster,
    daily_methods: list[str],
    residual_matrix: np.ndarray,
    alpha: float,
    start_day: int,
    end_day: int,
    e_start: float,
    *,
    use_milp: bool,
    mpc_horizon: int,
    terminal_shadow_factor: float = DEFAULT_TERMINAL_SHADOW_FACTOR,
    enforce_terminal_floor: bool = True,
    progress_label: str = "",
) -> tuple[list[DayResult], float]:
    results: list[DayResult] = []
    e = float(e_start)
    for pos, day in enumerate(range(start_day, end_day)):
        result = simulate_day(
            data, forecaster, residual_matrix, day, daily_methods[day], alpha, e,
            use_milp=use_milp, mpc_horizon=mpc_horizon,
            terminal_shadow_factor=terminal_shadow_factor,
            enforce_terminal_floor=enforce_terminal_floor,
        )
        results.append(result)
        e = float(result.state[-1])
        if progress_label and ((pos + 1) % 31 == 0 or day + 1 == end_day):
            print(f"{progress_label}: {pos+1}/{end_day-start_day} days", flush=True)
    return results, e


def calibrate_alpha(
    data: InputData,
    forecaster: CausalForecaster,
    daily_methods: list[str],
    residual_matrix: np.ndarray,
    *,
    fast: bool = False,
) -> tuple[float, dict[str, dict[str, float]]]:
    """用1月因果滚动预测+完整闭环成本重新校准风险分位。

    每个候选alpha都从1月1日 E=6000 kWh开始，完整重放36 h日前规划、
    10 min因果MPC及紧急购电。只有1月31日结束后才选定alpha*，
    因而该参数从2月1日起使用，不回写1月真实预热轨迹。
    """
    detail: dict[str, dict[str, float]] = {}
    for alpha in ALPHA_CANDIDATES:
        results, _ = simulate_range(
            data, forecaster, daily_methods, residual_matrix, alpha,
            0, min(31, len(data.dates)), E_INITIAL,
            use_milp=not fast, mpc_horizon=6 if fast else MPC_HORIZON,
        )
        planned = sum(x.planned_cost for x in results)
        emergency = sum(x.emergency_cost for x in results)
        emergency_energy = sum(float(x.emergency.sum()) for x in results)
        states = np.concatenate([x.state for x in results])
        detail[f"{alpha:.2f}"] = {
            "planned_cost": planned,
            "emergency_cost": emergency,
            "total_cost": planned + emergency,
            "emergency_energy": emergency_energy,
            "min_soc_kwh": float(states.min()),
            "max_soc_kwh": float(states.max()),
            "mean_plan_terminal_soc_kwh": float(np.mean([
                x.plan_terminal_state for x in results
            ])),
            "days_plan_terminal_floor_binding": int(sum(
                abs(x.plan_terminal_state - x.state[0]) <= 1e-5 for x in results
            )),
        }
        print(f"alpha={alpha:.2f}: January closed-loop cost={planned+emergency:.2f}", flush=True)
    selected = min(ALPHA_CANDIDATES, key=lambda a: detail[f"{a:.2f}"]["total_cost"])
    return float(selected), detail


def terminal_sensitivity(
    data: InputData,
    forecaster: CausalForecaster,
    daily_methods: list[str],
    residual_matrix: np.ndarray,
    alpha: float,
    alpha_detail: dict[str, dict[str, float]],
    *,
    fast: bool = False,
) -> dict:
    """用1月闭环回放检查MPC末端SOC影子价值和36 h末端约束。"""
    factors = (0.30, DEFAULT_TERMINAL_SHADOW_FACTOR, 0.60)
    factor_detail: dict[str, dict[str, float]] = {}
    for factor in factors:
        if abs(factor - DEFAULT_TERMINAL_SHADOW_FACTOR) < 1e-12:
            base = alpha_detail[f"{alpha:.2f}"]
            factor_detail[f"{factor:.2f}"] = {
                "total_cost": float(base["total_cost"]),
                "emergency_energy": float(base["emergency_energy"]),
                "min_soc_kwh": float(base["min_soc_kwh"]),
                "max_soc_kwh": float(base["max_soc_kwh"]),
                "mean_plan_terminal_soc_kwh": float(base["mean_plan_terminal_soc_kwh"]),
                "days_plan_terminal_floor_binding": int(
                    base["days_plan_terminal_floor_binding"]
                ),
            }
            continue
        results, _ = simulate_range(
            data, forecaster, daily_methods, residual_matrix, alpha,
            0, min(31, len(data.dates)), E_INITIAL,
            use_milp=not fast, mpc_horizon=6 if fast else MPC_HORIZON,
            terminal_shadow_factor=factor,
        )
        states = np.concatenate([x.state for x in results])
        factor_detail[f"{factor:.2f}"] = {
            "total_cost": float(sum(x.planned_cost + x.emergency_cost for x in results)),
            "emergency_energy": float(sum(x.emergency.sum() for x in results)),
            "min_soc_kwh": float(states.min()),
            "max_soc_kwh": float(states.max()),
            "mean_plan_terminal_soc_kwh": float(np.mean([
                x.plan_terminal_state for x in results
            ])),
            "days_plan_terminal_floor_binding": int(sum(
                abs(x.plan_terminal_state - x.state[0]) <= 1e-5 for x in results
            )),
        }

    without_floor, _ = simulate_range(
        data, forecaster, daily_methods, residual_matrix, alpha,
        0, min(31, len(data.dates)), E_INITIAL,
        use_milp=not fast, mpc_horizon=6 if fast else MPC_HORIZON,
        terminal_shadow_factor=DEFAULT_TERMINAL_SHADOW_FACTOR,
        enforce_terminal_floor=False,
    )
    states_no_floor = np.concatenate([x.state for x in without_floor])
    no_floor = {
        "total_cost": float(sum(x.planned_cost + x.emergency_cost for x in without_floor)),
        "emergency_energy": float(sum(x.emergency.sum() for x in without_floor)),
        "min_soc_kwh": float(states_no_floor.min()),
        "max_soc_kwh": float(states_no_floor.max()),
        "mean_plan_terminal_soc_kwh": float(np.mean([x.plan_terminal_state for x in without_floor])),
        "days_plan_terminal_at_min": int(sum(
            x.plan_terminal_state <= E_MIN + 1e-5 for x in without_floor
        )),
    }
    return {
        "terminal_shadow_factor": factor_detail,
        "without_36h_terminal_floor": no_floor,
        "kept_terminal_shadow_factor": DEFAULT_TERMINAL_SHADOW_FACTOR,
        "kept_36h_terminal_floor": True,
    }


def interval_labels_from_template(template_path: Path) -> list[str]:
    df = pd.read_excel(template_path, sheet_name="计划购电量", header=None, nrows=1)
    labels = [str(x) for x in df.iloc[0, 1:1+SLOTS_PER_DAY].tolist()]
    if len(labels) != SLOTS_PER_DAY:
        raise ValueError("result2模板中的时段列数不是144。")
    return labels


def group_emergency(interval_labels: list[str], values: np.ndarray, tol: float = 1e-6) -> list[dict]:
    events: list[dict] = []
    active = np.flatnonzero(values > tol)
    if len(active) == 0:
        return events
    starts = [int(active[0])]
    ends: list[int] = []
    for prev, cur in zip(active[:-1], active[1:]):
        if int(cur) != int(prev) + 1:
            ends.append(int(prev))
            starts.append(int(cur))
    ends.append(int(active[-1]))
    def normalize_clock(token: str) -> str:
        match = re.fullmatch(r"\s*(\d{1,2}):(\d{1,2})(\+1)?\s*", token)
        if not match:
            return token.strip()
        return f"{int(match.group(1))}:{int(match.group(2)):02d}{match.group(3) or ''}"
    for a, b in zip(starts, ends):
        left = normalize_clock(interval_labels[a].split("-", 1)[0])
        right = normalize_clock(interval_labels[b].split("-", 1)[1])
        events.append({"period": f"{left}-{right}", "energy": float(values[a:b+1].sum()),
                       "start_slot": a, "end_slot": b})
    return events


def validate_results(data: InputData, days: Iterable[int], results: list[DayResult]) -> dict[str, float]:
    max_balance = 0.0
    max_soc = 0.0
    max_simultaneous = 0.0
    max_cross_day = 0.0
    e_min_seen = math.inf
    e_max_seen = -math.inf
    min_variable = math.inf
    max_charge = 0.0
    max_discharge = 0.0
    max_emergency_charge_product = 0.0
    terminal_slacks: list[float] = []
    prev_end = None
    for day, r in zip(days, results):
        balance = r.grid + r.emergency + data.pv[day] + r.discharge \
                  - data.load[day] - r.charge - r.spill
        soc = r.state[1:] - r.state[:-1] - ETA_C*r.charge + r.discharge/ETA_D
        max_balance = max(max_balance, float(np.max(np.abs(balance))))
        max_soc = max(max_soc, float(np.max(np.abs(soc))))
        max_simultaneous = max(max_simultaneous, float(np.max(r.charge*r.discharge)))
        max_emergency_charge_product = max(
            max_emergency_charge_product, float(np.max(r.emergency * r.charge))
        )
        min_variable = min(
            min_variable,
            float(r.grid.min()), float(r.charge.min()), float(r.discharge.min()),
            float(r.emergency.min()), float(r.spill.min()),
        )
        max_charge = max(max_charge, float(r.charge.max()))
        max_discharge = max(max_discharge, float(r.discharge.max()))
        e_min_seen = min(e_min_seen, float(r.state.min()))
        e_max_seen = max(e_max_seen, float(r.state.max()))
        terminal_slacks.append(float(r.plan_terminal_state - r.state[0]))
        if prev_end is not None:
            max_cross_day = max(max_cross_day, abs(float(r.state[0]) - prev_end))
        prev_end = float(r.state[-1])
    return {
        "max_balance_residual_kwh": max_balance,
        "max_soc_residual_kwh": max_soc,
        "max_charge_discharge_product": max_simultaneous,
        "max_emergency_charge_product": max_emergency_charge_product,
        "max_cross_day_soc_gap_kwh": max_cross_day,
        "min_soc_kwh": e_min_seen,
        "max_soc_kwh": e_max_seen,
        "minimum_nonnegative_variable_kwh": min_variable,
        "max_charge_kwh_per_slot": max_charge,
        "max_discharge_kwh_per_slot": max_discharge,
        "minimum_36h_terminal_slack_kwh": float(min(terminal_slacks)),
        "days_36h_terminal_floor_binding": int(sum(abs(x) <= 1e-5 for x in terminal_slacks)),
    }


def serialize_output(
    data: InputData,
    forecaster: CausalForecaster,
    output_days: list[int],
    results: list[DayResult],
    interval_labels: list[str],
    daily_methods: list[str],
    daily_selection_records: list[dict],
    selected_predictions: np.ndarray,
    residual_matrix: np.ndarray,
    selected_alpha: float,
    alpha_scores: dict,
    terminal_checks: dict,
    e_feb_start: float,
    runtime_seconds: float,
) -> dict:
    dates = [data.dates[d].strftime("%Y-%m-%d") for d in output_days]
    planned_cost = np.array([r.planned_cost for r in results])
    emergency_cost = np.array([r.emergency_cost for r in results])
    grid = np.vstack([r.grid for r in results])
    charge = np.vstack([r.charge for r in results])
    discharge = np.vstack([r.discharge for r in results])
    emergency = np.vstack([r.emergency for r in results])
    spill = np.vstack([r.spill for r in results])
    state = np.vstack([r.state for r in results])
    events = {date: group_emergency(interval_labels, row)
              for date, row in zip(dates, emergency)}

    summaries = []
    for i, date in enumerate(dates):
        summaries.append({
            "date": date,
            "grid_energy": float(grid[i].sum()),
            "planned_cost": float(planned_cost[i]),
            "emergency_energy": float(emergency[i].sum()),
            "emergency_cost": float(emergency_cost[i]),
            "total_cost": float(planned_cost[i] + emergency_cost[i]),
            "charge_energy": float(charge[i].sum()),
            "discharge_energy": float(discharge[i].sum()),
            "spill_energy": float(spill[i].sum()),
            "e_start": float(state[i, 0]),
            "e_end": float(state[i, -1]),
        })

    validation = validate_results(data, output_days, results)
    specified = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
    specified_detail = {}
    for date in specified:
        if date in dates:
            i = dates.index(date)
            specified_detail[date] = {
                "summary": summaries[i],
                "emergency_events": events[date],
                "four_hour_charge": [float(charge[i, k:k+24].sum()) for k in range(0, 144, 24)],
                "four_hour_discharge": [float(discharge[i, k:k+24].sum()) for k in range(0, 144, 24)],
                "specified_grid": {
                    interval_labels[j]: float(grid[i, j])
                    for j in (59, 71, 83, 95, 107, 119)
                },
            }

    point_predictions = selected_predictions[output_days]
    risk_predictions = []
    margins = []
    for d in output_days:
        center = forecaster.horizon_forecast(d, daily_methods[d])
        risk_net, _, _ = risk_adjustment(d, center, residual_matrix, selected_alpha)
        risk_predictions.append(risk_net[:SLOTS_PER_DAY])
        margins.append(risk_net[:SLOTS_PER_DAY] - center[:SLOTS_PER_DAY])
    risk_predictions = np.vstack(risk_predictions)
    margins = np.vstack(margins)
    actual_output = data.net[output_days]
    point_error = actual_output - point_predictions
    forecast_evaluation = {
        "point_mae_kwh_per_slot": float(np.mean(np.abs(point_error))),
        "point_rmse_kwh_per_slot": float(np.sqrt(np.mean(point_error ** 2))),
        "risk_forecast_empirical_coverage": float(np.mean(actual_output <= risk_predictions + 1e-9)),
        "mean_risk_margin_kwh_per_slot": float(np.mean(margins)),
    }

    method_counts_all = {method: int(daily_methods.count(method))
                         for method in FORECAST_CANDIDATES}
    output_method_list = [daily_methods[d] for d in output_days]
    method_counts_output = {method: int(output_method_list.count(method))
                            for method in FORECAST_CANDIDATES}
    switches_output = int(sum(
        output_method_list[i] != output_method_list[i - 1]
        for i in range(1, len(output_method_list))
    ))
    grouped_total = float(sum(
        event["energy"]
        for day_events in events.values()
        for event in day_events
    ))
    validation.update({
        "dates_count": int(len(dates)),
        "date_start_ok": bool(dates[0] == "2025-02-01"),
        "date_end_ok": bool(dates[-1] == "2025-12-31"),
        "slots_per_day": int(grid.shape[1]),
        "last_interval_has_day_offset": bool("+1" in interval_labels[-1]),
        "emergency_grouping_energy_gap_kwh": abs(grouped_total - float(emergency.sum())),
        "specified_dates_present": bool(all(x in specified_detail for x in specified)),
        "ordinary_plan_intraday_updates": 0,
    })

    return {
        "metadata": {
            "model": "risk-aware 36h day-ahead MILP + causal 10min MPC",
            "forecast_policy": "daily causal rolling selection over previous 28 complete days",
            "forecast_candidates": list(FORECAST_CANDIDATES),
            "forecast_selection_window_days": FORECAST_SELECTION_WINDOW,
            "risk_residual_window_days": RISK_RESIDUAL_WINDOW,
            "underpurchase_extra_multiplier": UNDERPURCHASE_EXTRA_MULTIPLIER,
            "selected_alpha": selected_alpha,
            "forecast_evaluation_feb_dec": forecast_evaluation,
            "alpha_calibration": alpha_scores,
            "terminal_sensitivity": terminal_checks,
            "daily_forecast_selection": daily_selection_records,
            "forecast_method_counts_all_year": method_counts_all,
            "forecast_method_counts_feb_dec": method_counts_output,
            "forecast_method_switches_feb_dec": switches_output,
            "feb_1_initial_soc_kwh": e_feb_start,
            "runtime_seconds": runtime_seconds,
            "parameters": {
                "eta_c": ETA_C, "eta_d": ETA_D, "e_min": E_MIN, "e_max": E_MAX,
                "q_max_kwh_per_slot": Q_MAX, "plan_horizon_slots": PLAN_HORIZON,
                "mpc_horizon_slots": MPC_HORIZON, "emergency_multiplier": EMERGENCY_MULTIPLIER,
                "terminal_shadow_factor": DEFAULT_TERMINAL_SHADOW_FACTOR,
            },
        },
        "dates": dates,
        "interval_labels": interval_labels,
        "plan_grid": grid.tolist(),
        "charge": charge.tolist(),
        "discharge": discharge.tolist(),
        "emergency": emergency.tolist(),
        "spill": spill.tolist(),
        "state": state.tolist(),
        "selected_forecast_method_by_day": output_method_list,
        "daily_summary": summaries,
        "emergency_events": events,
        "specified_dates": specified_detail,
        "overall": {
            "planned_grid_energy": float(grid.sum()),
            "planned_cost": float(planned_cost.sum()),
            "emergency_energy": float(emergency.sum()),
            "emergency_cost": float(emergency_cost.sum()),
            "total_cost": float((planned_cost + emergency_cost).sum()),
            "charge_energy": float(charge.sum()),
            "discharge_energy": float(discharge.sum()),
            "spill_energy": float(spill.sum()),
            "emergency_intervals": int(np.sum(emergency > 1e-6)),
            "days_with_emergency": int(np.sum(np.any(emergency > 1e-6, axis=1))),
        },
        "validation": validation,
    }


_EMBEDDED_RESULT2_EXPORTER = r'''
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

let artifactTool;
try {
  artifactTool = await import("@oai/artifact-tool");
} catch (error) {
  const runtimeModules = process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES;
  if (!runtimeModules) throw error;
  const modulePath = path.join(runtimeModules, "@oai/artifact-tool/dist/artifact_tool.mjs");
  artifactTool = await import(pathToFileURL(modulePath).href);
}
const { FileBlob, SpreadsheetFile } = artifactTool;
const [templatePath, jsonPath, outputPath] = process.argv.slice(2);
if (!templatePath || !jsonPath || !outputPath) throw new Error("缺少Excel导出参数");

const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(templatePath));
const payload = JSON.parse(await fs.readFile(jsonPath, "utf8"));
const plan = wb.worksheets.getItem("计划购电量");
const storage = wb.worksheets.getItem("充放电量");
const emergency = wb.worksheets.getItem("紧急购电量");
const dates = payload.dates.map((x) => new Date(`${x}T00:00:00Z`));
const n = dates.length;
if (n !== 334 || payload.plan_grid.some((row) => row.length !== 144)) {
  throw new Error("结果数据应为334天×144时段");
}

plan.getRange("A1:EQ1").values = [["日期\\时间", ...payload.interval_labels,
  "全天购电量", "全天购电费"]];
storage.getRange("A1:F1").values = [["日期", "时间段", "充电量", "放电量", "时刻", "储电量"]];
emergency.getRange("A1:C1").values = [["日期", "购电时间段", "购电量"]];

const planRows = payload.plan_grid.map((row, i) => [dates[i], ...row,
  payload.daily_summary[i].grid_energy, payload.daily_summary[i].planned_cost]);
plan.getRange(`A2:EQ${n + 1}`).values = planRows;

const periods = ["0:00-4:00", "4:00-8:00", "8:00-12:00",
                 "12:00-16:00", "16:00-20:00", "20:00-24:00"];
const storageRows = [];
for (let i = 0; i < n; i++) {
  for (let b = 0; b < 6; b++) {
    const a = b * 24;
    const z = a + 24;
    storageRows.push([
      b === 0 ? dates[i] : null,
      periods[b],
      payload.charge[i].slice(a, z).reduce((s, x) => s + x, 0),
      payload.discharge[i].slice(a, z).reduce((s, x) => s + x, 0),
      b === 0 ? "0:00" : (b === 1 ? "24:00" : null),
      b === 0 ? payload.state[i][0] : (b === 1 ? payload.state[i][144] : null),
    ]);
  }
}
storage.getRange(`A2:F${storageRows.length + 1}`).values = storageRows;

const emergencyRows = [];
for (let i = 0; i < n; i++) {
  const events = payload.emergency_events[payload.dates[i]] || [];
  if (events.length === 0) {
    emergencyRows.push([dates[i], "无", 0]);
  } else {
    events.forEach((event, j) => emergencyRows.push([
      j === 0 ? dates[i] : null, event.period, event.energy,
    ]));
  }
}
emergency.getRange(`A2:C${emergencyRows.length + 1}`).values = emergencyRows;

const fontName = "Noto Sans CJK SC";
const headerFill = "#E7E6E6";
const border = { preset: "all", style: "thin", color: "#7F7F7F" };
for (const [sheet, used] of [
  [plan, `A1:EQ${n + 1}`],
  [storage, `A1:F${storageRows.length + 1}`],
  [emergency, `A1:C${emergencyRows.length + 1}`],
]) {
  sheet.showGridLines = false;
  sheet.getRange(used).format.font = { name: fontName, size: 10, color: "#000000" };
  sheet.getRange(used).format.verticalAlignment = "center";
  sheet.getRange(used).format.borders = border;
  sheet.getRange(used).format.rowHeight = 20;
}
for (const [sheet, range, size] of [
  [plan, "A1:EQ1", 9], [storage, "A1:F1", 10], [emergency, "A1:C1", 10],
]) {
  sheet.getRange(range).format = { fill: headerFill,
    font: { name: fontName, size, bold: true, color: "#000000" },
    horizontalAlignment: "center", verticalAlignment: "center", wrapText: true,
    borders: border };
}
plan.getRange("A2:A335").setNumberFormat("yyyy-mm-dd");
plan.getRange("B2:EQ335").setNumberFormat("0.000");
plan.getRange("A1:A335").format.columnWidth = 13;
plan.getRange("B1:EO335").format.columnWidth = 13;
plan.getRange("EP1:EQ335").format.columnWidth = 15;
plan.freezePanes.freezeRows(1); plan.freezePanes.freezeColumns(1);

storage.getRange(`A2:A${storageRows.length + 1}`).setNumberFormat("yyyy-mm-dd");
storage.getRange(`C2:D${storageRows.length + 1}`).setNumberFormat("0.000");
storage.getRange(`F2:F${storageRows.length + 1}`).setNumberFormat("0.000");
storage.getRange(`A1:A${storageRows.length + 1}`).format.columnWidth = 14;
storage.getRange(`B1:B${storageRows.length + 1}`).format.columnWidth = 16;
storage.getRange(`C1:D${storageRows.length + 1}`).format.columnWidth = 14;
storage.getRange(`E1:E${storageRows.length + 1}`).format.columnWidth = 11;
storage.getRange(`F1:F${storageRows.length + 1}`).format.columnWidth = 14;
storage.freezePanes.freezeRows(1);

emergency.getRange(`A2:A${emergencyRows.length + 1}`).setNumberFormat("yyyy-mm-dd");
emergency.getRange(`C2:C${emergencyRows.length + 1}`).setNumberFormat("0.000");
emergency.getRange(`A2:C${emergencyRows.length + 1}`).format.horizontalAlignment = "center";
emergency.getRange(`A1:A${emergencyRows.length + 1}`).format.columnWidth = 15;
emergency.getRange(`B1:B${emergencyRows.length + 1}`).format.columnWidth = 24;
emergency.getRange(`C1:C${emergencyRows.length + 1}`).format.columnWidth = 17;
emergency.freezePanes.freezeRows(1);

wb.recalculate();
await fs.mkdir(path.dirname(path.resolve(outputPath)), { recursive: true });
const out = await SpreadsheetFile.exportXlsx(wb);
await out.save(outputPath);
'''


def export_result2_workbook(template: Path, json_path: Path, output_path: Path) -> None:
    """使q2.py单文件即可将求解结果写入官方result2模板。"""
    node = os.environ.get("CODEX_PRIMARY_RUNTIME_NODE", "node")
    with tempfile.TemporaryDirectory(prefix="q2_result2_export_") as temp_dir:
        exporter = Path(temp_dir) / "build_result2.mjs"
        exporter.write_text(_EMBEDDED_RESULT2_EXPORTER, encoding="utf-8")
        subprocess.run([
            node, str(exporter), str(template), str(json_path), str(output_path)
        ], check=True)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="附件2_处理后.xlsx")
    parser.add_argument("--price", type=Path, required=True, help="附件1.xlsx")
    parser.add_argument("--template", type=Path, required=True, help="官方result2.xlsx模板")
    parser.add_argument("--output-json", type=Path, default=here / "q2_results.json")
    parser.add_argument("--output-xlsx", type=Path, default=here / "result2_问题二.xlsx")
    parser.add_argument("--skip-xlsx", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="调试模式：日前层用LP、实时层缩短到1h；正式结果不要使用。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_clock = time.perf_counter()
    data = load_inputs(args.data, args.price)
    labels = interval_labels_from_template(args.template)
    forecaster = CausalForecaster(data)

    daily_methods, selection_records, selected_predictions, _ = build_daily_forecast_policy(
        data, forecaster
    )
    residual_matrix = data.net - selected_predictions
    method_counts = {method: daily_methods.count(method) for method in FORECAST_CANDIDATES}
    print(f"daily forecast policy counts: {method_counts}", flush=True)
    selected_alpha, alpha_scores = calibrate_alpha(
        data, forecaster, daily_methods, residual_matrix, fast=args.fast
    )
    print(f"selected alpha: {selected_alpha:.2f}", flush=True)

    terminal_checks = terminal_sensitivity(
        data, forecaster, daily_methods, residual_matrix,
        selected_alpha, alpha_scores, fast=args.fast,
    )
    print(f"terminal sensitivity: {json.dumps(terminal_checks, ensure_ascii=False)}", flush=True)

    # 1月真实预热在alpha*尚不可知时，使用经济理论起点0.80；
    # 每天预测器仍严格根据当日之前已发生数据选择。
    warmup, e_feb = simulate_range(
        data, forecaster, daily_methods, residual_matrix, DEFAULT_ALPHA,
        0, 31, E_INITIAL,
        use_milp=not args.fast, mpc_horizon=6 if args.fast else MPC_HORIZON,
        progress_label="January warm-up",
    )
    del warmup

    output_days = list(range(31, len(data.dates)))
    results, _ = simulate_range(
        data, forecaster, daily_methods, residual_matrix, selected_alpha,
        31, len(data.dates), e_feb,
        use_milp=not args.fast, mpc_horizon=6 if args.fast else MPC_HORIZON,
        progress_label="February-December",
    )
    runtime = time.perf_counter() - start_clock
    payload = serialize_output(
        data, forecaster, output_days, results, labels,
        daily_methods, selection_records, selected_predictions, residual_matrix,
        selected_alpha, alpha_scores, terminal_checks, e_feb, runtime,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"overall": payload["overall"], "validation": payload["validation"]},
                     ensure_ascii=False, indent=2), flush=True)

    if not args.skip_xlsx:
        export_result2_workbook(args.template, args.output_json, args.output_xlsx)
        print(f"saved: {args.output_xlsx}", flush=True)


if __name__ == "__main__":
    main()
