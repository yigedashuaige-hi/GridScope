#!/usr/bin/env python3
"""问题三：多时刻光伏预测驱动的合同滚动调整与 10 min 因果 MPC。

本程序把最新版 q2.py 作为唯一基线模块，继承其中的 28 天预测器选择、
56 天风险残差、风险分位参数、36 h 日前规划及 10 min 储能 MPC。附件3
在 0:00、6:00、12:00、18:00 发布的未来24 h光伏预测按目标时刻线性
插值；超过24 h的影子时域仍由问题二的严格因果历史预测补齐。

正式输出同时回放 M0、M1、M2、M3 四种信息策略，result3 写入 M3。
四种策略均从2025-01-01 00:00的6000 kWh独立开始因果预热，1月沿用
Q2的DEFAULT_ALPHA风险冷启动规则，2月起才使用Q2在1月校准的分位参数。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, lil_matrix


SLOTS_PER_DAY = 144
PLAN_HORIZON = 216
MPC_HORIZON = 36
ISSUE_HOURS = (0, 6, 12, 18)
ISSUE_SLOTS = {0: 0, 6: 36, 12: 72, 18: 108}
POLICIES = {
    "M0": (),
    "M1": (6,),
    "M2": (6, 12),
    "M3": (6, 12, 18),
}
SELECTED_ALPHA = 0.70
ADJUSTMENT_MULTIPLIER = 0.50
CHANGE_TOL = 1.0e-5


@dataclass
class PVForecasts:
    dates: pd.DatetimeIndex
    values_kw: np.ndarray  # [day, issue(0/6/12/18), lead hour 1..24]


@dataclass
class ContractPlan:
    grid: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    emergency: np.ndarray
    spill: np.ndarray
    state: np.ndarray
    objective: float
    mip_gap: float


@dataclass
class Q3DayResult:
    g0: np.ndarray
    ga: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    emergency: np.ndarray
    spill: np.ndarray
    state: np.ndarray
    base_plan_cost: float
    adjusted_grid_cost: float
    emergency_cost: float
    node_records: list[dict[str, Any]]
    minimum_terminal_slack: float


Q2: Any = None
_POLICY_CONTEXT: dict[str, Any] | None = None


def load_q2_module(path: Path) -> Any:
    global Q2
    spec = importlib.util.spec_from_file_location("q2_baseline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载Q2基线: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    Q2 = module
    return module


def load_pv_forecasts(path: Path, dates: pd.DatetimeIndex) -> PVForecasts:
    df = pd.read_excel(path)
    if df.shape != (len(dates) * 4, 26):
        raise ValueError("附件3应为365天×4次发布，每次含24个小时预测。")
    issue_dates = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0].ffill())).normalize()
    values = np.full((len(dates), 4, 24), np.nan, dtype=float)
    date_to_idx = {date: i for i, date in enumerate(dates)}
    issue_to_idx = {hour: i for i, hour in enumerate(ISSUE_HOURS)}
    seen: set[tuple[int, int]] = set()
    for row, date in enumerate(issue_dates):
        if date not in date_to_idx:
            raise ValueError(f"附件3日期不在附件2中: {date:%Y-%m-%d}")
        token = str(df.iloc[row, 1]).strip()
        hour = int(token.split(":", 1)[0])
        if hour not in issue_to_idx:
            raise ValueError(f"附件3存在非0/6/12/18发布时间: {token}")
        key = (date_to_idx[date], issue_to_idx[hour])
        if key in seen:
            raise ValueError("附件3存在重复的日期与发布时间。")
        seen.add(key)
        values[key] = pd.to_numeric(df.iloc[row, 2:], errors="raise").to_numpy(float)
    if len(seen) != len(dates) * 4 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("附件3存在缺失、非数值或负光伏预测。")
    return PVForecasts(dates=dates, values_kw=values)


class CausalProfileForecaster:
    """按Q2同一候选方法预测任意非负日曲线，cutoff日及以后不可见。"""

    def __init__(self, values: np.ndarray, dates: pd.DatetimeIndex, reference: np.ndarray):
        self.values = np.asarray(values, dtype=float)
        self.dates = dates
        self.reference = np.asarray(reference, dtype=float)
        self.cache: dict[tuple[int, int, str], np.ndarray] = {}

    def _fallback(self, cutoff: int) -> np.ndarray:
        if cutoff <= 0:
            return self.reference.copy()
        recent = self.values[max(0, cutoff - 7):cutoff]
        profile = np.median(recent, axis=0)
        if cutoff < 3:
            w = cutoff / 3.0
            return w * profile + (1.0 - w) * self.reference
        return profile

    def forecast(self, target_day: int, cutoff: int, method: str) -> np.ndarray:
        key = (target_day, cutoff, method)
        if key in self.cache:
            return self.cache[key].copy()
        fallback = self._fallback(cutoff)
        target_date = self.dates[0] + pd.Timedelta(days=int(target_day))
        target_dow = target_date.dayofweek
        lw = target_day - 7
        if 0 <= lw < cutoff:
            last_week = self.values[lw]
        else:
            same = [i for i in range(cutoff - 1, -1, -1)
                    if self.dates[i].dayofweek == target_dow]
            last_week = self.values[same[0]] if same else fallback
        same = [i for i in range(cutoff)
                if self.dates[i].dayofweek == target_dow][-8:]
        weekday_median = np.median(self.values[same], axis=0) if same else fallback
        if method == "last_week":
            out = last_week
        elif method == "weekday_median":
            out = weekday_median
        elif method == "weighted_blend":
            recent = (np.median(self.values[max(0, cutoff - 7):cutoff], axis=0)
                      if cutoff else fallback)
            out = 0.55 * last_week + 0.30 * weekday_median + 0.15 * recent
        else:
            raise ValueError(f"未知预测方法: {method}")
        out = np.maximum(np.asarray(out, dtype=float), 0.0)
        self.cache[key] = out.copy()
        return out


def read_reference_pv(price_path: Path) -> np.ndarray:
    df = pd.read_excel(price_path)
    return pd.to_numeric(df.iloc[:, 3], errors="raise").to_numpy(float) / 6.0


def issue_profile_kwh(
    pv_forecasts: PVForecasts,
    data: Any,
    day: int,
    issue_hour: int,
) -> np.ndarray:
    issue_idx = ISSUE_HOURS.index(issue_hour)
    hourly = pv_forecasts.values_kw[day, issue_idx]
    if issue_hour == 0:
        anchor_kw = float(data.pv[day - 1, -1] * 6.0) if day > 0 else 0.0
    else:
        anchor_kw = float(data.pv[day, ISSUE_SLOTS[issue_hour] - 1] * 6.0)
    anchors_x = np.arange(25, dtype=float)
    anchors_y = np.concatenate([[anchor_kw], hourly])
    target_x = np.arange(1, SLOTS_PER_DAY + 1, dtype=float) / 6.0
    return np.maximum(np.interp(target_x, anchors_x, anchors_y), 0.0) / 6.0


def historical_horizon(
    day: int,
    start_slot: int,
    horizon: int,
    method: str,
    net_forecaster: Any,
    pv_forecaster: CausalProfileForecaster,
) -> tuple[np.ndarray, np.ndarray]:
    net = np.empty(horizon)
    pv = np.empty(horizon)
    profile_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(horizon):
        absolute = day * SLOTS_PER_DAY + start_slot + k
        target_day, slot = divmod(absolute, SLOTS_PER_DAY)
        if target_day not in profile_cache:
            profile_cache[target_day] = (
                net_forecaster.forecast(target_day, day, method),
                pv_forecaster.forecast(target_day, day, method),
            )
        net[k] = profile_cache[target_day][0][slot]
        pv[k] = profile_cache[target_day][1][slot]
    return net, pv


def build_center_horizon(
    data: Any,
    net_forecaster: Any,
    pv_forecaster: CausalProfileForecaster,
    pv_forecasts: PVForecasts,
    day: int,
    issue_hour: int | None,
    start_slot: int,
    horizon: int,
    method: str,
) -> np.ndarray:
    hist_net, hist_pv = historical_horizon(
        day, start_slot, horizon, method, net_forecaster, pv_forecaster
    )
    if issue_hour is None:
        return hist_net
    official = issue_profile_kwh(pv_forecasts, data, day, issue_hour)
    issue_start = ISSUE_SLOTS[issue_hour]
    center = hist_net.copy()
    for k in range(horizon):
        global_slot = start_slot + k
        offset = global_slot - issue_start
        if 0 <= offset < SLOTS_PER_DAY:
            center[k] = hist_net[k] + hist_pv[k] - official[offset]
    return center


def risk_adjustment_at(
    day: int,
    start_slot: int,
    center: np.ndarray,
    residual_matrix: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist = residual_matrix[max(0, day - 56):day]
    if len(hist) < 4:
        slot_sigma = 0.12 * np.abs(center) + 25.0
        q = np.full(len(center), 0.8416212335729143)
    else:
        median = np.median(hist, axis=0)
        mad = 1.4826 * np.median(np.abs(hist - median), axis=0)
        std = np.std(hist, axis=0, ddof=1)
        sigma_day = np.maximum(np.maximum(mad, 0.65 * std), 15.0)
        standardized = hist / sigma_day[None, :]
        q_blocks = np.array([
            np.quantile(standardized[:, b * 36:(b + 1) * 36], alpha)
            for b in range(4)
        ], dtype=float)
        slots = (start_slot + np.arange(len(center))) % SLOTS_PER_DAY
        slot_sigma = sigma_day[slots]
        q = q_blocks[slots // 36]
    margin = np.minimum(3.0 * slot_sigma, np.maximum(0.0, q * slot_sigma))
    return center + margin, slot_sigma, q


def solve_adjusted_contract(
    risk_net: np.ndarray,
    price: np.ndarray,
    e_start: float,
    base_contract: np.ndarray,
    formal_length: int,
    *,
    use_milp: bool,
) -> ContractPlan:
    """调整节点优化；H使保持原合同始终是可行候选，调整费相对G0计。"""
    h = len(risk_net)
    if len(price) != h or len(base_contract) != formal_length or formal_length > h:
        raise ValueError("调整优化的时域长度不一致。")
    g0, c0, d0, h0, w0 = 0, h, 2 * h, 3 * h, 4 * h
    e0, z0, up0, um0 = 5 * h, 6 * h + 1, 7 * h + 1, 8 * h + 1
    nvar = 9 * h + 1
    obj = np.zeros(nvar)
    obj[g0:g0 + h] = price
    obj[c0:c0 + h] = Q2.EPS_THROUGHPUT
    obj[d0:d0 + h] = Q2.EPS_THROUGHPUT
    obj[h0:h0 + h] = Q2.EMERGENCY_MULTIPLIER * price
    obj[up0:up0 + formal_length] = ADJUSTMENT_MULTIPLIER * price[:formal_length]
    obj[um0:um0 + formal_length] = ADJUSTMENT_MULTIPLIER * price[:formal_length]

    lower = np.zeros(nvar)
    upper = np.full(nvar, np.inf)
    upper[c0:c0 + h] = Q2.Q_MAX
    upper[d0:d0 + h] = Q2.Q_MAX
    lower[e0:e0 + h + 1] = Q2.E_MIN
    upper[e0:e0 + h + 1] = Q2.E_MAX
    lower[e0] = upper[e0] = float(e_start)
    upper[z0:z0 + h] = 1.0
    upper[up0 + formal_length:up0 + h] = 0.0
    upper[um0 + formal_length:um0 + h] = 0.0

    rows = 4 * h + formal_length + 1
    A = lil_matrix((rows, nvar), dtype=float)
    lb = np.full(rows, -np.inf)
    ub = np.full(rows, np.inf)
    for k in range(h):
        A[k, g0 + k] = 1.0
        A[k, c0 + k] = -1.0
        A[k, d0 + k] = 1.0
        A[k, h0 + k] = 1.0
        A[k, w0 + k] = -1.0
        lb[k] = ub[k] = risk_net[k]

        row = h + k
        A[row, c0 + k] = -Q2.ETA_C
        A[row, d0 + k] = 1.0 / Q2.ETA_D
        A[row, e0 + k] = -1.0
        A[row, e0 + k + 1] = 1.0
        lb[row] = ub[row] = 0.0

        row = 2 * h + k
        A[row, c0 + k] = 1.0
        A[row, z0 + k] = -Q2.Q_MAX
        ub[row] = 0.0
        row = 3 * h + k
        A[row, d0 + k] = 1.0
        A[row, z0 + k] = Q2.Q_MAX
        ub[row] = Q2.Q_MAX

    for k in range(formal_length):
        row = 4 * h + k
        A[row, g0 + k] = 1.0
        A[row, up0 + k] = -1.0
        A[row, um0 + k] = 1.0
        lb[row] = ub[row] = base_contract[k]

    A[rows - 1, e0 + h] = 1.0
    lb[rows - 1] = float(e_start)
    integrality = np.zeros(nvar, dtype=np.int8)
    if use_milp:
        integrality[z0:z0 + h] = 1
    res = milp(
        c=obj,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(csc_matrix(A), lb, ub),
        options={"mip_rel_gap": 1e-8, "time_limit": 30.0, "presolve": True},
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"合同调整优化失败: {res.message}")
    x = res.x
    return ContractPlan(
        grid=x[g0:g0 + h].copy(),
        charge=x[c0:c0 + h].copy(),
        discharge=x[d0:d0 + h].copy(),
        emergency=x[h0:h0 + h].copy(),
        spill=x[w0:w0 + h].copy(),
        state=x[e0:e0 + h + 1].copy(),
        objective=float(res.fun),
        mip_gap=float(getattr(res, "mip_gap", 0.0) or 0.0),
    )


def price_horizon(price_day: np.ndarray, start_slot: int, horizon: int) -> np.ndarray:
    return np.asarray([price_day[(start_slot + k) % SLOTS_PER_DAY]
                       for k in range(horizon)], dtype=float)


def simulate_day(
    data: Any,
    net_forecaster: Any,
    pv_forecaster: CausalProfileForecaster,
    pv_forecasts: PVForecasts,
    residual_matrix: np.ndarray,
    daily_methods: list[str],
    day: int,
    e_start: float,
    policy: str,
    alpha: float,
    *,
    use_milp: bool,
    mpc_horizon: int,
) -> Q3DayResult:
    method = daily_methods[day]
    center = build_center_horizon(
        data, net_forecaster, pv_forecaster, pv_forecasts,
        day, 0, 0, PLAN_HORIZON, method,
    )
    risk_net, _, _ = risk_adjustment_at(day, 0, center, residual_matrix, alpha)
    prices = price_horizon(data.price, 0, PLAN_HORIZON)
    plan0 = Q2.solve_day_ahead(risk_net, prices, e_start, use_milp=use_milp)
    g0 = plan0.grid[:SLOTS_PER_DAY].copy()
    ga = g0.copy()

    ref_grid = np.full(SLOTS_PER_DAY + PLAN_HORIZON, np.nan)
    ref_state = np.full(SLOTS_PER_DAY + PLAN_HORIZON + 1, np.nan)
    ref_grid[:PLAN_HORIZON] = plan0.grid
    ref_state[:PLAN_HORIZON + 1] = plan0.state
    latest_issue_slot = 0
    latest_center = center

    C = np.zeros(SLOTS_PER_DAY)
    D = np.zeros(SLOTS_PER_DAY)
    H = np.zeros(SLOTS_PER_DAY)
    W = np.zeros(SLOTS_PER_DAY)
    E = np.zeros(SLOTS_PER_DAY + 1)
    E[0] = e_start
    observed_residuals: list[float] = []
    node_records: list[dict[str, Any]] = []
    terminal_slacks = [float(plan0.state[-1] - e_start)]
    allowed_updates = set(POLICIES[policy])

    for t in range(SLOTS_PER_DAY):
        issue_hour = t // 6
        if t in (36, 72, 108) and issue_hour in allowed_updates:
            latest_issue_slot = t
            latest_center = build_center_horizon(
                data, net_forecaster, pv_forecaster, pv_forecasts,
                day, issue_hour, t, PLAN_HORIZON, method,
            )
            risk_updated, _, _ = risk_adjustment_at(
                day, t, latest_center, residual_matrix, alpha
            )
            update_prices = price_horizon(data.price, t, PLAN_HORIZON)
            old_contract = ga[t:].copy()
            update = solve_adjusted_contract(
                risk_updated, update_prices, E[t], g0[t:], SLOTS_PER_DAY - t,
                use_milp=use_milp,
            )
            ga[t:] = update.grid[:SLOTS_PER_DAY - t]
            ref_grid[t:t + PLAN_HORIZON] = update.grid
            ref_state[t:t + PLAN_HORIZON + 1] = update.state
            terminal_slacks.append(float(update.state[-1] - E[t]))
            delta = ga[t:] - old_contract
            changed = np.abs(delta) > CHANGE_TOL
            deviation_g0 = np.abs(ga[t:] - g0[t:])
            node_records.append({
                "issue_hour": issue_hour,
                "eligible_slots": int(len(delta)),
                "changed_slots_vs_previous": int(changed.sum()),
                "adjustment_rate_vs_previous": float(changed.mean()),
                "mean_abs_adjustment_changed_kwh": (
                    float(np.mean(np.abs(delta[changed]))) if changed.any() else 0.0
                ),
                "mean_abs_adjustment_all_eligible_kwh": float(np.mean(np.abs(delta))),
                "slots_deviating_from_g0": int(np.sum(deviation_g0 > CHANGE_TOL)),
                "mean_abs_deviation_from_g0_kwh": float(np.mean(deviation_g0)),
                "mip_gap": update.mip_gap,
            })
            observed_residuals = []

        h = min(mpc_horizon, SLOTS_PER_DAY + PLAN_HORIZON - t)
        offset = t - latest_issue_slot
        center_h = latest_center[offset:offset + h].copy()
        if len(center_h) != h:
            raise RuntimeError("MPC预测时域没有被最近一次36 h影子规划覆盖。")
        observed_residuals.append(float(data.net[day, t] - center_h[0]))
        recent = np.asarray(observed_residuals[-6:])
        weights = 0.70 ** np.arange(len(recent) - 1, -1, -1)
        bias = float(np.dot(weights, recent) / weights.sum())
        online_net = center_h + bias * np.exp(-np.arange(h) / 18.0)
        online_net[0] = data.net[day, t]
        if not np.isfinite(ref_grid[t:t + h]).all() or not np.isfinite(ref_state[t + h]):
            raise RuntimeError("MPC引用了未生成的影子合同或SOC轨迹。")
        c, d, emergency, spill, e_next = Q2.solve_mpc_step(
            online_net,
            ref_grid[t:t + h],
            price_horizon(data.price, t, h),
            E[t],
            ref_state[t + h],
        )
        C[t], D[t], H[t], W[t], E[t + 1] = c, d, emergency, spill, e_next

    base_cost = float(np.dot(data.price, g0))
    adjusted_cost = float(np.dot(data.price, ga)
                          + ADJUSTMENT_MULTIPLIER * np.dot(data.price, np.abs(ga - g0)))
    emergency_cost = float(np.dot(Q2.EMERGENCY_MULTIPLIER * data.price, H))
    return Q3DayResult(
        g0=g0,
        ga=ga,
        charge=C,
        discharge=D,
        emergency=H,
        spill=W,
        state=E,
        base_plan_cost=base_cost,
        adjusted_grid_cost=adjusted_cost,
        emergency_cost=emergency_cost,
        node_records=node_records,
        minimum_terminal_slack=float(min(terminal_slacks)),
    )


def simulate_policy(policy: str) -> list[Q3DayResult]:
    if _POLICY_CONTEXT is None:
        raise RuntimeError("策略上下文尚未初始化。")
    ctx = _POLICY_CONTEXT
    e = float(Q2.E_INITIAL)
    out: list[Q3DayResult] = []
    all_days = range(len(ctx["data"].dates))
    for pos, day in enumerate(all_days):
        alpha = Q2.DEFAULT_ALPHA if day < 31 else SELECTED_ALPHA
        result = simulate_day(
            ctx["data"], ctx["net_forecaster"], ctx["pv_forecaster"],
            ctx["pv_forecasts"], ctx["residual_matrix"], ctx["daily_methods"],
            day, e, policy, alpha, use_milp=ctx["use_milp"],
            mpc_horizon=ctx["mpc_horizon"],
        )
        out.append(result)
        e = float(result.state[-1])
        if (pos + 1) % 31 == 0 or pos + 1 == len(ctx["data"].dates):
            print(f"Q3 {policy}: {pos + 1}/{len(ctx['data'].dates)} days", flush=True)
    return out


def validate_policy(data: Any, days: Iterable[int], results: list[Q3DayResult], policy: str) -> dict:
    max_balance = max_soc = max_simultaneous = max_cross = 0.0
    max_emergency_charge = 0.0
    min_soc, max_soc_seen = math.inf, -math.inf
    min_var = math.inf
    max_charge = max_discharge = 0.0
    prev_end: float | None = None
    preupdate_violation = 0.0
    for day, r in zip(days, results):
        balance = r.ga + r.emergency + data.pv[day] + r.discharge \
                  - data.load[day] - r.charge - r.spill
        soc = r.state[1:] - r.state[:-1] - Q2.ETA_C * r.charge + r.discharge / Q2.ETA_D
        max_balance = max(max_balance, float(np.max(np.abs(balance))))
        max_soc = max(max_soc, float(np.max(np.abs(soc))))
        max_simultaneous = max(max_simultaneous, float(np.max(r.charge * r.discharge)))
        max_emergency_charge = max(max_emergency_charge, float(np.max(r.emergency * r.charge)))
        min_soc = min(min_soc, float(r.state.min()))
        max_soc_seen = max(max_soc_seen, float(r.state.max()))
        min_var = min(min_var, float(r.g0.min()), float(r.ga.min()), float(r.charge.min()),
                      float(r.discharge.min()), float(r.emergency.min()), float(r.spill.min()))
        max_charge = max(max_charge, float(r.charge.max()))
        max_discharge = max(max_discharge, float(r.discharge.max()))
        first_update = min(POLICIES[policy]) * 6 if POLICIES[policy] else SLOTS_PER_DAY
        preupdate_violation = max(
            preupdate_violation,
            float(np.max(np.abs(r.ga[:first_update] - r.g0[:first_update])))
            if first_update else 0.0,
        )
        if prev_end is not None:
            max_cross = max(max_cross, abs(float(r.state[0]) - prev_end))
        prev_end = float(r.state[-1])
    return {
        "max_balance_residual_kwh": max_balance,
        "max_soc_residual_kwh": max_soc,
        "max_charge_discharge_product": max_simultaneous,
        "max_emergency_charge_product": max_emergency_charge,
        "max_cross_day_soc_gap_kwh": max_cross,
        "min_soc_kwh": min_soc,
        "max_soc_kwh": max_soc_seen,
        "minimum_nonnegative_variable_kwh": min_var,
        "max_charge_kwh_per_slot": max_charge,
        "max_discharge_kwh_per_slot": max_discharge,
        "max_contract_change_before_first_allowed_node_kwh": preupdate_violation,
        "minimum_36h_terminal_slack_kwh": float(min(r.minimum_terminal_slack for r in results)),
    }


def evaluate_pv_bias_candidate(pv_forecasts: PVForecasts, data: Any) -> dict:
    raw_abs: list[float] = []
    corrected_abs: list[float] = []
    issue_detail: dict[str, dict[str, float]] = {}
    for issue_hour in ISSUE_HOURS:
        raw_issue: list[float] = []
        corrected_issue: list[float] = []
        issue_idx = ISSUE_HOURS.index(issue_hour)
        for day in range(31, len(data.dates)):
            actual = np.full(24, np.nan)
            for lead in range(1, 25):
                total_min = issue_hour * 60 + lead * 60
                target_day = day + total_min // 1440
                minute = total_min % 1440
                if minute == 0:
                    target_day -= 1
                    slot = 143
                else:
                    slot = minute // 10 - 1
                if 0 <= target_day < len(data.dates):
                    actual[lead - 1] = data.pv[target_day, slot] * 6.0
            raw_fc = pv_forecasts.values_kw[day, issue_idx]
            hist_errors = []
            for past in range(max(0, day - 56), day):
                past_actual = np.full(24, np.nan)
                for lead in range(1, 25):
                    total_min = issue_hour * 60 + lead * 60
                    target_day = past + total_min // 1440
                    minute = total_min % 1440
                    if minute == 0:
                        target_day -= 1
                        slot = 143
                    else:
                        slot = minute // 10 - 1
                    if 0 <= target_day < len(data.dates):
                        past_actual[lead - 1] = data.pv[target_day, slot] * 6.0
                hist_errors.append(past_actual - pv_forecasts.values_kw[past, issue_idx])
            bias = np.nanmedian(np.vstack(hist_errors), axis=0)
            mask = np.isfinite(actual)
            raw_err = np.abs(actual[mask] - raw_fc[mask])
            corrected_err = np.abs(actual[mask] - (raw_fc + bias)[mask])
            raw_issue.extend(raw_err.tolist())
            corrected_issue.extend(corrected_err.tolist())
        raw_abs.extend(raw_issue)
        corrected_abs.extend(corrected_issue)
        issue_detail[str(issue_hour)] = {
            "raw_mae_kw": float(np.mean(raw_issue)),
            "rolling_bias_corrected_mae_kw": float(np.mean(corrected_issue)),
        }
    raw_mae = float(np.mean(raw_abs))
    corrected_mae = float(np.mean(corrected_abs))
    return {
        "validation_period": "2025-02-01/2025-12-31 walk-forward",
        "raw_mae_kw": raw_mae,
        "rolling_56d_issue_lead_bias_corrected_mae_kw": corrected_mae,
        "relative_improvement": float(1.0 - corrected_mae / raw_mae),
        "calibration_adopted": bool(corrected_mae < raw_mae),
        "formal_forecast": "raw official Attachment 3 forecast",
        "issue_detail": issue_detail,
    }


def policy_summary(results: list[Q3DayResult]) -> dict[str, float]:
    base = sum(r.base_plan_cost for r in results)
    ordinary = sum(r.adjusted_grid_cost for r in results)
    emergency = sum(r.emergency_cost for r in results)
    return {
        "base_plan_cost": float(base),
        "adjusted_ordinary_grid_cost": float(ordinary),
        "emergency_cost": float(emergency),
        "total_closed_loop_cost": float(ordinary + emergency),
        "base_grid_energy": float(sum(r.g0.sum() for r in results)),
        "active_grid_energy": float(sum(r.ga.sum() for r in results)),
        "emergency_energy": float(sum(r.emergency.sum() for r in results)),
        "adjustment_penalty": float(sum(
            r.adjusted_grid_cost - np.dot(_POLICY_CONTEXT["data"].price, r.ga)
            for r in results
        )),
    }


def aggregate_node_metrics(results: list[Q3DayResult]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for hour in (6, 12, 18):
        records = [x for r in results for x in r.node_records if x["issue_hour"] == hour]
        eligible = sum(x["eligible_slots"] for x in records)
        changed = sum(x["changed_slots_vs_previous"] for x in records)
        weighted_amount = sum(
            x["mean_abs_adjustment_changed_kwh"] * x["changed_slots_vs_previous"]
            for x in records
        )
        out[str(hour)] = {
            "days_evaluated": int(len(records)),
            "eligible_slots": int(eligible),
            "changed_slots": int(changed),
            "contract_adjustment_rate": float(changed / eligible) if eligible else 0.0,
            "mean_abs_adjustment_changed_kwh": float(weighted_amount / changed) if changed else 0.0,
            "mean_abs_adjustment_all_eligible_kwh": float(
                sum(x["mean_abs_adjustment_all_eligible_kwh"] * x["eligible_slots"]
                    for x in records) / eligible
            ) if eligible else 0.0,
        }
    return out


def serialize_output(
    data: Any,
    days: list[int],
    labels: list[str],
    formal: list[Q3DayResult],
    all_results: dict[str, list[Q3DayResult]],
    full_results: dict[str, list[Q3DayResult]],
    pv_evaluation: dict,
    warmup_end_soc: dict[str, float],
    runtime: float,
) -> dict:
    dates = [data.dates[d].strftime("%Y-%m-%d") for d in days]
    g0 = np.vstack([r.g0 for r in formal])
    ga = np.vstack([r.ga for r in formal])
    charge = np.vstack([r.charge for r in formal])
    discharge = np.vstack([r.discharge for r in formal])
    emergency = np.vstack([r.emergency for r in formal])
    spill = np.vstack([r.spill for r in formal])
    state = np.vstack([r.state for r in formal])
    events = {date: Q2.group_emergency(labels, row)
              for date, row in zip(dates, emergency)}
    daily = []
    for i, r in enumerate(formal):
        daily.append({
            "date": dates[i],
            "base_grid_energy": float(r.g0.sum()),
            "base_plan_cost": r.base_plan_cost,
            "active_grid_energy": float(r.ga.sum()),
            "adjusted_grid_cost": r.adjusted_grid_cost,
            "adjustment_penalty": float(
                r.adjusted_grid_cost - np.dot(data.price, r.ga)
            ),
            "emergency_energy": float(r.emergency.sum()),
            "emergency_cost": r.emergency_cost,
            "total_cost": float(r.adjusted_grid_cost + r.emergency_cost),
            "charge_energy": float(r.charge.sum()),
            "discharge_energy": float(r.discharge.sum()),
            "spill_energy": float(r.spill.sum()),
            "e_start": float(r.state[0]),
            "e_end": float(r.state[-1]),
        })
    summaries = {name: policy_summary(results) for name, results in all_results.items()}
    costs = {name: summaries[name]["total_closed_loop_cost"] for name in POLICIES}
    voi = {
        "V6": float(costs["M0"] - costs["M1"]),
        "V12": float(costs["M1"] - costs["M2"]),
        "V18": float(costs["M2"] - costs["M3"]),
        "total_M0_to_M3": float(costs["M0"] - costs["M3"]),
    }
    grouped = sum(event["energy"] for x in events.values() for event in x)
    validation = {name: validate_policy(data, days, res, name)
                  for name, res in all_results.items()}
    full_validation = {
        name: validate_policy(data, range(len(data.dates)), res, name)
        for name, res in full_results.items()
    }
    for name in POLICIES:
        validation[name]["january_to_february_soc_gap_kwh"] = abs(
            float(full_results[name][30].state[-1])
            - float(full_results[name][31].state[0])
        )
        validation[name]["full_year_max_cross_day_soc_gap_kwh"] = (
            full_validation[name]["max_cross_day_soc_gap_kwh"]
        )
    validation["M3"].update({
        "dates_count": len(dates),
        "date_start_ok": dates[0] == "2025-02-01",
        "date_end_ok": dates[-1] == "2025-12-31",
        "slots_per_day": int(g0.shape[1]),
        "last_interval_has_day_offset": "+1" in labels[-1],
        "emergency_grouping_energy_gap_kwh": abs(grouped - float(emergency.sum())),
        "past_contract_periods_modified": 0,
        "allowed_adjustment_nodes": [6, 12, 18],
    })
    return {
        "metadata": {
            "model": "Q2 baseline + causal 0/6/12/18 PV forecast updates + 10min MPC",
            "q2_baseline_alpha": SELECTED_ALPHA,
            "initial_soc_2025_01_01_kwh": Q2.E_INITIAL,
            "january_warmup_alpha": Q2.DEFAULT_ALPHA,
            "feb_dec_alpha": SELECTED_ALPHA,
            "january_warmup_end_soc_kwh_by_policy": warmup_end_soc,
            "statistics_period": "2025-02-01/2025-12-31",
            "pv_interpolation": "issue-time observed PV anchor plus linear interpolation to 10min endpoints",
            "beyond_24h_rule": "Q2 causal historical net/PV forecast; no future actual PV",
            "pv_bias_calibration": pv_evaluation,
            "adjustment_settlement": "p*GA + 0.5*p*abs(GA-G0)",
            "policy_definitions": {k: [0, *v] for k, v in POLICIES.items()},
            "policy_summary": summaries,
            "voi": voi,
            "M3_node_adjustment_metrics": aggregate_node_metrics(formal),
            "runtime_seconds": runtime,
            "parameters": {
                "eta_c": Q2.ETA_C, "eta_d": Q2.ETA_D,
                "e_min_kwh": Q2.E_MIN, "e_max_kwh": Q2.E_MAX,
                "q_max_kwh_per_slot": Q2.Q_MAX,
                "plan_horizon_slots": PLAN_HORIZON,
                "mpc_horizon_slots": MPC_HORIZON,
                "emergency_multiplier": Q2.EMERGENCY_MULTIPLIER,
                "adjustment_multiplier": ADJUSTMENT_MULTIPLIER,
            },
        },
        "dates": dates,
        "interval_labels": labels,
        "plan_grid": g0.tolist(),
        "active_grid": ga.tolist(),
        "charge": charge.tolist(),
        "discharge": discharge.tolist(),
        "emergency": emergency.tolist(),
        "spill": spill.tolist(),
        "state": state.tolist(),
        "daily_summary": daily,
        "emergency_events": events,
        "overall": summaries["M3"],
        "validation": validation,
    }


_EMBEDDED_RESULT3_EXPORTER = r'''
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
let artifactTool;
try { artifactTool = await import("@oai/artifact-tool"); }
catch (error) {
  const runtimeModules = process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES;
  if (!runtimeModules) throw error;
  artifactTool = await import(pathToFileURL(path.join(runtimeModules,
    "@oai/artifact-tool/dist/artifact_tool.mjs")).href);
}
const { FileBlob, SpreadsheetFile } = artifactTool;
const [templatePath, jsonPath, outputPath] = process.argv.slice(2);
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(templatePath));
const payload = JSON.parse(await fs.readFile(jsonPath, "utf8"));
const plan = wb.worksheets.getItem("计划购电量");
const adjusted = wb.worksheets.getItem("调整购电量");
const storage = wb.worksheets.getItem("充放电量");
const emergency = wb.worksheets.getItem("紧急购电量");
const dates = payload.dates.map((x) => new Date(`${x}T00:00:00Z`));
const n = dates.length;
if (n !== 334 || payload.plan_grid.some((r) => r.length !== 144) ||
    payload.active_grid.some((r) => r.length !== 144)) throw new Error("Q3结果尺寸错误");
const headers = [["日期\\时间", ...payload.interval_labels, "全天购电量", "全天购电费"]];
plan.getRange("A1:EQ1").values = headers;
adjusted.getRange("A1:EQ1").values = headers;
plan.getRange(`A2:EQ${n+1}`).values = payload.plan_grid.map((row, i) => [dates[i], ...row,
  payload.daily_summary[i].base_grid_energy, payload.daily_summary[i].base_plan_cost]);
adjusted.getRange(`A2:EQ${n+1}`).values = payload.active_grid.map((row, i) => [dates[i], ...row,
  payload.daily_summary[i].active_grid_energy, payload.daily_summary[i].adjusted_grid_cost]);
const periods = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00",
                 "16:00-20:00", "20:00-24:00"];
const storageRows = [];
for (let i=0; i<n; i++) for (let b=0; b<6; b++) {
  const a=b*24, z=a+24;
  storageRows.push([b===0?dates[i]:null, periods[b],
    payload.charge[i].slice(a,z).reduce((s,x)=>s+x,0),
    payload.discharge[i].slice(a,z).reduce((s,x)=>s+x,0),
    b===0?"0:00":(b===1?"24:00":null),
    b===0?payload.state[i][0]:(b===1?payload.state[i][144]:null)]);
}
storage.getRange("A1:F1").values = [["日期","时间段","充电量","放电量","时刻","储电量"]];
storage.getRange(`A2:F${storageRows.length+1}`).values = storageRows;
const emergencyRows=[];
for (let i=0; i<n; i++) {
  const events=payload.emergency_events[payload.dates[i]]||[];
  if (!events.length) emergencyRows.push([dates[i],"无",0]);
  else events.forEach((event,j)=>emergencyRows.push([j===0?dates[i]:null,event.period,event.energy]));
}
emergency.getRange("A1:C1").values = [["日期","购电时间段","购电量"]];
emergency.getRange(`A2:C${emergencyRows.length+1}`).values = emergencyRows;
const font="Noto Sans CJK SC";
const border={preset:"all",style:"thin",color:"#7F7F7F"};
for (const [sheet, used] of [[plan,`A1:EQ${n+1}`],[adjusted,`A1:EQ${n+1}`],
  [storage,`A1:F${storageRows.length+1}`],[emergency,`A1:C${emergencyRows.length+1}`]]) {
  sheet.showGridLines=false;
  sheet.getRange(used).format.font={name:font,size:10,color:"#000000"};
  sheet.getRange(used).format.verticalAlignment="center";
  sheet.getRange(used).format.borders=border;
  sheet.getRange(used).format.rowHeight=20;
}
for (const [sheet, range, size] of [[plan,"A1:EQ1",9],[adjusted,"A1:EQ1",9],
  [storage,"A1:F1",10],[emergency,"A1:C1",10]]) sheet.getRange(range).format={
    fill:"#E7E6E6",font:{name:font,size,bold:true,color:"#000000"},
    horizontalAlignment:"center",verticalAlignment:"center",wrapText:true,borders:border};
for (const sheet of [plan,adjusted]) {
  sheet.getRange("A2:A335").setNumberFormat("yyyy-mm-dd");
  sheet.getRange("B2:EQ335").setNumberFormat("0.000");
  sheet.getRange("A1:A335").format.columnWidth=13;
  sheet.getRange("B1:EO335").format.columnWidth=13;
  sheet.getRange("EP1:EQ335").format.columnWidth=15;
  sheet.freezePanes.freezeRows(1); sheet.freezePanes.freezeColumns(1);
}
storage.getRange(`A2:A${storageRows.length+1}`).setNumberFormat("yyyy-mm-dd");
storage.getRange(`C2:D${storageRows.length+1}`).setNumberFormat("0.000");
storage.getRange(`F2:F${storageRows.length+1}`).setNumberFormat("0.000");
storage.getRange(`A1:A${storageRows.length+1}`).format.columnWidth=14;
storage.getRange(`B1:B${storageRows.length+1}`).format.columnWidth=16;
storage.getRange(`C1:D${storageRows.length+1}`).format.columnWidth=14;
storage.getRange(`E1:E${storageRows.length+1}`).format.columnWidth=11;
storage.getRange(`F1:F${storageRows.length+1}`).format.columnWidth=14;
storage.freezePanes.freezeRows(1);
emergency.getRange(`A2:A${emergencyRows.length+1}`).setNumberFormat("yyyy-mm-dd");
emergency.getRange(`C2:C${emergencyRows.length+1}`).setNumberFormat("0.000");
emergency.getRange(`A2:C${emergencyRows.length+1}`).format.horizontalAlignment="center";
emergency.getRange(`A1:A${emergencyRows.length+1}`).format.columnWidth=15;
emergency.getRange(`B1:B${emergencyRows.length+1}`).format.columnWidth=24;
emergency.getRange(`C1:C${emergencyRows.length+1}`).format.columnWidth=17;
emergency.freezePanes.freezeRows(1);
wb.recalculate();
await fs.mkdir(path.dirname(path.resolve(outputPath)),{recursive:true});
await (await SpreadsheetFile.exportXlsx(wb)).save(outputPath);
'''


def export_result3(template: Path, payload_path: Path, output: Path) -> None:
    node = os.environ.get("CODEX_PRIMARY_RUNTIME_NODE", "node")
    with tempfile.TemporaryDirectory(prefix="q3_result3_export_") as temp_dir:
        script = Path(temp_dir) / "build_result3.mjs"
        script.write_text(_EMBEDDED_RESULT3_EXPORTER, encoding="utf-8")
        subprocess.run([node, str(script), str(template), str(payload_path), str(output)], check=True)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q2", type=Path, required=True, help="最新版q2.py")
    parser.add_argument("--data", type=Path, required=True, help="附件2_处理后.xlsx")
    parser.add_argument("--price", type=Path, required=True, help="附件1_处理后.xlsx")
    parser.add_argument("--pv-forecast", type=Path, required=True, help="附件3.xlsx")
    parser.add_argument("--template", type=Path, required=True, help="官方result3.xlsx")
    parser.add_argument("--output-json", type=Path, default=here / "q3_results.json")
    parser.add_argument("--output-xlsx", type=Path, default=here / "result3_问题三.xlsx")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--fast", action="store_true",
                        help="仅调试：日前层用LP、MPC缩为1h；正式结果不得使用。")
    parser.add_argument("--skip-xlsx", action="store_true")
    return parser.parse_args()


def main() -> None:
    global _POLICY_CONTEXT
    args = parse_args()
    start_clock = time.perf_counter()
    q2 = load_q2_module(args.q2)
    data = q2.load_inputs(args.data, args.price)
    pv_forecasts = load_pv_forecasts(args.pv_forecast, data.dates)
    reference_pv = read_reference_pv(args.price)
    net_forecaster = q2.CausalForecaster(data)
    daily_methods, _, selected_predictions, _ = q2.build_daily_forecast_policy(
        data, net_forecaster
    )
    residual_matrix = data.net - selected_predictions
    pv_forecaster = CausalProfileForecaster(data.pv, data.dates, reference_pv)
    labels = q2.interval_labels_from_template(args.template)
    output_days = list(range(31, len(data.dates)))
    pv_evaluation = evaluate_pv_bias_candidate(pv_forecasts, data)
    if pv_evaluation["calibration_adopted"]:
        raise RuntimeError("当前代码仅在偏差校准无改善时采用原始预测；本数据检验结果异常。")

    _POLICY_CONTEXT = {
        "data": data, "net_forecaster": net_forecaster,
        "pv_forecaster": pv_forecaster, "pv_forecasts": pv_forecasts,
        "residual_matrix": residual_matrix, "daily_methods": daily_methods,
        "use_milp": not args.fast,
        "mpc_horizon": 6 if args.fast else MPC_HORIZON,
    }
    if args.jobs > 1:
        try:
            context = mp.get_context("fork")
        except ValueError:
            context = mp.get_context()
        with context.Pool(processes=min(args.jobs, len(POLICIES))) as pool:
            values = pool.map(simulate_policy, POLICIES.keys())
        full_results = dict(zip(POLICIES.keys(), values))
    else:
        full_results = {name: simulate_policy(name) for name in POLICIES}
    all_results = {name: results[31:] for name, results in full_results.items()}
    warmup_end_soc = {
        name: float(results[30].state[-1]) for name, results in full_results.items()
    }
    runtime = time.perf_counter() - start_clock
    payload = serialize_output(
        data, output_days, labels, all_results["M3"], all_results,
        full_results, pv_evaluation, warmup_end_soc, runtime,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "policy_summary": payload["metadata"]["policy_summary"],
        "voi": payload["metadata"]["voi"],
        "node_metrics": payload["metadata"]["M3_node_adjustment_metrics"],
        "validation_M3": payload["validation"]["M3"],
    }, ensure_ascii=False, indent=2), flush=True)
    if not args.skip_xlsx:
        export_result3(args.template, args.output_json, args.output_xlsx)
        print(f"saved: {args.output_xlsx}", flush=True)


if __name__ == "__main__":
    main()
