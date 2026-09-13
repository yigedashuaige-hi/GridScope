#!/usr/bin/env python3
"""问题四：因果电价预测下的 Q4-2 与 Q4-3 闭环调度。

q4.py 与问题三代码分开。它把最新版 q2.py 作为物理与风险基线，并调用
q3.py 中已经核验的附件3时标映射及合同调整求解函数。电价分解为周期基线
m与短期残差x；AR(1)、AR(2)、ARX仅用1月严格walk-forward误差选型，
2月起固定模型族并每日用最近28天数据重估参数。周期基线只用预测发起日
之前已经发生的附件4价格；零历史时取0，样本不足时使用expanding past-only
回退。实际未来价格从不进入合同规划；新价格观测只更新残差状态和实时MPC，
Q4-3另在6/12/18允许改合同。两种策略均从1月1日6000 kWh独立因果预热。
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


SLOTS_PER_DAY = 144
PLAN_HORIZON = 216
MPC_HORIZON = 36
PRICE_TRAINING_DAYS = 28
PRICE_BASELINE_SAME_WEEKDAYS = 8
PRICE_MODELS = ("AR1", "AR2", "ARX")
STRATEGIES = ("Q4-2", "Q4-3")
SELECTED_ALPHA = 0.70
ADJUSTMENT_MULTIPLIER = 0.50


@dataclass
class Q4DayResult:
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
    node_records: list[dict]
    minimum_terminal_slack: float


Q2: Any = None
Q3: Any = None
_STRATEGY_CONTEXT: dict[str, Any] | None = None


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_actual_prices(path: Path, dates: pd.DatetimeIndex) -> np.ndarray:
    df = pd.read_excel(path)
    parsed = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0])).normalize()
    if not parsed.equals(dates) or df.shape != (len(dates), SLOTS_PER_DAY + 1):
        raise ValueError("附件4必须与附件2包含相同的365天×144时段。")
    values = df.iloc[:, 1:].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("附件4存在空值、非数值或负电价。")
    return values


class CausalPriceForecaster:
    """周期基线加短期残差模型；模型选型与参数估计均只看历史。"""

    def __init__(
        self,
        prices: np.ndarray,
        dates: pd.DatetimeIndex,
        net_residual: np.ndarray,
    ):
        self.prices = np.asarray(prices, dtype=float)
        self.dates = dates
        self.net_residual = np.asarray(net_residual, dtype=float)
        self.n_days = len(dates)
        self._baseline_cache: dict[tuple[int, int], np.ndarray] = {}
        self.baseline = np.vstack([
            self._baseline_profile(day, day) for day in range(self.n_days)
        ])
        self.residual = self.prices - self.baseline
        self.x_flat = self.residual.ravel()
        self.net_flat = self.net_residual.ravel()
        self.coefficients = {
            model: [self._fit(model, day) for day in range(self.n_days)]
            for model in PRICE_MODELS
        }
        self.january_scores = self._score_days(range(7, 31))
        self.selected_model = min(
            PRICE_MODELS,
            key=lambda m: (self.january_scores[m]["mae"],
                           self.january_scores[m]["rmse"], PRICE_MODELS.index(m)),
        )
        self.model_by_day: list[str] = []
        self.selection_records: list[dict[str, Any]] = []
        for day in range(self.n_days):
            if day < 7:
                model = "AR1"
                rule = "cold_start_AR1"
                scores: dict[str, dict[str, float]] = {}
                hist_start = 0
            elif day < 31:
                hist_start = max(1, day - 7)
                scores = self._score_days(range(hist_start, day))
                model = min(
                    PRICE_MODELS,
                    key=lambda m: (scores[m]["mae"], scores[m]["rmse"],
                                   PRICE_MODELS.index(m)),
                )
                rule = "january_past_only_rolling_selection"
            else:
                model = self.selected_model
                rule = "fixed_after_january_selection"
                scores = {}
                hist_start = 7
            self.model_by_day.append(model)
            self.selection_records.append({
                "date": self.dates[day].strftime("%Y-%m-%d"),
                "model": model,
                "rule": rule,
                "history_start": (
                    self.dates[hist_start].strftime("%Y-%m-%d")
                    if day >= 7 and day < 31 else None
                ),
                "history_end": (
                    self.dates[day - 1].strftime("%Y-%m-%d")
                    if day >= 7 and day < 31 else None
                ),
                "candidate_scores": scores,
            })
        self.holdout_scores = self._score_days(range(31, self.n_days))

    def _baseline_profile(self, target_day: int, cutoff_day: int) -> np.ndarray:
        """仅用日期小于cutoff_day的附件4价格预测target_day周期基线。"""
        key = (int(target_day), int(cutoff_day))
        if key in self._baseline_cache:
            return self._baseline_cache[key].copy()
        if cutoff_day <= 0:
            baseline = np.zeros(SLOTS_PER_DAY, dtype=float)
        else:
            target = self.dates[0] + pd.Timedelta(days=int(target_day))
            expanding = np.median(self.prices[:cutoff_day], axis=0)
            same = [
                i for i in range(cutoff_day)
                if self.dates[i].dayofweek == target.dayofweek
            ][-PRICE_BASELINE_SAME_WEEKDAYS:]
            if len(same) >= 2:
                baseline = np.median(self.prices[same], axis=0)
            elif len(same) == 1:
                baseline = 0.5 * self.prices[same[0]] + 0.5 * expanding
            else:
                baseline = expanding
        baseline = np.maximum(np.asarray(baseline, dtype=float), 0.0)
        self._baseline_cache[key] = baseline.copy()
        return baseline

    def _fit(self, model: str, day: int) -> np.ndarray:
        start = max(2, max(0, day - PRICE_TRAINING_DAYS) * SLOTS_PER_DAY)
        end = day * SLOTS_PER_DAY
        idx = np.arange(start, end, dtype=int)
        if len(idx) < 20:
            return np.array([0.0, 0.5]) if model == "AR1" else np.array([0.0, 0.5, 0.0])
        y = self.x_flat[idx]
        if model == "AR1":
            X = np.column_stack([np.ones(len(idx)), self.x_flat[idx - 1]])
        elif model == "AR2":
            X = np.column_stack([
                np.ones(len(idx)), self.x_flat[idx - 1], self.x_flat[idx - 2]
            ])
        elif model == "ARX":
            X = np.column_stack([
                np.ones(len(idx)), self.x_flat[idx - 1], self.net_flat[idx - 1] / 500.0
            ])
        else:
            raise ValueError(f"未知电价模型: {model}")
        ridge = 1.0e-8 * np.eye(X.shape[1])
        ridge[0, 0] = 0.0
        return np.linalg.solve(X.T @ X + ridge, X.T @ y)

    def _one_step_error(self, model: str, day: int) -> np.ndarray:
        coef = self.coefficients[model][day]
        idx = np.arange(day * SLOTS_PER_DAY, (day + 1) * SLOTS_PER_DAY)
        if model == "AR1":
            predicted = coef[0] + coef[1] * self.x_flat[idx - 1]
        elif model == "AR2":
            predicted = (coef[0] + coef[1] * self.x_flat[idx - 1]
                         + coef[2] * self.x_flat[idx - 2])
        else:
            predicted = (coef[0] + coef[1] * self.x_flat[idx - 1]
                         + coef[2] * self.net_flat[idx - 1] / 500.0)
        return self.x_flat[idx] - predicted

    def _score_days(self, days: Iterable[int]) -> dict[str, dict[str, float]]:
        scores = {}
        for model in PRICE_MODELS:
            errors = np.concatenate([self._one_step_error(model, day) for day in days])
            scores[model] = {
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "observations": int(len(errors)),
            }
        return scores

    def _states_at(self, absolute_slot: int) -> tuple[float, float]:
        if absolute_slot < 0:
            return 0.0, 0.0
        latest = self.x_flat[min(absolute_slot, len(self.x_flat) - 1)]
        previous = self.x_flat[max(0, min(absolute_slot - 1, len(self.x_flat) - 1))]
        return float(latest), float(previous)

    def forecast_before_slot(self, day: int, start_slot: int, horizon: int) -> np.ndarray:
        """在start_slot执行前预测；最后已知价格是start_slot-1。"""
        absolute_start = day * SLOTS_PER_DAY + start_slot
        latest, previous = self._states_at(absolute_start - 1)
        return self._recursive(day, absolute_start, horizon, latest, previous, None)

    def forecast_with_current(self, day: int, slot: int, horizon: int) -> np.ndarray:
        """当前时段价格已观测；第一个价格取真实值，其后递推。"""
        absolute = day * SLOTS_PER_DAY + slot
        current = float(self.residual[day, slot])
        previous, _ = self._states_at(absolute - 1)
        out = np.empty(horizon)
        out[0] = self.prices[day, slot]
        if horizon > 1:
            out[1:] = self._recursive(
                day, absolute + 1, horizon - 1, current, previous,
                float(self.net_residual[day, slot]),
            )
        return out

    def _recursive(
        self,
        day: int,
        absolute_start: int,
        horizon: int,
        latest: float,
        previous: float,
        latest_net_error: float | None,
    ) -> np.ndarray:
        model = self.model_by_day[day]
        coef = self.coefficients[model][day]
        out = np.empty(horizon)
        x1, x2 = latest, previous
        for k in range(horizon):
            if model == "AR1":
                xnew = coef[0] + coef[1] * x1
            elif model == "AR2":
                xnew = coef[0] + coef[1] * x1 + coef[2] * x2
            else:
                exog = (latest_net_error / 500.0) if (k == 0 and latest_net_error is not None) else 0.0
                xnew = coef[0] + coef[1] * x1 + coef[2] * exog
            target_abs = absolute_start + k
            target_day, target_slot = divmod(target_abs, SLOTS_PER_DAY)
            base = self._baseline_profile(target_day, day)[target_slot]
            out[k] = max(float(base + xnew), 0.0)
            x2, x1 = x1, float(xnew)
        return out

    def metadata(self) -> dict:
        return {
            "decomposition": "p[d,t] = m[d,t] + x[d,t]",
            "baseline": (
                "past-only median of up to 8 prior same-weekday observations; "
                "with fewer than 2 samples use expanding past-only fallback; "
                "zero when no prior Attachment 4 price exists"
            ),
            "baseline_source": "Attachment 4 observations strictly before forecast origin day",
            "attachment1_price_used_for_cold_start": False,
            "annual_or_future_price_used": False,
            "forecast_price_lower_bound": 0.0,
            "forecast_price_upper_bound": None,
            "parameter_window_days": PRICE_TRAINING_DAYS,
            "candidate_models": list(PRICE_MODELS),
            "selection_period": "2025-01-08/2025-01-31 strict walk-forward",
            "january_walk_forward": self.january_scores,
            "selected_model": self.selected_model,
            "formal_model_applied_from": "2025-02-01",
            "january_warmup_model_counts": {
                model: self.model_by_day[:31].count(model) for model in PRICE_MODELS
            },
            "january_warmup_model_path": self.selection_records[:31],
            "feb_dec_holdout_walk_forward": self.holdout_scores,
            "intraday_update": "latest observed residual initializes recursive forecast; coefficients fixed within day",
        }


def build_net_center(
    strategy: str,
    data: Any,
    net_forecaster: Any,
    pv_forecaster: Any,
    pv_forecasts: Any,
    day: int,
    issue_hour: int,
    start_slot: int,
    horizon: int,
    method: str,
) -> np.ndarray:
    official_issue = issue_hour if strategy == "Q4-3" else None
    return Q3.build_center_horizon(
        data, net_forecaster, pv_forecaster, pv_forecasts,
        day, official_issue, start_slot, horizon, method,
    )


def simulate_day(
    strategy: str,
    data: Any,
    actual_prices: np.ndarray,
    price_forecaster: CausalPriceForecaster,
    net_forecaster: Any,
    pv_forecaster: Any,
    pv_forecasts: Any,
    residual_matrix: np.ndarray,
    daily_methods: list[str],
    day: int,
    e_start: float,
    alpha: float,
    *,
    use_milp: bool,
    mpc_horizon: int,
) -> Q4DayResult:
    method = daily_methods[day]
    center = build_net_center(
        strategy, data, net_forecaster, pv_forecaster, pv_forecasts,
        day, 0, 0, PLAN_HORIZON, method,
    )
    risk_net, _, _ = Q3.risk_adjustment_at(
        day, 0, center, residual_matrix, alpha
    )
    forecast_price = price_forecaster.forecast_before_slot(day, 0, PLAN_HORIZON)
    plan0 = Q2.solve_day_ahead(
        risk_net, forecast_price, e_start, use_milp=use_milp
    )
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
    observed_net_residuals: list[float] = []
    node_records: list[dict] = []
    terminal_slacks = [float(plan0.state[-1] - e_start)]

    for t in range(SLOTS_PER_DAY):
        if strategy == "Q4-3" and t in (36, 72, 108):
            issue_hour = t // 6
            latest_issue_slot = t
            latest_center = build_net_center(
                strategy, data, net_forecaster, pv_forecaster, pv_forecasts,
                day, issue_hour, t, PLAN_HORIZON, method,
            )
            risk_updated, _, _ = Q3.risk_adjustment_at(
                day, t, latest_center, residual_matrix, alpha
            )
            updated_price = price_forecaster.forecast_before_slot(day, t, PLAN_HORIZON)
            previous_contract = ga[t:].copy()
            update = Q3.solve_adjusted_contract(
                risk_updated, updated_price, E[t], g0[t:], SLOTS_PER_DAY - t,
                use_milp=use_milp,
            )
            ga[t:] = update.grid[:SLOTS_PER_DAY - t]
            ref_grid[t:t + PLAN_HORIZON] = update.grid
            ref_state[t:t + PLAN_HORIZON + 1] = update.state
            terminal_slacks.append(float(update.state[-1] - E[t]))
            delta = ga[t:] - previous_contract
            changed = np.abs(delta) > Q3.CHANGE_TOL
            node_records.append({
                "issue_hour": issue_hour,
                "eligible_slots": int(len(delta)),
                "changed_slots": int(changed.sum()),
                "adjustment_rate": float(changed.mean()),
                "mean_abs_adjustment_changed_kwh": (
                    float(np.mean(np.abs(delta[changed]))) if changed.any() else 0.0
                ),
            })
            observed_net_residuals = []

        h = min(mpc_horizon, SLOTS_PER_DAY + PLAN_HORIZON - t)
        offset = t - latest_issue_slot
        center_h = latest_center[offset:offset + h].copy()
        observed_net_residuals.append(float(data.net[day, t] - center_h[0]))
        recent = np.asarray(observed_net_residuals[-6:])
        weights = 0.70 ** np.arange(len(recent) - 1, -1, -1)
        bias = float(np.dot(weights, recent) / weights.sum())
        online_net = center_h + bias * np.exp(-np.arange(h) / 18.0)
        online_net[0] = data.net[day, t]
        mpc_prices = price_forecaster.forecast_with_current(day, t, h)
        c, d, emergency, spill, e_next = Q2.solve_mpc_step(
            online_net, ref_grid[t:t + h], mpc_prices,
            E[t], ref_state[t + h],
        )
        C[t], D[t], H[t], W[t], E[t + 1] = c, d, emergency, spill, e_next

    realized_price = actual_prices[day]
    base_cost = float(np.dot(realized_price, g0))
    adjusted_cost = float(np.dot(realized_price, ga)
                          + ADJUSTMENT_MULTIPLIER * np.dot(
                              realized_price, np.abs(ga - g0)
                          ))
    emergency_cost = float(np.dot(Q2.EMERGENCY_MULTIPLIER * realized_price, H))
    return Q4DayResult(
        g0=g0, ga=ga, charge=C, discharge=D, emergency=H, spill=W, state=E,
        base_plan_cost=base_cost, adjusted_grid_cost=adjusted_cost,
        emergency_cost=emergency_cost, node_records=node_records,
        minimum_terminal_slack=float(min(terminal_slacks)),
    )


def simulate_strategy(strategy: str) -> list[Q4DayResult]:
    if _STRATEGY_CONTEXT is None:
        raise RuntimeError("Q4策略上下文未初始化。")
    ctx = _STRATEGY_CONTEXT
    e = float(Q2.E_INITIAL)
    results = []
    all_days = range(len(ctx["data"].dates))
    for pos, day in enumerate(all_days):
        alpha = Q2.DEFAULT_ALPHA if day < 31 else SELECTED_ALPHA
        result = simulate_day(
            strategy, ctx["data"], ctx["actual_prices"], ctx["price_forecaster"],
            ctx["net_forecaster"], ctx["pv_forecaster"], ctx["pv_forecasts"],
            ctx["residual_matrix"], ctx["daily_methods"], day, e, alpha,
            use_milp=ctx["use_milp"], mpc_horizon=ctx["mpc_horizon"],
        )
        results.append(result)
        e = float(result.state[-1])
        if (pos + 1) % 31 == 0 or pos + 1 == len(ctx["data"].dates):
            print(f"{strategy}: {pos + 1}/{len(ctx['data'].dates)} days", flush=True)
    return results


def validate_results(
    data: Any,
    days: Iterable[int],
    results: list[Q4DayResult],
    strategy: str,
) -> dict:
    max_balance = max_soc = max_both = max_cross = max_hc = 0.0
    min_soc, max_soc_seen, min_var = math.inf, -math.inf, math.inf
    max_c = max_d = 0.0
    prev = None
    prechange = 0.0
    for day, r in zip(days, results):
        balance = r.ga + r.emergency + data.pv[day] + r.discharge \
                  - data.load[day] - r.charge - r.spill
        soc = r.state[1:] - r.state[:-1] - Q2.ETA_C * r.charge + r.discharge / Q2.ETA_D
        max_balance = max(max_balance, float(np.max(np.abs(balance))))
        max_soc = max(max_soc, float(np.max(np.abs(soc))))
        max_both = max(max_both, float(np.max(r.charge * r.discharge)))
        max_hc = max(max_hc, float(np.max(r.emergency * r.charge)))
        min_soc = min(min_soc, float(r.state.min()))
        max_soc_seen = max(max_soc_seen, float(r.state.max()))
        min_var = min(min_var, float(r.g0.min()), float(r.ga.min()), float(r.charge.min()),
                      float(r.discharge.min()), float(r.emergency.min()), float(r.spill.min()))
        max_c = max(max_c, float(r.charge.max()))
        max_d = max(max_d, float(r.discharge.max()))
        cutoff = 36 if strategy == "Q4-3" else SLOTS_PER_DAY
        prechange = max(prechange, float(np.max(np.abs(r.ga[:cutoff] - r.g0[:cutoff]))))
        if prev is not None:
            max_cross = max(max_cross, abs(float(r.state[0]) - prev))
        prev = float(r.state[-1])
    return {
        "max_balance_residual_kwh": max_balance,
        "max_soc_residual_kwh": max_soc,
        "max_charge_discharge_product": max_both,
        "max_emergency_charge_product": max_hc,
        "max_cross_day_soc_gap_kwh": max_cross,
        "min_soc_kwh": min_soc,
        "max_soc_kwh": max_soc_seen,
        "minimum_nonnegative_variable_kwh": min_var,
        "max_charge_kwh_per_slot": max_c,
        "max_discharge_kwh_per_slot": max_d,
        "max_contract_change_before_allowed_node_kwh": prechange,
        "minimum_36h_terminal_slack_kwh": float(min(r.minimum_terminal_slack for r in results)),
        "actual_future_price_used_by_contract_optimizer": False,
        "price_observation_lookahead_slots": 0,
    }


def summarize(results: list[Q4DayResult], actual_prices: np.ndarray, days: list[int]) -> dict:
    ordinary_without_penalty = sum(
        float(np.dot(actual_prices[day], r.ga)) for day, r in zip(days, results)
    )
    adjusted_cost = sum(r.adjusted_grid_cost for r in results)
    emergency = sum(r.emergency_cost for r in results)
    return {
        "base_plan_cost": float(sum(r.base_plan_cost for r in results)),
        "realized_ordinary_grid_cost_before_adjustment_fee": float(ordinary_without_penalty),
        "adjustment_penalty": float(adjusted_cost - ordinary_without_penalty),
        "ordinary_grid_cost_including_adjustment_fee": float(adjusted_cost),
        "emergency_cost": float(emergency),
        "total_closed_loop_cost": float(adjusted_cost + emergency),
        "base_grid_energy": float(sum(r.g0.sum() for r in results)),
        "active_grid_energy": float(sum(r.ga.sum() for r in results)),
        "emergency_energy": float(sum(r.emergency.sum() for r in results)),
    }


def node_metrics(results: list[Q4DayResult]) -> dict:
    out = {}
    for hour in (6, 12, 18):
        records = [x for r in results for x in r.node_records if x["issue_hour"] == hour]
        eligible = sum(x["eligible_slots"] for x in records)
        changed = sum(x["changed_slots"] for x in records)
        amount = sum(x["mean_abs_adjustment_changed_kwh"] * x["changed_slots"] for x in records)
        out[str(hour)] = {
            "days_evaluated": len(records),
            "eligible_slots": eligible,
            "changed_slots": changed,
            "contract_adjustment_rate": float(changed / eligible) if eligible else 0.0,
            "mean_abs_adjustment_changed_kwh": float(amount / changed) if changed else 0.0,
        }
    return out


def strategy_payload(
    name: str,
    data: Any,
    actual_prices: np.ndarray,
    days: list[int],
    labels: list[str],
    results: list[Q4DayResult],
    full_results: list[Q4DayResult],
) -> dict:
    dates = [data.dates[d].strftime("%Y-%m-%d") for d in days]
    events = {date: Q2.group_emergency(labels, result.emergency)
              for date, result in zip(dates, results)}
    daily = []
    for day, date, result in zip(days, dates, results):
        ordinary = float(np.dot(actual_prices[day], result.ga))
        daily.append({
            "date": date,
            "base_grid_energy": float(result.g0.sum()),
            "base_plan_cost": result.base_plan_cost,
            "active_grid_energy": float(result.ga.sum()),
            "adjusted_grid_cost": result.adjusted_grid_cost,
            "adjustment_penalty": float(result.adjusted_grid_cost - ordinary),
            "emergency_energy": float(result.emergency.sum()),
            "emergency_cost": result.emergency_cost,
            "total_cost": float(result.adjusted_grid_cost + result.emergency_cost),
            "charge_energy": float(result.charge.sum()),
            "discharge_energy": float(result.discharge.sum()),
            "spill_energy": float(result.spill.sum()),
            "e_start": float(result.state[0]),
            "e_end": float(result.state[-1]),
        })
    summary = summarize(results, actual_prices, days)
    validation = validate_results(data, days, results, name)
    full_validation = validate_results(
        data, range(len(data.dates)), full_results, name
    )
    grouped = sum(event["energy"] for x in events.values() for event in x)
    validation.update({
        "dates_count": len(dates),
        "date_start_ok": dates[0] == "2025-02-01",
        "date_end_ok": dates[-1] == "2025-12-31",
        "slots_per_day": 144,
        "last_interval_has_day_offset": "+1" in labels[-1],
        "emergency_grouping_energy_gap_kwh": abs(
            grouped - float(sum(r.emergency.sum() for r in results))
        ),
        "past_contract_periods_modified": 0,
        "allowed_contract_update_nodes": [] if name == "Q4-2" else [6, 12, 18],
        "january_to_february_soc_gap_kwh": abs(
            float(full_results[30].state[-1]) - float(full_results[31].state[0])
        ),
        "full_year_max_cross_day_soc_gap_kwh": (
            full_validation["max_cross_day_soc_gap_kwh"]
        ),
    })
    return {
        "dates": dates,
        "interval_labels": labels,
        "plan_grid": np.vstack([r.g0 for r in results]).tolist(),
        "active_grid": np.vstack([r.ga for r in results]).tolist(),
        "charge": np.vstack([r.charge for r in results]).tolist(),
        "discharge": np.vstack([r.discharge for r in results]).tolist(),
        "emergency": np.vstack([r.emergency for r in results]).tolist(),
        "spill": np.vstack([r.spill for r in results]).tolist(),
        "state": np.vstack([r.state for r in results]).tolist(),
        "daily_summary": daily,
        "emergency_events": events,
        "overall": summary,
        "node_adjustment_metrics": node_metrics(results) if name == "Q4-3" else {},
        "validation": validation,
    }


def serialize_output(
    data: Any,
    actual_prices: np.ndarray,
    price_forecaster: CausalPriceForecaster,
    days: list[int],
    labels: list[str],
    results: dict[str, list[Q4DayResult]],
    full_results: dict[str, list[Q4DayResult]],
    pv_evaluation: dict,
    warmup_end_soc: dict[str, float],
    runtime: float,
) -> dict:
    strategies = {
        name: strategy_payload(
            name, data, actual_prices, days, labels, results[name], full_results[name]
        )
        for name in STRATEGIES
    }
    return {
        "metadata": {
            "model": "causal periodic price baseline plus selected short-memory residual model",
            "price_model": price_forecaster.metadata(),
            "q2_baseline_alpha": SELECTED_ALPHA,
            "initial_soc_2025_01_01_kwh": Q2.E_INITIAL,
            "january_warmup_alpha": Q2.DEFAULT_ALPHA,
            "feb_dec_alpha": SELECTED_ALPHA,
            "january_warmup_end_soc_kwh_by_strategy": warmup_end_soc,
            "statistics_period": "2025-02-01/2025-12-31",
            "Q4-2_contract_rule": "ordinary contract fixed at 0:00; price updates only affect MPC",
            "Q4-3_contract_rule": "ordinary contract may be re-evaluated only at 6:00/12:00/18:00",
            "Q4-2_net_forecast": "Q2 causal historical forecast",
            "Q4-3_net_forecast": "Attachment 3 updates within 24h; Q2 causal forecast beyond 24h",
            "pv_bias_calibration": pv_evaluation,
            "perfect_foresight_used_as_formal_plan": False,
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
        "strategies": strategies,
        "comparison": {
            "Q4-2_total_cost": strategies["Q4-2"]["overall"]["total_closed_loop_cost"],
            "Q4-3_total_cost": strategies["Q4-3"]["overall"]["total_closed_loop_cost"],
            "Q4-2_minus_Q4-3": float(
                strategies["Q4-2"]["overall"]["total_closed_loop_cost"]
                - strategies["Q4-3"]["overall"]["total_closed_loop_cost"]
            ),
        },
    }


_EMBEDDED_Q4_EXPORTER = r'''
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
let artifactTool;
try { artifactTool = await import("@oai/artifact-tool"); }
catch (error) {
  const mods=process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES; if(!mods) throw error;
  artifactTool=await import(pathToFileURL(path.join(mods,"@oai/artifact-tool/dist/artifact_tool.mjs")).href);
}
const {FileBlob,SpreadsheetFile}=artifactTool;
const [template42,template43,jsonPath,output42,output43]=process.argv.slice(2);
const payload=JSON.parse(await fs.readFile(jsonPath,"utf8"));
async function build(templatePath,outputPath,key,hasAdjusted){
  const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(templatePath));
  const p=payload.strategies[key], dates=p.dates.map(x=>new Date(`${x}T00:00:00Z`)), n=dates.length;
  if(n!==334||p.plan_grid.some(r=>r.length!==144)) throw new Error(`${key}结果尺寸错误`);
  const plan=wb.worksheets.getItem("计划购电量");
  const storage=wb.worksheets.getItem("充放电量");
  const emergency=wb.worksheets.getItem("紧急购电量");
  const adjusted=hasAdjusted?wb.worksheets.getItem("调整购电量"):null;
  const headers=[["日期\\时间",...p.interval_labels,"全天购电量","全天购电费"]];
  plan.getRange("A1:EQ1").values=headers;
  plan.getRange(`A2:EQ${n+1}`).values=p.plan_grid.map((row,i)=>[dates[i],...row,
    p.daily_summary[i].base_grid_energy,p.daily_summary[i].base_plan_cost]);
  if(adjusted){
    adjusted.getRange("A1:EQ1").values=headers;
    adjusted.getRange(`A2:EQ${n+1}`).values=p.active_grid.map((row,i)=>[dates[i],...row,
      p.daily_summary[i].active_grid_energy,p.daily_summary[i].adjusted_grid_cost]);
  }
  const periods=["0:00-4:00","4:00-8:00","8:00-12:00","12:00-16:00","16:00-20:00","20:00-24:00"];
  const storageRows=[];
  for(let i=0;i<n;i++) for(let b=0;b<6;b++){
    const a=b*24,z=a+24;
    storageRows.push([b===0?dates[i]:null,periods[b],
      p.charge[i].slice(a,z).reduce((s,x)=>s+x,0),
      p.discharge[i].slice(a,z).reduce((s,x)=>s+x,0),
      b===0?"0:00":(b===1?"24:00":null),
      b===0?p.state[i][0]:(b===1?p.state[i][144]:null)]);
  }
  storage.getRange("A1:F1").values=[["日期","时间段","充电量","放电量","时刻","储电量"]];
  storage.getRange(`A2:F${storageRows.length+1}`).values=storageRows;
  const emergencyRows=[];
  for(let i=0;i<n;i++){
    const events=p.emergency_events[p.dates[i]]||[];
    if(!events.length) emergencyRows.push([dates[i],"无",0]);
    else events.forEach((e,j)=>emergencyRows.push([j===0?dates[i]:null,e.period,e.energy]));
  }
  emergency.getRange("A1:C1").values=[["日期","购电时间段","购电量"]];
  emergency.getRange(`A2:C${emergencyRows.length+1}`).values=emergencyRows;
  const font="Noto Sans CJK SC",border={preset:"all",style:"thin",color:"#7F7F7F"};
  const blocks=[[plan,`A1:EQ${n+1}`],[storage,`A1:F${storageRows.length+1}`],
    [emergency,`A1:C${emergencyRows.length+1}`]];
  if(adjusted) blocks.push([adjusted,`A1:EQ${n+1}`]);
  for(const [sheet,used] of blocks){sheet.showGridLines=false;sheet.getRange(used).format.font={name:font,size:10,color:"#000000"};
    sheet.getRange(used).format.verticalAlignment="center";sheet.getRange(used).format.borders=border;sheet.getRange(used).format.rowHeight=20;}
  const headerBlocks=[[plan,"A1:EQ1",9],[storage,"A1:F1",10],[emergency,"A1:C1",10]];
  if(adjusted) headerBlocks.push([adjusted,"A1:EQ1",9]);
  for(const [sheet,range,size] of headerBlocks) sheet.getRange(range).format={fill:"#E7E6E6",
    font:{name:font,size,bold:true,color:"#000000"},horizontalAlignment:"center",verticalAlignment:"center",wrapText:true,borders:border};
  for(const sheet of adjusted?[plan,adjusted]:[plan]){sheet.getRange("A2:A335").setNumberFormat("yyyy-mm-dd");
    sheet.getRange("B2:EQ335").setNumberFormat("0.000");sheet.getRange("A1:A335").format.columnWidth=13;
    sheet.getRange("B1:EO335").format.columnWidth=13;sheet.getRange("EP1:EQ335").format.columnWidth=15;
    sheet.freezePanes.freezeRows(1);sheet.freezePanes.freezeColumns(1);}
  storage.getRange(`A2:A${storageRows.length+1}`).setNumberFormat("yyyy-mm-dd");
  storage.getRange(`C2:D${storageRows.length+1}`).setNumberFormat("0.000");storage.getRange(`F2:F${storageRows.length+1}`).setNumberFormat("0.000");
  storage.getRange(`A1:A${storageRows.length+1}`).format.columnWidth=14;storage.getRange(`B1:B${storageRows.length+1}`).format.columnWidth=16;
  storage.getRange(`C1:D${storageRows.length+1}`).format.columnWidth=14;storage.getRange(`E1:E${storageRows.length+1}`).format.columnWidth=11;
  storage.getRange(`F1:F${storageRows.length+1}`).format.columnWidth=14;storage.freezePanes.freezeRows(1);
  emergency.getRange(`A2:A${emergencyRows.length+1}`).setNumberFormat("yyyy-mm-dd");emergency.getRange(`C2:C${emergencyRows.length+1}`).setNumberFormat("0.000");
  emergency.getRange(`A2:C${emergencyRows.length+1}`).format.horizontalAlignment="center";emergency.getRange(`A1:A${emergencyRows.length+1}`).format.columnWidth=15;
  emergency.getRange(`B1:B${emergencyRows.length+1}`).format.columnWidth=24;emergency.getRange(`C1:C${emergencyRows.length+1}`).format.columnWidth=17;
  emergency.freezePanes.freezeRows(1);wb.recalculate();await fs.mkdir(path.dirname(path.resolve(outputPath)),{recursive:true});
  await(await SpreadsheetFile.exportXlsx(wb)).save(outputPath);
}
await build(template42,output42,"Q4-2",false);
await build(template43,output43,"Q4-3",true);
'''


def export_workbooks(
    template42: Path,
    template43: Path,
    payload_path: Path,
    output42: Path,
    output43: Path,
) -> None:
    node = os.environ.get("CODEX_PRIMARY_RUNTIME_NODE", "node")
    with tempfile.TemporaryDirectory(prefix="q4_export_") as temp_dir:
        script = Path(temp_dir) / "build_q4.mjs"
        script.write_text(_EMBEDDED_Q4_EXPORTER, encoding="utf-8")
        subprocess.run([node, str(script), str(template42), str(template43),
                        str(payload_path), str(output42), str(output43)], check=True)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q2", type=Path, required=True)
    parser.add_argument("--q3", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-price", type=Path, required=True)
    parser.add_argument("--pv-forecast", type=Path, required=True)
    parser.add_argument("--actual-price", type=Path, required=True)
    parser.add_argument("--template42", type=Path, required=True)
    parser.add_argument("--template43", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=here / "q4_results.json")
    parser.add_argument("--output42", type=Path, default=here / "result4-2_问题四.xlsx")
    parser.add_argument("--output43", type=Path, default=here / "result4-3_问题四.xlsx")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--fast", action="store_true",
                        help="仅调试：日前层用LP、MPC缩为1h；正式结果不得使用。")
    parser.add_argument("--skip-xlsx", action="store_true")
    return parser.parse_args()


def main() -> None:
    global Q2, Q3, _STRATEGY_CONTEXT
    args = parse_args()
    start_clock = time.perf_counter()
    Q2 = load_module(args.q2, "q2_baseline_for_q4")
    Q3 = load_module(args.q3, "q3_support_for_q4")
    Q3.load_q2_module(args.q2)
    data = Q2.load_inputs(args.data, args.base_price)
    actual_prices = load_actual_prices(args.actual_price, data.dates)
    pv_forecasts = Q3.load_pv_forecasts(args.pv_forecast, data.dates)
    pv_forecaster = Q3.CausalProfileForecaster(
        data.pv, data.dates, Q3.read_reference_pv(args.base_price)
    )
    net_forecaster = Q2.CausalForecaster(data)
    daily_methods, _, selected_predictions, _ = Q2.build_daily_forecast_policy(
        data, net_forecaster
    )
    residual_matrix = data.net - selected_predictions
    price_forecaster = CausalPriceForecaster(
        actual_prices, data.dates, residual_matrix
    )
    if price_forecaster.selected_model not in PRICE_MODELS:
        raise RuntimeError("电价模型选择失败。")
    labels = Q2.interval_labels_from_template(args.template42)
    labels43 = Q2.interval_labels_from_template(args.template43)
    if labels43 != labels:
        raise ValueError("两个Q4模板的144个时段顺序不一致。")
    output_days = list(range(31, len(data.dates)))
    pv_evaluation = Q3.evaluate_pv_bias_candidate(pv_forecasts, data)
    _STRATEGY_CONTEXT = {
        "data": data, "actual_prices": actual_prices,
        "price_forecaster": price_forecaster, "net_forecaster": net_forecaster,
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
        with context.Pool(processes=min(args.jobs, len(STRATEGIES))) as pool:
            values = pool.map(simulate_strategy, STRATEGIES)
        full_results = dict(zip(STRATEGIES, values))
    else:
        full_results = {name: simulate_strategy(name) for name in STRATEGIES}
    results = {name: values[31:] for name, values in full_results.items()}
    warmup_end_soc = {
        name: float(values[30].state[-1]) for name, values in full_results.items()
    }
    runtime = time.perf_counter() - start_clock
    payload = serialize_output(
        data, actual_prices, price_forecaster, output_days, labels, results,
        full_results, pv_evaluation, warmup_end_soc, runtime,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "price_model": payload["metadata"]["price_model"],
        "comparison": payload["comparison"],
        "Q4-2": payload["strategies"]["Q4-2"]["overall"],
        "Q4-3": payload["strategies"]["Q4-3"]["overall"],
        "Q4-2_validation": payload["strategies"]["Q4-2"]["validation"],
        "Q4-3_validation": payload["strategies"]["Q4-3"]["validation"],
    }, ensure_ascii=False, indent=2), flush=True)
    if not args.skip_xlsx:
        export_workbooks(
            args.template42, args.template43, args.output_json,
            args.output42, args.output43,
        )
        print(f"saved: {args.output42}")
        print(f"saved: {args.output43}")


if __name__ == "__main__":
    main()
