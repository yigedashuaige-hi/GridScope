from __future__ import annotations

import csv
import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

SITE_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = SITE_ROOT / "model"
DATA_ROOT = SITE_ROOT / "source" / "processed"
SLOTS = 144
DELTA_H = 1 / 6
CAPACITY_KWH = 12000.0


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load model module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


Q1 = _load_module(MODEL_ROOT / "q1.py", "gridscope_q1")


@lru_cache(maxsize=1)
def load_dataset() -> dict[str, dict[str, list[float]]]:
    data: dict[str, dict[str, list[float]]] = {}
    with (DATA_ROOT / "全年实际数据.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            date = row["日期"]
            target = data.setdefault(date, {"load_kw": [], "pv_kw": [], "price_yuan_kwh": [], "time": []})
            target["time"].append(row["时间"])
            target["load_kw"].append(float(row["小区负载（kW）"]))
            target["pv_kw"].append(float(row["光伏实际功率（kW）"]))
            target["price_yuan_kwh"].append(float(row["实际电价（元/kWh）"]))
    if not data or any(len(row["load_kw"]) != SLOTS for row in data.values()):
        raise RuntimeError("actual dataset must contain 144 slots per date")
    return data


def _finite_vector(values: Any, name: str) -> np.ndarray:
    if not isinstance(values, list) or len(values) != SLOTS:
        raise ValueError(f"{name} must contain exactly {SLOTS} values")
    out = np.asarray(values, dtype=float)
    if not np.isfinite(out).all() or (out < 0).any():
        raise ValueError(f"{name} must contain finite non-negative values")
    return out


def _bounded_factor(value: Any, name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out < 0.1 or out > 3.0:
        raise ValueError(f"{name} must be between 0.1 and 3.0")
    return out


def _soc(payload: dict[str, Any]) -> float:
    value = float(payload.get("initial_soc_percent", 50.0))
    if not math.isfinite(value) or not 10 <= value <= 90:
        raise ValueError("initial_soc_percent must be between 10 and 90")
    return value


def _scenario_arrays(payload: dict[str, Any]) -> tuple[str, str | None, np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
    mode = str(payload.get("mode", "quick"))
    if mode == "quick":
        date = str(payload.get("base_date", ""))
        base = load_dataset().get(date)
        if base is None:
            raise ValueError("base_date must be a date in the processed dataset")
        return (mode, date, np.asarray(base["load_kw"], dtype=float) * _bounded_factor(payload.get("load_multiplier", 1), "load_multiplier"), np.asarray(base["pv_kw"], dtype=float) * _bounded_factor(payload.get("pv_multiplier", 1), "pv_multiplier"), np.asarray(base["price_yuan_kwh"], dtype=float) * _bounded_factor(payload.get("price_multiplier", 1), "price_multiplier"), base["time"])
    if mode == "expert":
        return (mode, None, _finite_vector(payload.get("load_kw"), "load_kw"), _finite_vector(payload.get("pv_kw"), "pv_kw"), _finite_vector(payload.get("price_yuan_kwh"), "price_yuan_kwh"), payload.get("time"))
    raise ValueError("mode must be quick or expert")


def solve_scenario(payload: dict[str, Any]) -> dict[str, Any]:
    mode, date, load, pv, price, time_labels = _scenario_arrays(payload)
    initial_energy = CAPACITY_KWH * _soc(payload) / 100.0
    solution = Q1.solve_one_scenario(initial_energy, price, load * DELTA_H, pv * DELTA_H)
    arrays = {key: np.asarray(solution[key], dtype=float).round(8).tolist() for key in ("G", "C", "D", "W", "E")}
    return {"mode": mode, "source": f"processed/{date}" if date else "custom_input", "date": date, "slot_minutes": 10, "time": time_labels, "load_kw": load.tolist(), "pv_kw": pv.tolist(), "price_yuan_kwh": price.tolist(), "grid_kwh": arrays["G"], "charge_kwh": arrays["C"], "discharge_kwh": arrays["D"], "spill_kwh": arrays["W"], "soc_kwh": arrays["E"], "alpha": None, "alpha_applied": False, "alpha_note": "Q1 deterministic MILP; risk alpha is not used in Quick/Expert Scenario.", "metrics": {"total_cost_yuan": float(solution["total_cost"]), "grid_energy_kwh": float(solution["total_grid"]), "charge_energy_kwh": float(solution["total_charge"]), "discharge_energy_kwh": float(solution["total_discharge"]), "spill_energy_kwh": float(solution["total_curtailment"]), "emergency_energy_kwh": 0.0, "emergency_cost_yuan": 0.0, "initial_soc_kwh": initial_energy, "terminal_soc_kwh": float(np.asarray(solution["E"])[-1]), "max_balance_residual_kwh": solution["diagnostics"]["max_balance_residual"]}}


@lru_cache(maxsize=1)
def q2_context() -> dict[str, Any]:
    q2 = _load_module(MODEL_ROOT / "q2.py", "gridscope_q2")
    data = q2.load_inputs(DATA_ROOT / "附件2_处理后.xlsx", DATA_ROOT / "附件1_处理后.xlsx")
    forecaster = q2.CausalForecaster(data)
    methods, records, selected_predictions, _ = q2.build_daily_forecast_policy(data, forecaster)
    return {"q2": q2, "data": data, "forecaster": forecaster, "methods": methods, "records": records, "selected_predictions": selected_predictions, "residual": data.net - selected_predictions}


def _history_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    q2 = _load_module(MODEL_ROOT / "q2.py", "gridscope_q2_upload")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        date = str(row.get("date", ""))[:10]
        if not date:
            raise ValueError("history rows require a date column")
        grouped.setdefault(date, []).append(row)
    dates = sorted(grouped)
    if len(dates) < q2.RISK_RESIDUAL_WINDOW:
        raise ValueError(f"uploaded history must contain at least {q2.RISK_RESIDUAL_WINDOW} dates")
    if any(len(grouped[date]) != SLOTS for date in dates):
        raise ValueError("each uploaded date must contain exactly 144 rows")
    load_kw = np.asarray([[float(row["load_kw"]) for row in grouped[date]] for date in dates], dtype=float)
    pv_kw = np.asarray([[float(row["pv_kw"]) for row in grouped[date]] for date in dates], dtype=float)
    prices = np.asarray([[float(row["price_yuan_kwh"]) for row in grouped[date]] for date in dates], dtype=float)
    if not np.isfinite(np.c_[load_kw.ravel(), pv_kw.ravel(), prices.ravel()]).all() or (np.c_[load_kw, pv_kw, prices] < 0).any():
        raise ValueError("uploaded history contains invalid negative or non-finite values")
    pd = q2.pd
    data = q2.InputData(
        dates=pd.DatetimeIndex(pd.to_datetime(dates)),
        time_headers=[str(row.get("time", "")) for row in grouped[dates[0]]],
        load=load_kw * q2.DELTA_H, pv=pv_kw * q2.DELTA_H,
        net=(load_kw - pv_kw) * q2.DELTA_H,
        price=prices[0], reference_net=(load_kw[0] - pv_kw[0]) * q2.DELTA_H,
    )
    forecaster = q2.CausalForecaster(data)
    methods, records, selected_predictions, _ = q2.build_daily_forecast_policy(data, forecaster)
    return {"q2": q2, "data": data, "forecaster": forecaster, "methods": methods, "records": records, "selected_predictions": selected_predictions, "residual": data.net - selected_predictions}


def _day_index(date: str, data: Any) -> int:
    matches = np.flatnonzero(np.asarray([d.strftime("%Y-%m-%d") for d in data.dates]) == date)
    if len(matches) != 1:
        raise ValueError("date must be a valid date in the processed history")
    return int(matches[0])


def solve_forecast(payload: dict[str, Any]) -> dict[str, Any]:
    ctx = _history_context(payload["history_rows"]) if payload.get("history_rows") else q2_context(); q2, data, forecaster = ctx["q2"], ctx["data"], ctx["forecaster"]
    date = str(payload.get("date", ""))
    try:
        day = _day_index(date, data); target_in_history = True
    except ValueError:
        day = len(data.dates); target_in_history = False
        if date != data.dates[0].strftime("%Y-%m-%d") and date != (data.dates[0] + q2.pd.Timedelta(days=day)).strftime("%Y-%m-%d"):
            raise
    if day < q2.RISK_RESIDUAL_WINDOW:
        earliest = (data.dates[0] + q2.pd.Timedelta(days=q2.RISK_RESIDUAL_WINDOW)).strftime("%Y-%m-%d")
        raise ValueError(f"Forecast Mode requires at least {q2.RISK_RESIDUAL_WINDOW} completed history days; earliest supported target date is {earliest}")
    alpha = float(payload.get("alpha", 0.70))
    if not math.isfinite(alpha) or alpha not in q2.ALPHA_CANDIDATES:
        raise ValueError(f"alpha must be one of {q2.ALPHA_CANDIDATES}")
    e0 = CAPACITY_KWH * _soc(payload) / 100.0
    if target_in_history:
        method = ctx["methods"][day]
    else:
        candidates = {}
        hist = np.arange(max(0, day - q2.FORECAST_SELECTION_WINDOW), day, dtype=int)
        for candidate in q2.FORECAST_CANDIDATES:
            prediction = np.vstack([forecaster.forecast(h, h, candidate) for h in hist])
            error = data.net[hist] - prediction
            loss = np.sum(data.price[None, :] * (np.maximum(-error, 0.0) + q2.UNDERPURCHASE_EXTRA_MULTIPLIER * np.maximum(error, 0.0)))
            candidates[candidate] = float(loss)
        method = min(candidates, key=lambda name: (candidates[name], q2.FORECAST_CANDIDATES.index(name)))
    center = forecaster.horizon_forecast(day, method)
    risk_net, sigma, quantile = q2.risk_adjustment(day, center, ctx["residual"], alpha)
    plan = q2.solve_day_ahead(risk_net, np.tile(data.price, 2)[:q2.PLAN_HORIZON], e0, use_milp=True)
    if target_in_history:
        result = q2.simulate_day(data, forecaster, ctx["residual"], day, method, alpha, e0, use_milp=True, mpc_horizon=q2.MPC_HORIZON)
    else:
        grid = plan.grid[:SLOTS]; charge = np.zeros(SLOTS); discharge = np.zeros(SLOTS); emergency = np.zeros(SLOTS); spill = np.zeros(SLOTS); state = np.zeros(SLOTS + 1); state[0] = e0
        for t in range(SLOTS):
            h = min(q2.MPC_HORIZON, q2.PLAN_HORIZON - t)
            c, d, emergency_t, spill_t, e_next = q2.solve_mpc_step(center[t:t+h], plan.grid[t:t+h], np.tile(data.price, 2)[t:t+h], state[t], plan.state[t+h])
            charge[t], discharge[t], emergency[t], spill[t], state[t+1] = c, d, emergency_t, spill_t, e_next
        result = q2.DayResult(grid=grid, charge=charge, discharge=discharge, emergency=emergency, spill=spill, state=state, plan_state=plan.state[:SLOTS + 1], plan_terminal_state=float(plan.state[-1]), planned_cost=float(np.dot(data.price, grid)), emergency_cost=float(np.dot(q2.EMERGENCY_MULTIPLIER * data.price, emergency)))
    def arr(values: Any) -> list[float]: return np.asarray(values, dtype=float).round(8).tolist()
    selection_record = ctx["records"][day] if target_in_history else {"date": date, "selected_method": method, "selection_rule": "rolling_28d", "history_days": min(day, q2.FORECAST_SELECTION_WINDOW)}
    return {"mode":"forecast", "date":date, "history_days":int(payload.get("history_days", day)), "history_source":"repository processed history" if not payload.get("uploaded") else "uploaded history", "selected_predictor":method, "selection_rule":selection_record["selection_rule"], "selection_record":selection_record, "alpha":alpha, "slot_minutes":10, "center_forecast_kwh":arr(center[:SLOTS]), "risk_adjusted_forecast_kwh":arr(risk_net[:SLOTS]), "risk_margin_kwh":arr(risk_net[:SLOTS]-center[:SLOTS]), "day_ahead_plan_kwh":arr(plan.grid[:SLOTS]), "grid_kwh":arr(result.grid), "charge_kwh":arr(result.charge), "discharge_kwh":arr(result.discharge), "emergency_kwh":arr(result.emergency), "spill_kwh":arr(result.spill), "soc_kwh":arr(result.state), "non_anticipative":True, "current_observation_source":"historical current-step measurement" if target_in_history else "forecast-only current step (no target-day actual supplied)", "forecast_horizon_slots":q2.PLAN_HORIZON, "mpc_horizon_slots":q2.MPC_HORIZON, "metrics":{"plan_cost_yuan":result.planned_cost, "emergency_cost_yuan":result.emergency_cost, "total_cost_yuan":result.planned_cost+result.emergency_cost, "grid_energy_kwh":float(result.grid.sum()), "emergency_energy_kwh":float(result.emergency.sum()), "charge_energy_kwh":float(result.charge.sum()), "discharge_energy_kwh":float(result.discharge.sum()), "spill_energy_kwh":float(result.spill.sum()), "min_soc_kwh":float(result.state.min()), "max_soc_kwh":float(result.state.max()), "risk_sigma_mean_kwh":float(np.mean(sigma[:SLOTS])), "risk_quantile_mean":float(np.mean(quantile[:SLOTS]))}}


def solve_q3(payload: dict[str, Any]) -> dict[str, Any]:
    ctx = q2_context(); q2 = ctx["q2"]; q3 = _load_module(MODEL_ROOT / "q3.py", "gridscope_q3"); q3.load_q2_module(MODEL_ROOT / "q2.py")
    data = ctx["data"]; date = str(payload.get("date", "")); day = _day_index(date, data)
    if day < 31: raise ValueError("Q3 online mode requires historical context from 2025-02-01 onward")
    pv_forecasts = q3.load_pv_forecasts(DATA_ROOT / "附件3_处理后.xlsx", data.dates)
    pv_forecaster = q3.CausalProfileForecaster(data.pv, data.dates, q3.read_reference_pv(DATA_ROOT / "附件1_处理后.xlsx"))
    policy = str(payload.get("policy", "M3")); policy = policy if policy in q3.POLICIES else "M3"
    result = q3.simulate_day(data, ctx["forecaster"], pv_forecaster, pv_forecasts, ctx["residual"], ctx["methods"], day, CAPACITY_KWH * _soc(payload) / 100.0, policy, float(payload.get("alpha", 0.70)), use_milp=True, mpc_horizon=q3.MPC_HORIZON)
    return {"mode":"q3", "date":date, "policy":policy, "allowed_update_nodes":list(q3.POLICIES[policy]), "non_anticipative":True, "plan_kwh":result.g0.tolist(), "adjusted_plan_kwh":result.ga.tolist(), "charge_kwh":result.charge.tolist(), "discharge_kwh":result.discharge.tolist(), "emergency_kwh":result.emergency.tolist(), "spill_kwh":result.spill.tolist(), "soc_kwh":result.state.tolist(), "node_records":result.node_records, "metrics":{"base_plan_cost_yuan":result.base_plan_cost,"adjusted_grid_cost_yuan":result.adjusted_grid_cost,"emergency_cost_yuan":result.emergency_cost,"total_cost_yuan":result.adjusted_grid_cost+result.emergency_cost,"emergency_energy_kwh":float(result.emergency.sum())}}


def solve_q4(payload: dict[str, Any]) -> dict[str, Any]:
    ctx = q2_context(); q2 = ctx["q2"]; q3 = _load_module(MODEL_ROOT / "q3.py", "gridscope_q3_for_q4"); q3.load_q2_module(MODEL_ROOT / "q2.py"); q4 = _load_module(MODEL_ROOT / "q4.py", "gridscope_q4"); q4.Q2, q4.Q3 = q2, q3
    data = ctx["data"]; date = str(payload.get("date", "")); day = _day_index(date, data)
    if day < 31: raise ValueError("Q4 online mode requires historical context from 2025-02-01 onward")
    actual_prices = q4.load_actual_prices(DATA_ROOT / "附件4_处理后.xlsx", data.dates)
    pv_forecasts = q3.load_pv_forecasts(DATA_ROOT / "附件3_处理后.xlsx", data.dates)
    pv_forecaster = q3.CausalProfileForecaster(data.pv, data.dates, q3.read_reference_pv(DATA_ROOT / "附件1_处理后.xlsx"))
    price_forecaster = q4.CausalPriceForecaster(actual_prices, data.dates, ctx["residual"])
    strategy = str(payload.get("strategy", "Q4-2")); strategy = strategy if strategy in q4.STRATEGIES else "Q4-2"
    result = q4.simulate_day(strategy, data, actual_prices, price_forecaster, ctx["forecaster"], pv_forecaster, pv_forecasts, ctx["residual"], ctx["methods"], day, CAPACITY_KWH * _soc(payload) / 100.0, float(payload.get("alpha", 0.70)), use_milp=True, mpc_horizon=q4.MPC_HORIZON)
    def arr(values: Any) -> list[float]: return np.asarray(values, dtype=float).round(8).tolist()
    return {"mode":"q4", "date":date, "strategy":strategy, "price_model":price_forecaster.selected_model, "allowed_update_nodes":[] if strategy=="Q4-2" else [6,12,18], "non_anticipative":True, "price_forecast_yuan_kwh":arr(price_forecaster.forecast_before_slot(day,0,SLOTS)), "price_actual_yuan_kwh":arr(actual_prices[day]), "plan_kwh":arr(result.g0), "adjusted_plan_kwh":arr(result.ga), "charge_kwh":arr(result.charge), "discharge_kwh":arr(result.discharge), "emergency_kwh":arr(result.emergency), "spill_kwh":arr(result.spill), "soc_kwh":arr(result.state), "node_records":result.node_records, "metrics":{"base_plan_cost_yuan":result.base_plan_cost,"adjusted_grid_cost_yuan":result.adjusted_grid_cost,"emergency_cost_yuan":result.emergency_cost,"total_cost_yuan":result.adjusted_grid_cost+result.emergency_cost,"emergency_energy_kwh":float(result.emergency.sum())}}
