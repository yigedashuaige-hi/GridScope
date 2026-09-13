# GridScope / 微网智能调度站

## Live Demo

- 前端（GitHub Pages）：<https://yigedashuaige-hi.github.io/GridScope/>
- API 状态（Render）：<https://gridscope-api-yigedashuaige.onrender.com/api/health>
- Evidence Mode 完全读取站点内的 `data/web_data.json`，无需后端即可浏览 334 天、144 个十分钟时段和 Causal Replay。
- Lab / Forecast Mode 需要在线 Solver；Render 免费实例首次访问可能显示 `SOLVER WAKING` 并等待冷启动。

GitHub Pages 由 `.github/workflows/deploy-pages.yml` 在 `main` push 后自动发布，只上传首页运行所需的静态文件和 `data/web_data.json`。Render Blueprint 已配置并在线运行；服务固定 Python 3.12，NumPy/SciPy/Pandas 使用锁定的预编译 wheel，构建命令禁止这三个包从源码编译。服务名保持 `gridscope-api-yigedashuaige` 时，上面的 API 地址无需修改。

若需要重新创建 Render 服务，可选择 `New` → `Web Service`，仓库选 `yigedashuaige-hi/GridScope`，Runtime 选 Python，使用仓库中的 `render.yaml`；手动配置时 Build Command 填 `python -m pip install --upgrade pip && python -m pip install --no-cache-dir --only-binary=numpy,scipy,pandas -r backend/requirements.txt`，Start Command 填 `uvicorn backend.server:app --host 0.0.0.0 --port $PORT`，Health Check Path 填 `/api/health`，并将服务名设为 `gridscope-api-yigedashuaige`。

GridScope 是一个面向光储微网的科研控制台：用最终 Q1–Q4 结果做可追溯的历史回放，并在隔离的 Scenario Lab 中调用仓库内最终 Q1–Q4 模型进行自定义情景优化。

## 功能边界

- **Evidence Mode**：`data/web_data.json` 由最终 processed/result 文件只读导出；日期范围为 2025-02-01 至 2025-12-31，每日 144 个十分钟点。紧急购电保留结果工作簿记录的连续事件，不推断不存在的逐点拆分。
- **Lab Mode / Quick Scenario**：选择真实 base day，调整负荷、PV、电价倍率和初始 SOC，后端复用仓库 `model/q1.py` 的 `solve_one_scenario` 做真实 MILP。
- **Lab Mode / Expert Input**：粘贴 `time,load_kw,pv_kw,price_yuan_kwh` 表头加 144 行，或上传 CSV/XLSX，再调用相同求解器。
- **Forecast Mode**：直接调用 `model/q2.py` 的 28 天滚动预测器选择、56 天风险残差、风险分位、36 h 日前 MILP 与 10 min MPC；站内历史可用，目标日期至少需要 56 个已发生日。
- **Q3 online**：`/api/q3/solve` 复用 `model/q3.py`，支持 M0/M1/M2/M3（0/6/12/18 PV 更新与合同调整权限）。
- **Q4 online**：`/api/q4/solve` 复用 `model/q4.py`，支持 Q4-2（合同固定）和 Q4-3（6/12/18 可调合同），并返回 ARX/实际价格路径。
- **Causal Replay**：按 10 分钟回放，当前时刻后的实际值由信息幕帘隐藏；首页策略指标和电池 SOC 随日期更新。

Quick/Expert 的 `alpha` 会被校验并随实验记录返回，但确定性 MILP 不会擅自添加未在最终模型中定义的风险裕量；真正的 Q2 风险闭环仍以最终 q2.py 的完整历史上下文为前提。

## 目录

```text
微网智能调度站/
├── index.html                 # 静态前端
├── styles.css / app.js
├── data/web_data.json         # 导出后的 Evidence Mode 快照
├── model/q1.py ... q4.py      # 在线模型代码（仓库自包含）
├── source/processed/          # 在线模型需要的处理数据
├── source/results/            # 只读论文结果工作簿
├── exporter/export_web_data.py
└── backend/
    ├── server.py              # FastAPI API
    ├── model_service.py       # q1.py MILP 适配层
    └── requirements.txt
```

## 运行

终端一：生成或刷新网页快照（只写入站点 `data/`，不会写入官方/结果目录）：

```bash
backend/.venv/bin/python exporter/export_web_data.py
```

终端二：启动场景求解 API：

```bash
backend/.venv/bin/uvicorn server:app --app-dir backend --host 127.0.0.1 --port 8000
```

终端三：启动静态前端：

```bash
python3 -m http.server 4173
```

打开 `http://127.0.0.1:4173`。API 健康检查为 `http://127.0.0.1:8000/api/health`。

若需要重新创建环境：

```bash
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
```

## API 示例

```bash
curl -X POST http://127.0.0.1:8000/api/scenario/solve \
  -H 'Content-Type: application/json' \
  -d '{"mode":"quick","base_date":"2025-02-01","load_multiplier":1.05,"pv_multiplier":0.95,"price_multiplier":1.1,"initial_soc_percent":55,"alpha":0.70}'
```

响应包含 144 个 `grid_kwh/charge_kwh/discharge_kwh` 动作、145 个 `soc_kwh` 状态以及总成本、购电量和求解诊断。用户实验只保存在浏览器 `localStorage` 的最近摘要中，不写回正式结果。

Forecast API 示例：

```bash
curl -X POST http://127.0.0.1:8000/api/forecast/solve \
  -H 'Content-Type: application/json' \
  -d '{"date":"2025-03-01","initial_soc_percent":50,"alpha":0.70,"history_days":56}'
```

Forecast Mode 的单日输入不会被称为预测；站内历史或上传 CSV/XLSX 均需长表 `date,time,load_kw,pv_kw,price_yuan_kwh`，每日至少 144 行且至少 56 个日期。Expert Input 的格式固定为 `time,load_kw,pv_kw,price_yuan_kwh` 表头加 144 行十分钟数据。Q1 Quick/Expert 隐藏风险 α，因为确定性 MILP 不使用该参数。
