# Prophet Forecasting UI (shadcn)

React/Next.js front end styled with shadcn/ui + Tailwind that sits on top of the Flask API in this repo.

## Prereqs
- Backend running: `.venv/bin/python src/prophet_forecasting_tool/app.py` (defaults to http://127.0.0.1:5001)
- Node 18+ (already installed) and npm dependencies (`npm install` already run)

## Run it
```bash
# from repo root
cd frontend
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:5001 npm run dev
```
Open http://localhost:3000 to use the UI.

## What it calls
- `GET /get_columns/<table>` to load column names
- `POST /api/historical_data` with `table`, `ts_column`, `y_column`, optional `start/end`, `resample_to_freq`, `max_rows`
- `POST /api/forecast` with frequency/horizon/series details and optional regressors/resample/training window

## Defaults
- Table `call_center_metrics`, columns `ts`/`y` (from the dummy seed script)
- Forecast horizon `90D`, training window `730 days`, frequency `D`

## Notes
- Regresors list auto-populates from table columns excluding `ts`/`y`
- If you need a different API host/port, override `NEXT_PUBLIC_API_BASE_URL`
- The UI shows preview rows plus base64 plots from the Flask service (no file I/O needed)
