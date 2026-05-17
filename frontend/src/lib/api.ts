export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:5001";

// ─── error extraction ────────────────────────────────────────────────────────

function extractError(payload: unknown, status: number): string {
  if (payload && typeof payload === "object" && "error" in payload) {
    const { error } = payload as { error?: unknown };
    if (typeof error === "string") return error;
  }
  return `Request failed with status ${status}`;
}

// ─── narrowing helpers ───────────────────────────────────────────────────────

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isStringArray(v: unknown): v is string[] {
  return Array.isArray(v) && v.every((x) => typeof x === "string");
}

function isRecordArray(v: unknown): v is Record<string, unknown>[] {
  return Array.isArray(v) && v.every(isRecord);
}

function fail(endpoint: string, problem: string): never {
  throw new Error(`Invalid response from ${endpoint}: ${problem}`);
}

// ─── fetch wrapper ───────────────────────────────────────────────────────────

async function apiFetch<T>(
  path: string,
  validate: (raw: unknown) => T,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
    ...init,
  });

  let payload: unknown = null;
  let parsedAsJson = true;
  try {
    payload = await response.json();
  } catch {
    parsedAsJson = false;
  }

  if (!response.ok) {
    throw new Error(extractError(payload, response.status));
  }
  if (!parsedAsJson) {
    throw new Error(
      `Expected JSON from ${path} but got a non-JSON response (status ${response.status}).`,
    );
  }

  return validate(payload);
}

// ─── type definitions + validators ───────────────────────────────────────────

export type ColumnsResponse = {
  columns?: string[];
  error?: string;
};

function validateColumns(raw: unknown): ColumnsResponse {
  if (!isRecord(raw)) fail("/get_columns", "not an object");
  const { columns, error } = raw as { columns?: unknown; error?: unknown };
  const out: ColumnsResponse = {};
  if (columns !== undefined) {
    if (!isStringArray(columns)) fail("/get_columns", "'columns' must be a string array");
    out.columns = columns;
  }
  if (error !== undefined) {
    if (typeof error !== "string") fail("/get_columns", "'error' must be a string");
    out.error = error;
  }
  return out;
}

export type HistoricalRequest = {
  table: string;
  ts_column: string;
  y_column: string;
  /** e.g. "2024-01-01" */
  start?: string;
  /** e.g. "2024-12-31" */
  end?: string;
  /** Resample to a pandas frequency before returning, e.g. "H", "D", "W". */
  resample_to_freq?: string;
  columns?: string[];
  max_rows?: number;
};

export type HistoricalResponse = {
  table: string;
  row_count: number;
  columns: string[];
  preview: Record<string, unknown>[];
};

function validateHistorical(raw: unknown): HistoricalResponse {
  if (!isRecord(raw)) fail("/api/historical_data", "not an object");
  const { table, row_count, columns, preview } = raw as Record<string, unknown>;
  if (typeof table !== "string") fail("/api/historical_data", "'table' must be a string");
  if (typeof row_count !== "number") fail("/api/historical_data", "'row_count' must be a number");
  if (!isStringArray(columns)) fail("/api/historical_data", "'columns' must be a string array");
  if (!isRecordArray(preview)) fail("/api/historical_data", "'preview' must be an array of objects");
  return { table, row_count, columns, preview };
}

export type ForecastRequest = {
  table: string;
  ts_column: string;
  y_column: string;
  /** One of: "D", "W", "M", "H". */
  freq: string;
  /** Calendar-aware duration, e.g. "90D", "12M", "2Y", "365 days". */
  horizon: string;
  series_name: string;
  regressors: string[];
  resample_to_freq?: string;
  /** Calendar-aware duration, e.g. "730 days", "2Y". */
  training_window_duration?: string;
  auto_tune?: boolean;
  /** Opt-in IQR outlier removal (holiday rows are preserved). */
  remove_outliers?: boolean;
  preview_rows?: number;
};

export type ForecastResponse = {
  series_name: string;
  row_count: number;
  preview: Record<string, unknown>[];
  forecast_plot?: string;
  components_plot?: string;
  day_breakdown_plot?: string;
  training_rows?: number;
  training_start?: string;
  training_end?: string;
  outliers_removed?: number;
};

function optString(raw: unknown, key: string, endpoint: string): string | undefined {
  if (raw === undefined || raw === null) return undefined;
  if (typeof raw !== "string") fail(endpoint, `'${key}' must be a string`);
  return raw;
}

function optNumber(raw: unknown, key: string, endpoint: string): number | undefined {
  if (raw === undefined || raw === null) return undefined;
  if (typeof raw !== "number") fail(endpoint, `'${key}' must be a number`);
  return raw;
}

function validateForecast(raw: unknown): ForecastResponse {
  const endpoint = "/api/forecast";
  if (!isRecord(raw)) fail(endpoint, "not an object");
  const r = raw as Record<string, unknown>;
  if (typeof r.series_name !== "string") fail(endpoint, "'series_name' must be a string");
  if (typeof r.row_count !== "number") fail(endpoint, "'row_count' must be a number");
  if (!isRecordArray(r.preview)) fail(endpoint, "'preview' must be an array of objects");

  return {
    series_name: r.series_name,
    row_count: r.row_count,
    preview: r.preview,
    forecast_plot: optString(r.forecast_plot, "forecast_plot", endpoint),
    components_plot: optString(r.components_plot, "components_plot", endpoint),
    day_breakdown_plot: optString(r.day_breakdown_plot, "day_breakdown_plot", endpoint),
    training_rows: optNumber(r.training_rows, "training_rows", endpoint),
    training_start: optString(r.training_start, "training_start", endpoint),
    training_end: optString(r.training_end, "training_end", endpoint),
    outliers_removed: optNumber(r.outliers_removed, "outliers_removed", endpoint),
  };
}

// ─── public API ──────────────────────────────────────────────────────────────

export async function fetchColumns(tableName: string) {
  return apiFetch<ColumnsResponse>(
    `/get_columns/${encodeURIComponent(tableName)}`,
    validateColumns,
  );
}

export async function fetchHistoricalData(payload: HistoricalRequest) {
  return apiFetch<HistoricalResponse>(
    "/api/historical_data",
    validateHistorical,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

export async function fetchForecast(payload: ForecastRequest) {
  return apiFetch<ForecastResponse>(
    "/api/forecast",
    validateForecast,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

// ─── client-side duration preview ────────────────────────────────────────────

/** Best-effort preview of how the backend will interpret a duration string. */
export function previewDuration(value: string): string | null {
  if (!value) return null;
  const trimmed = value.trim();
  const compact = trimmed.match(/^(\d+(?:\.\d+)?)\s*([A-Za-z]+)$/);
  const spaced = trimmed.match(
    /^(\d+(?:\.\d+)?)\s+(day|days|hour|hours|minute|minutes|week|weeks|month|months|year|years)$/i,
  );
  let num: number | null = null;
  let unit = "";
  if (compact) {
    num = Number(compact[1]);
    unit = compact[2];
  } else if (spaced) {
    num = Number(spaced[1]);
    unit = spaced[2];
  }
  if (num === null) return null;
  const u = unit.toLowerCase();
  if (u === "d" || u.startsWith("day")) return `${num} day${num === 1 ? "" : "s"}`;
  if (u === "h" || u.startsWith("hour")) return `${num} hour${num === 1 ? "" : "s"}`;
  if (u === "min" || u.startsWith("minute")) return `${num} minute${num === 1 ? "" : "s"}`;
  if (u === "w" || u.startsWith("week")) return `${num} week${num === 1 ? "" : "s"}`;
  if (u === "m" || u.startsWith("month")) return `${num} month${num === 1 ? "" : "s"}`;
  if (u === "y" || u.startsWith("year")) return `${num} year${num === 1 ? "" : "s"}`;
  return null;
}
