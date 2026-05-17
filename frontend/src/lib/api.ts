export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:5001";

function extractError(payload: unknown, status: number): string {
  if (payload && typeof payload === "object" && "error" in payload) {
    const { error } = payload as { error?: unknown };
    if (typeof error === "string") return error;
  }
  return `Request failed with status ${status}`;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
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

  return payload as T;
}

export type ColumnsResponse = {
  columns?: string[];
  error?: string;
};

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

export async function fetchColumns(tableName: string) {
  return apiFetch<ColumnsResponse>(
    `/get_columns/${encodeURIComponent(tableName)}`,
  );
}

export async function fetchHistoricalData(payload: HistoricalRequest) {
  return apiFetch<HistoricalResponse>("/api/historical_data", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchForecast(payload: ForecastRequest) {
  return apiFetch<ForecastResponse>("/api/forecast", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** Best-effort preview of how the backend will interpret a duration string. */
export function previewDuration(value: string): string | null {
  if (!value) return null;
  const trimmed = value.trim();
  const compact = trimmed.match(/^(\d+(?:\.\d+)?)\s*([A-Za-z]+)$/);
  const spaced = trimmed.match(/^(\d+(?:\.\d+)?)\s+(day|days|hour|hours|minute|minutes|week|weeks|month|months|year|years)$/i);
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
  if (u.startsWith("d") || u === "day" || u === "days") return `${num} day${num === 1 ? "" : "s"}`;
  if (u.startsWith("h") || u === "hour" || u === "hours") return `${num} hour${num === 1 ? "" : "s"}`;
  if (u === "min" || u.startsWith("minute")) return `${num} minute${num === 1 ? "" : "s"}`;
  if (u === "w" || u.startsWith("week")) return `${num} week${num === 1 ? "" : "s"}`;
  if (u === "m" || u.startsWith("month")) return `${num} month${num === 1 ? "" : "s"}`;
  if (u === "y" || u.startsWith("year")) return `${num} year${num === 1 ? "" : "s"}`;
  return null;
}
