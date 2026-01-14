const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:5001";

function extractError(payload: unknown, status: number) {
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
  try {
    payload = await response.json();
  } catch {
    // If the server responded with non-JSON, surface a generic error.
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }
    return {} as T;
  }

  if (!response.ok) {
    throw new Error(extractError(payload, response.status));
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
  start?: string;
  end?: string;
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
  freq: string;
  horizon: string;
  series_name: string;
  regressors: string[];
  resample_to_freq?: string;
  training_window_duration?: string;
  auto_tune?: boolean;
  preview_rows?: number;
};

export type ForecastResponse = {
  series_name: string;
  row_count: number;
  preview: Record<string, unknown>[];
  forecast_plot?: string;
  components_plot?: string;
  day_breakdown_plot?: string;
};

export async function fetchColumns(tableName: string) {
  return apiFetch<ColumnsResponse>(`/get_columns/${tableName}`);
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
