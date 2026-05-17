/* eslint-disable @next/next/no-img-element */
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { Loader2, RefreshCcw } from "lucide-react";

import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  API_BASE,
  ColumnsResponse,
  ForecastResponse,
  HistoricalResponse,
  fetchColumns,
  fetchForecast,
  fetchHistoricalData,
  previewDuration,
} from "@/lib/api";

const DEFAULT_TABLE = "call_center_metrics";
const DEFAULT_TS = "ts";
const DEFAULT_Y = "y";
const DEFAULT_SERIES = "call_volume";
const DEFAULT_HORIZON = "90D";
const DEFAULT_TRAINING_WINDOW = "730 days";

type ColumnState = "loading" | "ok" | "missing" | "error";

function formatNumber(value?: number) {
  if (typeof value !== "number") return "";
  return value.toLocaleString();
}

function DataPreviewTable({
  rows,
  columns,
  emptyLabel,
  maxColumns = 8,
}: {
  rows: Record<string, unknown>[];
  columns: string[];
  emptyLabel: string;
  maxColumns?: number;
}) {
  const [showAll, setShowAll] = useState(false);
  const displayedColumns = showAll ? columns : columns.slice(0, maxColumns);
  const truncated = !showAll && columns.length > maxColumns;

  if (!rows.length) {
    return <p className="text-sm text-muted-foreground">{emptyLabel}</p>;
  }

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto rounded-sm border bg-card">
        <Table>
          <TableHeader>
            <TableRow>
              {displayedColumns.map((col) => (
                <TableHead key={col} className="whitespace-nowrap">
                  {col}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row, idx) => (
              <TableRow key={idx}>
                {displayedColumns.map((col) => (
                  <TableCell key={col} className="whitespace-nowrap text-sm">
                    {String(row[col] ?? "").slice(0, 120)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      {truncated && (
        <button
          type="button"
          className="text-xs text-muted-foreground underline"
          onClick={() => setShowAll(true)}
        >
          Showing {maxColumns} of {columns.length} columns — show all
        </button>
      )}
    </div>
  );
}

function PlotPanel({ title, src, alt }: { title: string; src?: string; alt: string }) {
  if (!src) return null;
  return (
    <figure className="rounded-md border bg-muted/40 p-3">
      <figcaption className="mb-2 flex items-center justify-between text-sm font-semibold">
        <span>{title}</span>
        <a
          href={src}
          download={`${title.toLowerCase().replace(/\s+/g, "-")}.png`}
          className="text-xs font-normal text-muted-foreground underline"
        >
          Download
        </a>
      </figcaption>
      <a href={src} target="_blank" rel="noreferrer noopener" aria-label={`Open ${title} full size`}>
        <img src={src} alt={alt} className="w-full rounded-sm border bg-background" loading="lazy" />
      </a>
    </figure>
  );
}

function useElapsed(running: boolean) {
  // All state mutations happen inside async callbacks (setInterval/setTimeout)
  // so we stay clear of react-hooks/set-state-in-effect.
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!running) {
      return;
    }
    const t0 = performance.now();
    const initial = window.setTimeout(() => setElapsed(0), 0);
    const id = window.setInterval(() => {
      setElapsed((performance.now() - t0) / 1000);
    }, 250);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(id);
      window.setTimeout(() => setElapsed(0), 0);
    };
  }, [running]);

  return elapsed;
}

const INITIAL_STATE = {
  tableName: DEFAULT_TABLE,
  tsColumn: DEFAULT_TS,
  yColumn: DEFAULT_Y,
  freq: "D",
  horizon: DEFAULT_HORIZON,
  trainingWindow: DEFAULT_TRAINING_WINDOW,
  resample: "",
  seriesName: DEFAULT_SERIES,
  autoTune: false,
  removeOutliers: false,
  selectedRegressors: [] as string[],
  histStart: "",
  histEnd: "",
  histResample: "",
  maxRows: 200,
};

export default function Home() {
  const [tableName, setTableName] = useState(INITIAL_STATE.tableName);
  const [tsColumn, setTsColumn] = useState(INITIAL_STATE.tsColumn);
  const [yColumn, setYColumn] = useState(INITIAL_STATE.yColumn);
  const [columns, setColumns] = useState<string[]>([]);
  const [columnState, setColumnState] = useState<ColumnState>("loading");
  const [columnError, setColumnError] = useState<string | null>(null);

  const [freq, setFreq] = useState(INITIAL_STATE.freq);
  const [horizon, setHorizon] = useState(INITIAL_STATE.horizon);
  const [trainingWindow, setTrainingWindow] = useState(INITIAL_STATE.trainingWindow);
  const [resample, setResample] = useState(INITIAL_STATE.resample);
  const [seriesName, setSeriesName] = useState(INITIAL_STATE.seriesName);
  const [autoTune, setAutoTune] = useState(INITIAL_STATE.autoTune);
  const [removeOutliers, setRemoveOutliers] = useState(INITIAL_STATE.removeOutliers);
  const [selectedRegressors, setSelectedRegressors] = useState<string[]>(INITIAL_STATE.selectedRegressors);

  const [histStart, setHistStart] = useState(INITIAL_STATE.histStart);
  const [histEnd, setHistEnd] = useState(INITIAL_STATE.histEnd);
  const [histResample, setHistResample] = useState(INITIAL_STATE.histResample);
  const [maxRows, setMaxRows] = useState(INITIAL_STATE.maxRows);

  const [historical, setHistorical] = useState<HistoricalResponse | null>(null);
  const [forecastResult, setForecastResult] = useState<ForecastResponse | null>(null);

  const [loadingForecast, setLoadingForecast] = useState(false);
  const [loadingHistorical, setLoadingHistorical] = useState(false);
  const [errorForecast, setErrorForecast] = useState<string | null>(null);
  const [errorHistorical, setErrorHistorical] = useState<string | null>(null);

  const [tab, setTab] = useState<string>("historical");

  const forecastElapsed = useElapsed(loadingForecast);

  const availableRegressors = useMemo(
    () => columns.filter((col) => col !== tsColumn && col !== yColumn),
    [columns, tsColumn, yColumn],
  );

  const tsValid = columnState === "ok" && columns.includes(tsColumn);
  const yValid = columnState === "ok" && columns.includes(yColumn);

  const horizonPreview = previewDuration(horizon);
  const trainingWindowPreview = previewDuration(trainingWindow);

  const horizonError = horizon && !horizonPreview ? "Could not parse duration. Try '90D', '12M', '2Y'." : null;
  const trainingWindowError = trainingWindow && !trainingWindowPreview
    ? "Could not parse duration. Try '730 days', '2Y'."
    : null;

  const canForecast =
    !loadingForecast &&
    columnState === "ok" &&
    tsValid &&
    yValid &&
    !horizonError &&
    !trainingWindowError &&
    seriesName.trim().length > 0;

  const loadColumns = useCallback(async () => {
    setColumnState("loading");
    setColumnError(null);
    try {
      const res: ColumnsResponse = await fetchColumns(tableName);
      if (res.columns) {
        setColumns(res.columns);
        setSelectedRegressors((prev) => prev.filter((col) => res.columns?.includes(col)));
        setColumnState("ok");
      } else {
        setColumns([]);
        setColumnState("missing");
        setColumnError(res.error || `Table "${tableName}" was not found.`);
      }
    } catch (err) {
      setColumns([]);
      setColumnState("error");
      setColumnError(err instanceof Error ? err.message : "Could not reach the API.");
    }
  }, [tableName]);

  useEffect(() => {
    loadColumns();
  }, [loadColumns]);

  const handleForecast = async () => {
    setLoadingForecast(true);
    setErrorForecast(null);
    try {
      const response = await fetchForecast({
        table: tableName,
        ts_column: tsColumn,
        y_column: yColumn,
        freq,
        horizon,
        series_name: seriesName || "forecast",
        regressors: selectedRegressors,
        resample_to_freq: resample || undefined,
        training_window_duration: trainingWindow,
        auto_tune: autoTune,
        remove_outliers: removeOutliers,
        preview_rows: 60,
      });
      setForecastResult(response);
    } catch (err) {
      setErrorForecast(err instanceof Error ? err.message : "Unknown error");
      setForecastResult(null);
    } finally {
      setLoadingForecast(false);
    }
  };

  const handleLoadHistorical = async () => {
    setLoadingHistorical(true);
    setErrorHistorical(null);
    try {
      const response = await fetchHistoricalData({
        table: tableName,
        ts_column: tsColumn,
        y_column: yColumn,
        start: histStart || undefined,
        end: histEnd || undefined,
        resample_to_freq: histResample || undefined,
        max_rows: maxRows,
      });
      setHistorical(response);
    } catch (err) {
      setErrorHistorical(err instanceof Error ? err.message : "Failed to load historical data");
      setHistorical(null);
    } finally {
      setLoadingHistorical(false);
    }
  };

  const toggleRegressor = (value: string) => {
    setSelectedRegressors((prev) =>
      prev.includes(value) ? prev.filter((r) => r !== value) : [...prev, value],
    );
  };

  const handleRestartSession = () => {
    setTableName(INITIAL_STATE.tableName);
    setTsColumn(INITIAL_STATE.tsColumn);
    setYColumn(INITIAL_STATE.yColumn);
    setFreq(INITIAL_STATE.freq);
    setHorizon(INITIAL_STATE.horizon);
    setTrainingWindow(INITIAL_STATE.trainingWindow);
    setResample(INITIAL_STATE.resample);
    setSeriesName(INITIAL_STATE.seriesName);
    setAutoTune(INITIAL_STATE.autoTune);
    setRemoveOutliers(INITIAL_STATE.removeOutliers);
    setSelectedRegressors(INITIAL_STATE.selectedRegressors);
    setHistStart(INITIAL_STATE.histStart);
    setHistEnd(INITIAL_STATE.histEnd);
    setHistResample(INITIAL_STATE.histResample);
    setMaxRows(INITIAL_STATE.maxRows);
    setHistorical(null);
    setForecastResult(null);
    setErrorForecast(null);
    setErrorHistorical(null);
    setTab("historical");
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <main className="mx-auto max-w-6xl px-6 pb-16 pt-10">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div
              aria-hidden="true"
              className="flex h-12 w-12 items-center justify-center rounded-sm bg-gradient-to-br from-primary to-emerald-400 text-lg font-semibold text-primary-foreground"
            >
              PF
            </div>
            <div>
              <p className="text-xs font-mono uppercase tracking-[0.4em] text-muted-foreground">
                Prophet Forecasting
              </p>
              <h1 className="text-2xl font-mono font-semibold uppercase tracking-[0.12em]">
                Control Room
              </h1>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            {process.env.NODE_ENV !== "production" && (
              <Badge variant="outline" className="font-mono">
                API: {API_BASE}
              </Badge>
            )}
            <Button asChild variant="secondary">
              <a href="https://facebook.github.io/prophet/" target="_blank" rel="noreferrer noopener">
                Prophet Docs
              </a>
            </Button>
            <Button onClick={handleRestartSession} variant="outline" className="gap-2">
              <RefreshCcw className="h-4 w-4" aria-hidden="true" />
              Restart Session
            </Button>
          </div>
        </header>

        <section className="mt-8 grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle>Source</CardTitle>
              <CardDescription>
                Point the UI at any PostgreSQL table reachable from the API.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="text-xs uppercase text-muted-foreground" htmlFor="table-input">
                    Table
                  </label>
                  <Input
                    id="table-input"
                    value={tableName}
                    onChange={(e) => setTableName(e.target.value)}
                    className="mt-1"
                  />
                </div>
                <div>
                  <label className="text-xs uppercase text-muted-foreground" htmlFor="ts-input">
                    Timestamp
                  </label>
                  <Input
                    id="ts-input"
                    value={tsColumn}
                    onChange={(e) => setTsColumn(e.target.value)}
                    aria-invalid={columnState === "ok" && !tsValid}
                    className="mt-1"
                  />
                  {columnState === "ok" && !tsValid && (
                    <p className="mt-1 text-xs text-destructive">Not a column in {tableName}.</p>
                  )}
                </div>
                <div>
                  <label className="text-xs uppercase text-muted-foreground" htmlFor="y-input">
                    Value
                  </label>
                  <Input
                    id="y-input"
                    value={yColumn}
                    onChange={(e) => setYColumn(e.target.value)}
                    aria-invalid={columnState === "ok" && !yValid}
                    className="mt-1"
                  />
                  {columnState === "ok" && !yValid && (
                    <p className="mt-1 text-xs text-destructive">Not a column in {tableName}.</p>
                  )}
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                Columns refresh when you change the table name.
              </p>
              {columnState !== "ok" && columnError && (
                <Alert variant="destructive">
                  <AlertTitle>Couldn&apos;t inspect table</AlertTitle>
                  <AlertDescription>{columnError}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Data health</CardTitle>
              <CardDescription>Quick snapshot of what the API returns for this source.</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <p className="text-xs uppercase text-muted-foreground">Columns</p>
                <p className="text-lg font-semibold">{columns.length || "–"}</p>
              </div>
              <div>
                <p className="text-xs uppercase text-muted-foreground">Regressors available</p>
                <p className="text-lg font-semibold">{availableRegressors.length || "–"}</p>
              </div>
              <div className="col-span-2 text-xs text-muted-foreground">
                Status:{" "}
                {columnState === "loading" && "loading…"}
                {columnState === "ok" && "API is reachable."}
                {columnState === "missing" && "Table not found in this database."}
                {columnState === "error" && "Could not reach the API."}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Runbook</CardTitle>
              <CardDescription>1. Inspect history. 2. Pick regressors. 3. Forecast.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-emerald-500" aria-hidden="true" />
                <span>
                  Dummy data lives in <code>{DEFAULT_TABLE}</code> ({DEFAULT_TS} / {DEFAULT_Y}).
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-amber-500" aria-hidden="true" />
                <span>Horizon accepts strings: 90D, 12M, 2Y.</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-primary" aria-hidden="true" />
                <span>Auto-tune adds 1-10 minutes for CV passes.</span>
              </div>
            </CardContent>
          </Card>
        </section>

        <Tabs value={tab} onValueChange={setTab} className="mt-10">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="historical">1. Historical Data</TabsTrigger>
            <TabsTrigger value="forecast">2. Forecast</TabsTrigger>
          </TabsList>

          <TabsContent value="historical" className="mt-6">
            <div className="grid gap-6 lg:grid-cols-[1fr_0.9fr]">
              <Card>
                <CardHeader>
                  <CardTitle>Historical data</CardTitle>
                  <CardDescription>
                    Pull a preview from the database the Flask API is reading.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium" htmlFor="hist-start">
                        Start (optional)
                      </label>
                      <Input
                        id="hist-start"
                        value={histStart}
                        onChange={(e) => setHistStart(e.target.value)}
                        placeholder="2021-01-01"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium" htmlFor="hist-end">
                        End (optional)
                      </label>
                      <Input
                        id="hist-end"
                        value={histEnd}
                        onChange={(e) => setHistEnd(e.target.value)}
                        placeholder="2022-12-31"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium" htmlFor="hist-resample">
                        Resample to (optional)
                      </label>
                      <Input
                        id="hist-resample"
                        value={histResample}
                        onChange={(e) => setHistResample(e.target.value)}
                        placeholder="H, D, W"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium" htmlFor="max-rows">
                        Max rows (preview)
                      </label>
                      <Input
                        id="max-rows"
                        type="number"
                        value={maxRows}
                        onChange={(e) => setMaxRows(Number(e.target.value) || 50)}
                        onBlur={() => setMaxRows((v) => Math.max(50, Math.min(10000, v || 50)))}
                        min={50}
                        max={10000}
                      />
                    </div>
                  </div>

                  {errorHistorical && (
                    <Alert variant="destructive">
                      <AlertTitle>Historical load failed</AlertTitle>
                      <AlertDescription>{errorHistorical}</AlertDescription>
                    </Alert>
                  )}

                  <div className="flex flex-wrap items-center gap-3">
                    <Button
                      onClick={handleLoadHistorical}
                      disabled={loadingHistorical || columnState !== "ok"}
                    >
                      {loadingHistorical && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />}
                      {loadingHistorical ? "Loading..." : "Load historical data"}
                    </Button>
                    <Button variant="secondary" onClick={() => setTab("forecast")}>
                      Continue to forecast →
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Preview</CardTitle>
                  <CardDescription>
                    Last {historical?.preview?.length ?? 0} rows (capped at {maxRows}).
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {historical && (
                    <div className="flex flex-wrap items-center gap-3 text-sm">
                      <Badge variant="outline">{historical.table}</Badge>
                      <Badge variant="outline">{formatNumber(historical.row_count)} rows</Badge>
                      <Badge variant="outline">{historical.columns.length} columns</Badge>
                    </div>
                  )}
                  <DataPreviewTable
                    columns={historical?.columns || []}
                    rows={historical?.preview || []}
                    emptyLabel="No data loaded yet. Press Load historical data to populate this preview."
                  />
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="forecast" className="mt-6">
            <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
              <Card>
                <CardHeader>
                  <CardTitle>Forecast settings</CardTitle>
                  <CardDescription>
                    Configure Prophet, choose regressors, and kick off a run.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium" htmlFor="freq-select">
                        Frequency
                      </label>
                      <Select value={freq} onValueChange={setFreq}>
                        <SelectTrigger id="freq-select">
                          <SelectValue placeholder="Select a frequency" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="D">Daily</SelectItem>
                          <SelectItem value="W">Weekly</SelectItem>
                          <SelectItem value="M">Monthly</SelectItem>
                          <SelectItem value="H">Hourly</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium" htmlFor="horizon-input">
                        Horizon
                      </label>
                      <Input
                        id="horizon-input"
                        value={horizon}
                        onChange={(e) => setHorizon(e.target.value)}
                        placeholder="e.g., 90D"
                        aria-invalid={!!horizonError}
                        aria-describedby="horizon-help"
                      />
                      <p id="horizon-help" className="text-xs text-muted-foreground">
                        {horizonError ? (
                          <span className="text-destructive">{horizonError}</span>
                        ) : horizonPreview ? (
                          <>→ {horizonPreview}</>
                        ) : (
                          <>Examples: 90D, 12M, 2Y, 365 days</>
                        )}
                      </p>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium" htmlFor="window-input">
                        Training window
                      </label>
                      <Input
                        id="window-input"
                        value={trainingWindow}
                        onChange={(e) => setTrainingWindow(e.target.value)}
                        placeholder="e.g., 730 days"
                        aria-invalid={!!trainingWindowError}
                        aria-describedby="window-help"
                      />
                      <p id="window-help" className="text-xs text-muted-foreground">
                        {trainingWindowError ? (
                          <span className="text-destructive">{trainingWindowError}</span>
                        ) : trainingWindowPreview ? (
                          <>→ {trainingWindowPreview}</>
                        ) : (
                          <>Examples: 730 days, 2Y, 18M</>
                        )}
                      </p>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium" htmlFor="resample-input">
                        Resample to (optional)
                      </label>
                      <Input
                        id="resample-input"
                        value={resample}
                        onChange={(e) => setResample(e.target.value)}
                        placeholder="H, D, W, 15min"
                      />
                    </div>
                    <div className="space-y-2 md:col-span-2">
                      <label className="text-sm font-medium" htmlFor="series-input">
                        Series name
                      </label>
                      <Input
                        id="series-input"
                        value={seriesName}
                        onChange={(e) => setSeriesName(e.target.value)}
                        placeholder="call_volume"
                      />
                    </div>
                  </div>

                  <div className="rounded-md border bg-muted/40 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <p className="text-sm font-semibold">Regressors</p>
                        <p className="text-xs text-muted-foreground">
                          Detected from your table (excluding the timestamp and value columns).
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSelectedRegressors([...availableRegressors])}
                          disabled={availableRegressors.length === 0}
                        >
                          Select all
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSelectedRegressors([])}
                          disabled={selectedRegressors.length === 0}
                        >
                          Clear
                        </Button>
                      </div>
                    </div>
                    <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-3">
                      {availableRegressors.length === 0 && (
                        <p className="col-span-full text-sm text-muted-foreground">
                          No extra columns detected. Adjust the table or add regressor columns to your data source.
                        </p>
                      )}
                      {availableRegressors.map((regressor) => {
                        const id = `regressor-${regressor}`;
                        return (
                          <label
                            key={regressor}
                            htmlFor={id}
                            className="flex items-center gap-2 rounded-sm border bg-card px-3 py-2 text-sm shadow-sm"
                          >
                            <Checkbox
                              id={id}
                              checked={selectedRegressors.includes(regressor)}
                              onCheckedChange={() => toggleRegressor(regressor)}
                            />
                            <span className="truncate">{regressor}</span>
                          </label>
                        );
                      })}
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
                    <label className="flex items-center gap-2 text-sm" htmlFor="auto-tune">
                      <Checkbox
                        id="auto-tune"
                        checked={autoTune}
                        onCheckedChange={(checked) => setAutoTune(Boolean(checked))}
                      />
                      <span>Auto-tune hyperparameters</span>
                    </label>
                    <label className="flex items-center gap-2 text-sm" htmlFor="remove-outliers">
                      <Checkbox
                        id="remove-outliers"
                        checked={removeOutliers}
                        onCheckedChange={(checked) => setRemoveOutliers(Boolean(checked))}
                      />
                      <span>Remove IQR outliers (preserves holidays)</span>
                    </label>
                    {autoTune && (
                      <Badge variant="outline" className="border-amber-400 text-amber-600">
                        Adds 1-10 minutes for cross-validation
                      </Badge>
                    )}
                  </div>

                  {errorForecast && (
                    <Alert variant="destructive">
                      <AlertTitle>Forecast failed</AlertTitle>
                      <AlertDescription>{errorForecast}</AlertDescription>
                    </Alert>
                  )}

                  <div className="flex flex-wrap items-center gap-3">
                    <Button onClick={handleForecast} disabled={!canForecast}>
                      {loadingForecast && (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                      )}
                      {loadingForecast ? "Running forecast..." : "Run forecast"}
                    </Button>
                    {loadingForecast && (
                      <p className="text-sm text-muted-foreground" aria-live="polite">
                        Elapsed: {forecastElapsed.toFixed(1)}s
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Results</CardTitle>
                  <CardDescription>Tail of the forecast plus the rendered plots.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {loadingForecast && (
                    <div className="space-y-3" aria-busy="true">
                      <div className="h-16 animate-pulse rounded-md bg-muted" />
                      <div className="h-40 animate-pulse rounded-md bg-muted" />
                      <div className="grid grid-cols-2 gap-3">
                        <div className="h-48 animate-pulse rounded-md bg-muted" />
                        <div className="h-48 animate-pulse rounded-md bg-muted" />
                      </div>
                    </div>
                  )}

                  {!loadingForecast && !forecastResult && (
                    <p className="text-sm text-muted-foreground">
                      Run a forecast to see results. Use <code>{DEFAULT_TABLE}</code> ({DEFAULT_TS}/{DEFAULT_Y})
                      to try the seeded dummy data.
                    </p>
                  )}

                  {!loadingForecast && forecastResult && (
                    <>
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant="outline">{forecastResult.series_name}</Badge>
                        <Badge variant="outline">
                          Rows: {formatNumber(forecastResult.row_count)}
                        </Badge>
                        {typeof forecastResult.training_rows === "number" && (
                          <Badge variant="outline">
                            Trained on {formatNumber(forecastResult.training_rows)} rows
                          </Badge>
                        )}
                        {typeof forecastResult.outliers_removed === "number" &&
                          forecastResult.outliers_removed > 0 && (
                            <Badge variant="outline" className="border-amber-400 text-amber-600">
                              Removed {forecastResult.outliers_removed} outliers
                            </Badge>
                          )}
                        {selectedRegressors.length > 0 && (
                          <Badge variant="outline">
                            {selectedRegressors.length} regressor(s)
                          </Badge>
                        )}
                      </div>

                      <DataPreviewTable
                        columns={forecastResult.preview[0] ? Object.keys(forecastResult.preview[0]) : []}
                        rows={forecastResult.preview}
                        emptyLabel="No forecast rows to show."
                      />

                      <div className="grid gap-4">
                        <PlotPanel
                          title="Forecast"
                          src={forecastResult.forecast_plot}
                          alt="Prophet forecast plot showing historical points in black, forecast in blue, and any actuals in green."
                        />
                        <PlotPanel
                          title="Components"
                          src={forecastResult.components_plot}
                          alt="Decomposition of the forecast into trend, weekly, and yearly components."
                        />
                        <PlotPanel
                          title="Day-of-week breakdown"
                          src={forecastResult.day_breakdown_plot}
                          alt="Stacked bar chart of answered (green) and abandoned (red) calls by day of week."
                        />
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
