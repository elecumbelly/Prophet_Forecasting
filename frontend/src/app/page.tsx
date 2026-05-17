/* eslint-disable @next/next/no-img-element */
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { Loader2 } from "lucide-react";

import { CwfLogo } from "@/components/CwfLogo";
import { RingComposition } from "@/components/RingComposition";
import { ThemeToggle } from "@/components/ThemeToggle";
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
type PhaseId = "history" | "forecast";

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

function formatNumber(value?: number) {
  if (typeof value !== "number") return "–";
  return value.toLocaleString();
}

function formatCompact(value: number): string {
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}

function useElapsed(running: boolean) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!running) return;
    const t0 = performance.now();
    // Reset is scheduled (not synchronous) to satisfy react-hooks/set-state-
    // in-effect, then immediately superseded by the interval tick.
    const initial = window.setTimeout(() => setElapsed(0), 0);
    const id = window.setInterval(() => {
      setElapsed((performance.now() - t0) / 1000);
    }, 250);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(id);
    };
  }, [running]);

  return elapsed;
}

/* ─── shared brutalist primitives ─────────────────────────────────────── */

function HairlineField({
  id,
  label,
  value,
  onChange,
  placeholder,
  hint,
  error,
  invalid,
  type = "text",
  ariaDescribedBy,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  hint?: string;
  error?: string | null;
  invalid?: boolean;
  type?: string;
  ariaDescribedBy?: string;
}) {
  const showError = !!error;
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="cwf-eyebrow text-muted-foreground">
        {label}
      </label>
      <input
        id={id}
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        aria-invalid={invalid || showError ? true : undefined}
        aria-describedby={ariaDescribedBy}
        className={`w-full border-0 border-b ${
          showError ? "border-destructive" : "border-border"
        } bg-transparent px-0 py-2 text-base font-medium tabular-nums focus:outline-none focus:border-primary focus:border-b-2`}
      />
      {(hint || showError) && (
        <p
          id={ariaDescribedBy}
          className={`cwf-eyebrow ${showError ? "text-destructive" : "text-muted-foreground"}`}
        >
          {showError ? error : hint}
        </p>
      )}
    </div>
  );
}

function HairlineSelect({
  id,
  label,
  value,
  onChange,
  options,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="cwf-eyebrow text-muted-foreground">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full appearance-none border-0 border-b border-border bg-transparent px-0 py-2 text-base font-medium focus:outline-none focus:border-primary focus:border-b-2"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

function Tag({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`cwf-eyebrow border border-border px-3 py-2 transition-colors ${
        active
          ? "bg-primary text-primary-foreground border-primary"
          : "bg-transparent text-foreground hover:bg-foreground hover:text-background"
      }`}
    >
      {children}
    </button>
  );
}

function PrimaryButton({
  onClick,
  disabled,
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="inline-flex items-center gap-2 bg-primary px-6 py-3 text-base font-bold text-primary-foreground hover:bg-foreground hover:text-background disabled:opacity-40 disabled:hover:bg-primary disabled:hover:text-primary-foreground"
    >
      {children}
    </button>
  );
}

function TextLink({
  onClick,
  children,
}: {
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="cwf-eyebrow underline-offset-4 hover:underline hover:decoration-primary focus-visible:underline"
    >
      {children}
    </button>
  );
}

/* ─── data preview ────────────────────────────────────────────────────── */

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
    return <p className="cwf-eyebrow text-muted-foreground">{emptyLabel}</p>;
  }

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto border-t border-b border-border">
        <table className="w-full text-sm tabular-nums">
          <thead>
            <tr className="bg-foreground text-background">
              {displayedColumns.map((col) => (
                <th
                  key={col}
                  className="cwf-eyebrow whitespace-nowrap px-3 py-2 text-left"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={idx} className="border-t border-border">
                {displayedColumns.map((col) => {
                  const v = row[col];
                  const isNumber = typeof v === "number";
                  return (
                    <td
                      key={col}
                      className={`whitespace-nowrap px-3 py-2 ${
                        isNumber ? "text-right" : ""
                      }`}
                    >
                      {String(v ?? "").slice(0, 120)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {truncated && (
        <button
          type="button"
          className="cwf-eyebrow underline-offset-4 underline decoration-muted-foreground hover:decoration-primary"
          onClick={() => setShowAll(true)}
        >
          Showing {maxColumns} of {columns.length} columns — show all
        </button>
      )}
    </div>
  );
}

/* ─── plot panel ──────────────────────────────────────────────────────── */

function PlotPanel({
  title,
  caption,
  src,
  alt,
}: {
  title: string;
  caption?: string;
  src?: string;
  alt: string;
}) {
  if (!src) return null;
  return (
    <figure className="border border-border bg-card">
      <figcaption className="flex items-center justify-between border-b border-border px-4 py-2">
        <span className="cwf-eyebrow">{title}</span>
        <a
          href={src}
          download={`${title.toLowerCase().replace(/\s+/g, "-")}.png`}
          className="cwf-eyebrow hover:underline hover:decoration-primary underline-offset-4"
        >
          Download
        </a>
      </figcaption>
      <a
        href={src}
        target="_blank"
        rel="noreferrer noopener"
        aria-label={`Open ${title} full size`}
        className="block"
      >
        <img src={src} alt={alt} className="w-full bg-background" loading="lazy" />
      </a>
      {caption && (
        <p className="cwf-eyebrow border-t border-border px-4 py-2 text-muted-foreground">
          {caption}
        </p>
      )}
    </figure>
  );
}

/* ─── loading skeleton ────────────────────────────────────────────────── */

function LineDrawSkeleton() {
  return (
    <div className="space-y-3" aria-busy="true">
      <div className="h-px bg-border cwf-line" />
      <div className="h-12 border border-border relative overflow-hidden">
        <div className="absolute inset-y-0 left-0 w-full bg-foreground/10 cwf-line" />
      </div>
      <div className="h-px bg-border cwf-line" />
      <div className="grid grid-cols-2 gap-3">
        <div className="h-40 border border-border relative overflow-hidden">
          <div className="absolute inset-y-0 left-0 w-full bg-foreground/10 cwf-line" />
        </div>
        <div className="h-40 border border-border relative overflow-hidden">
          <div className="absolute inset-y-0 left-0 w-full bg-foreground/10 cwf-line" />
        </div>
      </div>
    </div>
  );
}

/* ─── page ────────────────────────────────────────────────────────────── */

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

  const [phase, setPhase] = useState<PhaseId>("history");

  const forecastElapsed = useElapsed(loadingForecast);

  const availableRegressors = useMemo(
    () => columns.filter((c) => c !== tsColumn && c !== yColumn),
    [columns, tsColumn, yColumn],
  );

  const tsValid = columnState === "ok" && columns.includes(tsColumn);
  const yValid = columnState === "ok" && columns.includes(yColumn);

  const horizonPreview = previewDuration(horizon);
  const trainingWindowPreview = previewDuration(trainingWindow);
  const horizonError = horizon && !horizonPreview ? "Try 90D, 12M, 2Y" : null;
  const trainingWindowError = trainingWindow && !trainingWindowPreview ? "Try 730 days, 2Y" : null;

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
        setSelectedRegressors((prev) => prev.filter((c) => res.columns?.includes(c)));
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
    setPhase("history");
  };

  /* derived hero number for the ring composition centre */
  const lastForecastValue = (() => {
    if (!forecastResult?.preview?.length) return null;
    const row = forecastResult.preview[forecastResult.preview.length - 1];
    const yhat = row["yhat"];
    if (typeof yhat !== "number") return null;
    return yhat;
  })();

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* ─── header ───────────────────────────────────────────────────── */}
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-6 px-10 py-6">
          <div className="flex items-center gap-4">
            <CwfLogo size={56} />
            <div>
              <p className="cwf-eyebrow text-muted-foreground">
                Collapsing Wave Functions
              </p>
              <h1 className="text-2xl font-bold leading-none tracking-tight">
                PROPHET / FORECASTING
              </h1>
            </div>
          </div>
          <nav className="flex items-center gap-6">
            {process.env.NODE_ENV !== "production" && (
              <span className="cwf-eyebrow text-muted-foreground">
                API · {API_BASE.replace(/^https?:\/\//, "")}
              </span>
            )}
            <a
              href="https://facebook.github.io/prophet/"
              target="_blank"
              rel="noreferrer noopener"
              className="cwf-eyebrow underline-offset-4 hover:underline hover:decoration-primary"
            >
              Docs ▸
            </a>
            <TextLink onClick={handleRestartSession}>Reset session</TextLink>
            <ThemeToggle />
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-10 pb-24 pt-16">
        {/* ─── stat strip ─────────────────────────────────────────────── */}
        <section className="grid grid-cols-1 gap-y-10 border-y border-border py-10 md:grid-cols-3 md:gap-y-0">
          <Stat
            eyebrow="Columns detected"
            value={columns.length ? String(columns.length) : "–"}
            footnote={
              columnState === "loading"
                ? "Probing API…"
                : columnState === "ok"
                  ? "Source responding"
                  : columnState === "missing"
                    ? "Table not found"
                    : "API unreachable"
            }
            accent={columnState === "ok"}
          />
          <Stat
            eyebrow="Regressors available"
            value={availableRegressors.length ? String(availableRegressors.length) : "–"}
            footnote={`${selectedRegressors.length} selected`}
            divider
          />
          <Stat
            eyebrow="Default frequency"
            value={freq.toUpperCase()}
            footnote={`Horizon ${horizonPreview ?? horizon}`}
            divider
          />
        </section>

        {columnError && columnState !== "ok" && (
          <p className="mt-6 border-l-4 border-destructive bg-destructive/5 px-4 py-3 text-sm">
            <span className="cwf-eyebrow text-destructive">Source error</span>
            <br />
            <span className="text-foreground">{columnError}</span>
          </p>
        )}

        {/* ─── phase rail ─────────────────────────────────────────────── */}
        <section className="mt-20 space-y-16">
          <PhaseBlock
            number="01"
            title="Inspect history"
            description="Pull a preview of the source table and verify the column mapping before forecasting."
            active={phase === "history"}
            onActivate={() => setPhase("history")}
          >
            <div className="grid gap-x-10 gap-y-10 md:grid-cols-2">
              {/* source block */}
              <div className="space-y-6">
                <h3 className="cwf-eyebrow text-muted-foreground">Source</h3>
                <div className="grid grid-cols-1 gap-6">
                  <HairlineField
                    id="table-input"
                    label="Table"
                    value={tableName}
                    onChange={setTableName}
                  />
                  <HairlineField
                    id="ts-input"
                    label="Timestamp column"
                    value={tsColumn}
                    onChange={setTsColumn}
                    invalid={columnState === "ok" && !tsValid}
                    error={
                      columnState === "ok" && !tsValid
                        ? `Not a column in ${tableName}`
                        : undefined
                    }
                  />
                  <HairlineField
                    id="y-input"
                    label="Value column"
                    value={yColumn}
                    onChange={setYColumn}
                    invalid={columnState === "ok" && !yValid}
                    error={
                      columnState === "ok" && !yValid
                        ? `Not a column in ${tableName}`
                        : undefined
                    }
                  />
                </div>
              </div>

              {/* preview controls */}
              <div className="space-y-6 border-t border-border pt-10 md:border-t-0 md:border-l md:pl-10 md:pt-0">
                <h3 className="cwf-eyebrow text-muted-foreground">Preview window</h3>
                <div className="grid grid-cols-2 gap-6">
                  <HairlineField
                    id="hist-start"
                    label="Start"
                    value={histStart}
                    onChange={setHistStart}
                    placeholder="2021-01-01"
                  />
                  <HairlineField
                    id="hist-end"
                    label="End"
                    value={histEnd}
                    onChange={setHistEnd}
                    placeholder="2022-12-31"
                  />
                  <HairlineField
                    id="hist-resample"
                    label="Resample to"
                    value={histResample}
                    onChange={setHistResample}
                    placeholder="H, D, W"
                  />
                  <HairlineField
                    id="max-rows"
                    label="Max rows"
                    value={String(maxRows)}
                    onChange={(v) => setMaxRows(Number(v) || 50)}
                    type="number"
                  />
                </div>
                {errorHistorical && (
                  <p className="border-l-4 border-destructive bg-destructive/5 px-4 py-3 text-sm">
                    <span className="cwf-eyebrow text-destructive">Load failed</span>
                    <br />
                    {errorHistorical}
                  </p>
                )}
                <div className="flex flex-wrap items-center gap-6">
                  <PrimaryButton
                    onClick={handleLoadHistorical}
                    disabled={loadingHistorical || columnState !== "ok"}
                  >
                    {loadingHistorical && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
                    {loadingHistorical ? "Loading…" : "Load preview"}
                  </PrimaryButton>
                  <TextLink onClick={() => setPhase("forecast")}>
                    Continue to forecast ▸
                  </TextLink>
                </div>
              </div>
            </div>

            {historical && (
              <div className="mt-14 space-y-6">
                <div className="flex flex-wrap items-baseline justify-between gap-4 border-t border-border pt-6">
                  <h4 className="cwf-eyebrow">Preview · {historical.table}</h4>
                  <div className="flex gap-6">
                    <span className="cwf-eyebrow text-muted-foreground">
                      {formatNumber(historical.row_count)} rows
                    </span>
                    <span className="cwf-eyebrow text-muted-foreground">
                      {historical.columns.length} columns
                    </span>
                  </div>
                </div>
                <DataPreviewTable
                  columns={historical.columns}
                  rows={historical.preview}
                  emptyLabel="No rows returned."
                />
              </div>
            )}
          </PhaseBlock>

          <PhaseBlock
            number="02"
            title="Run forecast"
            description="Configure Prophet, choose regressors, and emit a forecast. Auto-tune adds 1–10 minutes for cross-validation."
            active={phase === "forecast"}
            onActivate={() => setPhase("forecast")}
          >
            <div className="grid gap-x-10 gap-y-10 md:grid-cols-[1fr_1fr]">
              {/* forecast settings */}
              <div className="space-y-8">
                <div className="grid grid-cols-2 gap-6">
                  <HairlineSelect
                    id="freq-select"
                    label="Frequency"
                    value={freq}
                    onChange={setFreq}
                    options={[
                      { value: "D", label: "Daily" },
                      { value: "W", label: "Weekly" },
                      { value: "M", label: "Monthly" },
                      { value: "H", label: "Hourly" },
                    ]}
                  />
                  <HairlineField
                    id="horizon-input"
                    label="Horizon"
                    value={horizon}
                    onChange={setHorizon}
                    placeholder="90D"
                    hint={horizonPreview ? `→ ${horizonPreview}` : "90D · 12M · 2Y"}
                    error={horizonError}
                    invalid={!!horizonError}
                    ariaDescribedBy="horizon-help"
                  />
                  <HairlineField
                    id="window-input"
                    label="Training window"
                    value={trainingWindow}
                    onChange={setTrainingWindow}
                    placeholder="730 days"
                    hint={trainingWindowPreview ? `→ ${trainingWindowPreview}` : "730 days · 2Y"}
                    error={trainingWindowError}
                    invalid={!!trainingWindowError}
                    ariaDescribedBy="window-help"
                  />
                  <HairlineField
                    id="resample-input"
                    label="Resample to (optional)"
                    value={resample}
                    onChange={setResample}
                    placeholder="H · D · W"
                  />
                  <HairlineField
                    id="series-input"
                    label="Series name"
                    value={seriesName}
                    onChange={setSeriesName}
                    placeholder="call_volume"
                  />
                </div>

                <div className="space-y-3 border-t border-border pt-6">
                  <div className="flex items-baseline justify-between">
                    <h4 className="cwf-eyebrow">Regressors</h4>
                    <div className="flex gap-4">
                      <TextLink onClick={() => setSelectedRegressors([...availableRegressors])}>
                        Select all
                      </TextLink>
                      <TextLink onClick={() => setSelectedRegressors([])}>Clear</TextLink>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {availableRegressors.length === 0 && (
                      <p className="cwf-eyebrow text-muted-foreground">
                        No additional columns detected on {tableName}.
                      </p>
                    )}
                    {availableRegressors.map((r) => (
                      <Tag
                        key={r}
                        active={selectedRegressors.includes(r)}
                        onClick={() => toggleRegressor(r)}
                      >
                        {r}
                      </Tag>
                    ))}
                  </div>
                </div>

                <div className="space-y-4 border-t border-border pt-6">
                  <h4 className="cwf-eyebrow">Switches</h4>
                  <div className="flex flex-wrap gap-3">
                    <Tag active={autoTune} onClick={() => setAutoTune((v) => !v)}>
                      {autoTune ? "Auto-tune ON" : "Auto-tune OFF"}
                    </Tag>
                    <Tag active={removeOutliers} onClick={() => setRemoveOutliers((v) => !v)}>
                      {removeOutliers ? "IQR filter ON" : "IQR filter OFF"}
                    </Tag>
                  </div>
                  {autoTune && (
                    <p className="cwf-eyebrow text-muted-foreground">
                      → Auto-tune runs cross-validation before training (1–10 minutes).
                    </p>
                  )}
                </div>

                {errorForecast && (
                  <p className="border-l-4 border-destructive bg-destructive/5 px-4 py-3 text-sm">
                    <span className="cwf-eyebrow text-destructive">Forecast failed</span>
                    <br />
                    {errorForecast}
                  </p>
                )}

                <div className="flex flex-wrap items-center gap-6 border-t border-border pt-6">
                  <PrimaryButton onClick={handleForecast} disabled={!canForecast}>
                    {loadingForecast && (
                      <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                    )}
                    {loadingForecast ? "Running forecast…" : "Run forecast ▸"}
                  </PrimaryButton>
                  {loadingForecast && (
                    <span className="cwf-eyebrow text-muted-foreground" aria-live="polite">
                      Elapsed · {forecastElapsed.toFixed(1)}s
                    </span>
                  )}
                </div>
              </div>

              {/* results */}
              <div className="border-t border-border pt-10 md:border-t-0 md:border-l md:pl-10 md:pt-0">
                <h3 className="cwf-eyebrow text-muted-foreground">Results</h3>

                {loadingForecast && (
                  <div className="mt-6">
                    <LineDrawSkeleton />
                  </div>
                )}

                {!loadingForecast && !forecastResult && (
                  <div className="mt-6 space-y-6">
                    <p className="text-sm text-muted-foreground">
                      Emit a forecast to populate this column. Default source is{" "}
                      <code className="bg-muted px-1 py-0.5">{DEFAULT_TABLE}</code> with{" "}
                      <code className="bg-muted px-1 py-0.5">{DEFAULT_TS}</code> /{" "}
                      <code className="bg-muted px-1 py-0.5">{DEFAULT_Y}</code>.
                    </p>
                    <div className="flex justify-center pt-4">
                      <CwfLogo size={160} className="opacity-30" />
                    </div>
                  </div>
                )}

                {!loadingForecast && forecastResult && (
                  <div className="mt-6 space-y-8">
                    <RingComposition
                      size={260}
                      centerValue={
                        lastForecastValue !== null ? formatCompact(lastForecastValue) : "—"
                      }
                      centerLabel="Forecast · last period (compact)"
                      tags={[
                        { label: "Series", value: forecastResult.series_name },
                        { label: "Rows", value: formatNumber(forecastResult.row_count) },
                        {
                          label: "Trained",
                          value: forecastResult.training_rows
                            ? formatNumber(forecastResult.training_rows)
                            : "–",
                        },
                        {
                          label: "Outliers",
                          value: forecastResult.outliers_removed ?? 0,
                        },
                      ]}
                    />
                  </div>
                )}
              </div>
            </div>

            {/* full-width result strip below the form/results pair */}
            {!loadingForecast && forecastResult && (
              <div className="mt-14 space-y-8 border-t border-border pt-10">
                <DataPreviewTable
                  columns={
                    forecastResult.preview[0] ? Object.keys(forecastResult.preview[0]) : []
                  }
                  rows={forecastResult.preview}
                  emptyLabel="No forecast rows to show."
                />
                <div className="grid gap-6 md:grid-cols-2">
                  <PlotPanel
                    title="Forecast"
                    caption="Black = historical · Blue = forecast · Green = actual overlay"
                    src={forecastResult.forecast_plot}
                    alt="Prophet forecast plot showing historical points in black, forecast in blue, and any actuals in green."
                  />
                  <PlotPanel
                    title="Components"
                    caption="Trend · weekly · yearly decomposition"
                    src={forecastResult.components_plot}
                    alt="Decomposition of the forecast into trend, weekly, and yearly components."
                  />
                  {forecastResult.day_breakdown_plot && (
                    <div className="md:col-span-2">
                      <PlotPanel
                        title="Day-of-week breakdown"
                        caption="Green = answered · Red = abandoned"
                        src={forecastResult.day_breakdown_plot}
                        alt="Stacked bar chart of answered and abandoned calls by day of week."
                      />
                    </div>
                  )}
                </div>
              </div>
            )}
          </PhaseBlock>
        </section>

        <footer className="mt-24 flex items-center justify-between border-t border-border pt-6">
          <span className="cwf-eyebrow text-muted-foreground">
            Prophet · Forecasting Console
          </span>
          <span className="cwf-eyebrow text-muted-foreground">
            Collapsing Wave Functions
          </span>
        </footer>
      </main>
    </div>
  );
}

/* ─── header section components (kept inline for layout cohesion) ────── */

function Stat({
  eyebrow,
  value,
  footnote,
  divider,
  accent,
}: {
  eyebrow: string;
  value: string;
  footnote: string;
  divider?: boolean;
  accent?: boolean;
}) {
  return (
    <div
      className={`space-y-3 ${
        divider ? "md:border-l md:border-border md:pl-10" : ""
      }`}
    >
      <p className="cwf-eyebrow text-muted-foreground">{eyebrow}</p>
      <p
        className={`text-5xl font-bold tabular-nums leading-none tracking-tight ${
          accent ? "text-primary-foreground" : ""
        }`}
        style={accent ? { color: "var(--primary)" } : undefined}
      >
        {value}
      </p>
      <p className="cwf-eyebrow text-muted-foreground">{footnote}</p>
    </div>
  );
}

function PhaseBlock({
  number,
  title,
  description,
  active,
  onActivate,
  children,
}: {
  number: string;
  title: string;
  description: string;
  active: boolean;
  onActivate: () => void;
  children: React.ReactNode;
}) {
  return (
    <article className="grid grid-cols-1 gap-x-10 gap-y-8 md:grid-cols-[120px_1fr]">
      <div className="space-y-2">
        <p
          className="text-5xl font-bold leading-none"
          style={{ color: active ? "var(--primary)" : "var(--muted-foreground)" }}
        >
          {number}
        </p>
        {!active && (
          <button
            type="button"
            onClick={onActivate}
            className="cwf-eyebrow underline-offset-4 hover:underline hover:decoration-primary"
          >
            Expand ▸
          </button>
        )}
      </div>
      <div className="space-y-2">
        <h2 className="text-3xl font-bold leading-tight">{title}</h2>
        <p className="max-w-xl text-sm text-muted-foreground">{description}</p>
      </div>
      {active && <div className="col-span-1 md:col-start-2">{children}</div>}
    </article>
  );
}
