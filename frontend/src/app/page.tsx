/* eslint-disable @next/next/no-img-element */
"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

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
import { Textarea } from "@/components/ui/textarea";
import {
  ColumnsResponse,
  ForecastResponse,
  HistoricalResponse,
  fetchColumns,
  fetchForecast,
  fetchHistoricalData,
} from "@/lib/api";

const DEFAULT_TABLE = "call_center_metrics";
const BUILD_TAG = "LYRA-V2";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:5001";

function formatNumber(value?: number) {
  if (typeof value !== "number") return "";
  return value.toLocaleString();
}

function DataPreviewTable({
  rows,
  columns,
  emptyLabel,
}: {
  rows: Record<string, unknown>[];
  columns: string[];
  emptyLabel: string;
}) {
  const displayedColumns = columns.slice(0, 6);

  if (!rows.length) {
    return (
      <p className="text-sm text-muted-foreground">
        {emptyLabel}
      </p>
    );
  }

  return (
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
  );
}

export default function Home() {
  const [tableName, setTableName] = useState(DEFAULT_TABLE);
  const [tsColumn, setTsColumn] = useState("ts");
  const [yColumn, setYColumn] = useState("y");
  const [columns, setColumns] = useState<string[]>([]);

  const [freq, setFreq] = useState("D");
  const [horizon, setHorizon] = useState("90D");
  const [trainingWindow, setTrainingWindow] = useState("730 days");
  const [resample, setResample] = useState("");
  const [seriesName, setSeriesName] = useState("call_volume");
  const [autoTune, setAutoTune] = useState(false);
  const [selectedRegressors, setSelectedRegressors] = useState<string[]>([]);

  const [histStart, setHistStart] = useState("");
  const [histEnd, setHistEnd] = useState("");
  const [histResample, setHistResample] = useState("");
  const [maxRows, setMaxRows] = useState(200);

  const [historical, setHistorical] = useState<HistoricalResponse | null>(
    null
  );
  const [forecastResult, setForecastResult] =
    useState<ForecastResponse | null>(null);

  const [loadingForecast, setLoadingForecast] = useState(false);
  const [loadingHistorical, setLoadingHistorical] = useState(false);
  const [errorForecast, setErrorForecast] = useState<string | null>(null);
  const [errorHistorical, setErrorHistorical] = useState<string | null>(null);

  const availableRegressors = useMemo(
    () => columns.filter((col) => col !== tsColumn && col !== yColumn),
    [columns, tsColumn, yColumn]
  );

  useEffect(() => {
    async function loadColumns() {
      try {
        const res: ColumnsResponse = await fetchColumns(tableName);
        if (res.columns) {
          setColumns(res.columns);
          setSelectedRegressors((prev) =>
            prev.filter((col) => res.columns?.includes(col))
          );
        } else if (res.error) {
          setColumns([]);
        }
      } catch (err) {
        console.error(err);
        setColumns([]);
      }
    }

    loadColumns();
  }, [tableName]);

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
      setErrorHistorical(
        err instanceof Error ? err.message : "Failed to load historical data"
      );
      setHistorical(null);
    } finally {
      setLoadingHistorical(false);
    }
  };

  const toggleRegressor = (value: string) => {
    setSelectedRegressors((prev) =>
      prev.includes(value)
        ? prev.filter((r) => r !== value)
        : [...prev, value]
    );
  };

  return (
    <div className="min-h-screen bg-[#f7f4ef] text-slate-900">
      <main className="mx-auto max-w-6xl px-6 pb-16 pt-10">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-sm bg-gradient-to-br from-primary to-emerald-400 text-lg font-semibold text-white">
              PF
            </div>
            <div>
              <p className="text-xs font-mono uppercase tracking-[0.4em] text-slate-500">
                Prophet Forecasting
              </p>
              <h1 className="text-2xl font-mono font-semibold uppercase tracking-[0.12em] text-slate-900">
                Lyra control room
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge className="bg-black/5 text-slate-700" variant="outline">
              API: {API_BASE}
            </Badge>
            <Badge className="bg-black text-white" variant="outline">
              {BUILD_TAG}
            </Badge>
            <Button asChild variant="secondary" className="border-black/20">
              <Link href="https://facebook.github.io/prophet/" target="_blank">
                Prophet Docs
              </Link>
            </Button>
            <Button asChild className="bg-primary text-primary-foreground">
              <Link href="/" scroll={false}>
                Restart Session
              </Link>
            </Button>
          </div>
        </header>

        <section className="mt-8 grid gap-4 md:grid-cols-3">
          <Card className="border-black/10 bg-white">
            <CardHeader>
              <CardTitle>Source</CardTitle>
              <CardDescription>
                Point the UI at any PostgreSQL table powering the service.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-3 gap-3">
                <div className="col-span-1">
                  <p className="text-xs uppercase text-slate-500">Table</p>
                  <Input
                    value={tableName}
                    onChange={(e) => setTableName(e.target.value)}
                    className="mt-1 border-black/10 bg-white"
                  />
                </div>
                <div className="col-span-1">
                  <p className="text-xs uppercase text-slate-500">Timestamp</p>
                  <Input
                    value={tsColumn}
                    onChange={(e) => setTsColumn(e.target.value)}
                    className="mt-1 border-black/10 bg-white"
                  />
                </div>
                <div className="col-span-1">
                  <p className="text-xs uppercase text-slate-500">Value</p>
                  <Input
                    value={yColumn}
                    onChange={(e) => setYColumn(e.target.value)}
                    className="mt-1 border-black/10 bg-white"
                  />
                </div>
              </div>
              <p className="text-xs text-slate-500">
                Columns auto-refresh when you change the table name.
              </p>
            </CardContent>
          </Card>
          <Card className="border-black/10 bg-white">
            <CardHeader>
              <CardTitle>Data health</CardTitle>
              <CardDescription>
                Quick snapshot of what the API returns for this source.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-3 text-sm text-slate-900">
              <div>
                <p className="text-xs uppercase text-slate-500">Columns</p>
                <p className="text-lg font-semibold">{columns.length || "–"}</p>
              </div>
              <div>
                <p className="text-xs uppercase text-slate-500">Regressors</p>
                <p className="text-lg font-semibold">
                  {availableRegressors.length || "–"}
                </p>
              </div>
              <div className="col-span-2 text-xs text-slate-500">
                Need more? Set NEXT_PUBLIC_API_BASE_URL to point at another API.
              </div>
            </CardContent>
          </Card>
          <Card className="border-black/10 bg-white">
            <CardHeader>
              <CardTitle>Runbook</CardTitle>
              <CardDescription>
                1) Load history. 2) Pick regressors. 3) Forecast.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-slate-900">
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                <span>Dummy data lives in {DEFAULT_TABLE} (ts / y)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-amber-500" />
                <span>Forecast horizon accepts natural strings (e.g., 90D)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-primary" />
                <span>Auto-tune will do heavier CV passes</span>
              </div>
            </CardContent>
          </Card>
        </section>

        <Tabs defaultValue="forecast" className="mt-10">
          <TabsList className="grid w-full grid-cols-2 border border-black/10 bg-white text-slate-900 border-black/10">
            <TabsTrigger value="forecast">Forecast</TabsTrigger>
            <TabsTrigger value="historical">Historical Data</TabsTrigger>
          </TabsList>

          <TabsContent value="forecast" className="mt-6">
            <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
              <Card className="bg-white text-slate-900 border-black/10">
                <CardHeader>
                  <CardTitle>Forecast settings</CardTitle>
                  <CardDescription>
                    Configure Prophet, choose regressors, and kick off a run.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">
                        Frequency
                      </label>
                      <Select value={freq} onValueChange={setFreq}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select a frequency" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="D">Daily</SelectItem>
                          <SelectItem value="W">Weekly</SelectItem>
                          <SelectItem value="M">Monthly</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">
                        Horizon
                      </label>
                      <Input
                        value={horizon}
                        onChange={(e) => setHorizon(e.target.value)}
                        placeholder="e.g., 90D"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">
                        Training window
                      </label>
                      <Input
                        value={trainingWindow}
                        onChange={(e) => setTrainingWindow(e.target.value)}
                        placeholder="e.g., 730 days"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">
                        Resample to (optional)
                      </label>
                      <Input
                        value={resample}
                        onChange={(e) => setResample(e.target.value)}
                        placeholder="H, D, W, 15min"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">
                        Series name
                      </label>
                      <Input
                        value={seriesName}
                        onChange={(e) => setSeriesName(e.target.value)}
                        placeholder="call_volume"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">
                        Notes (optional)
                      </label>
                      <Textarea
                        placeholder="Describe this run so teammates know what changed"
                        className="min-h-[80px]"
                      />
                    </div>
                  </div>

                  <div className="rounded-md border bg-neutral-50 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <p className="text-sm font-semibold text-slate-800">
                          Regressors
                        </p>
                        <p className="text-xs text-slate-600">
                          Pulled from your table (excluding ts/y).
                        </p>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setSelectedRegressors([])}
                      >
                        Clear
                      </Button>
                    </div>
                    <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-3">
                      {availableRegressors.length === 0 && (
                        <p className="col-span-full text-sm text-slate-500">
                          No extra columns detected yet. Change the table name
                          or add regressors to your data source.
                        </p>
                      )}
                      {availableRegressors.map((regressor) => (
                        <label
                          key={regressor}
                          className="flex items-center gap-2 rounded-sm border bg-white px-3 py-2 text-sm text-slate-800 shadow-sm"
                        >
                          <Checkbox
                            checked={selectedRegressors.includes(regressor)}
                            onCheckedChange={() => toggleRegressor(regressor)}
                          />
                          <span className="truncate">{regressor}</span>
                        </label>
                      ))}
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-3">
                    <label className="flex items-center gap-2 text-sm text-slate-700">
                      <Checkbox
                        checked={autoTune}
                        onCheckedChange={() => setAutoTune((v) => !v)}
                      />
                      Auto-tune hyperparameters
                    </label>
                    <Badge variant="outline" className="border-primary text-primary">
                      Runs cross-validation before training
                    </Badge>
                  </div>

                  {errorForecast && (
                    <Alert variant="destructive">
                      <AlertTitle>Forecast failed</AlertTitle>
                      <AlertDescription>{errorForecast}</AlertDescription>
                    </Alert>
                  )}

                  <div className="flex flex-wrap items-center gap-3">
                    <Button
                      onClick={handleForecast}
                      disabled={loadingForecast}
                      className="bg-primary text-primary-foreground"
                    >
                      {loadingForecast ? "Running forecast..." : "Run forecast"}
                    </Button>
                    <p className="text-sm text-slate-500">
                      Uses the Flask API at {API_BASE}
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-white text-slate-900 border-black/10">
                <CardHeader>
                  <CardTitle>Results</CardTitle>
                  <CardDescription>
                    Tail of the forecast plus quick visualizations.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {!forecastResult && (
                    <p className="text-sm text-slate-600">
                      Run a forecast to see results. Dummy data: call_center_metrics
                      (ts / y).
                    </p>
                  )}

                  {forecastResult && (
                    <>
                      <div className="flex flex-wrap items-center gap-3">
                        <Badge variant="outline">{forecastResult.series_name}</Badge>
                        <Badge variant="outline" className="border-emerald-300 text-emerald-600">
                          Rows: {formatNumber(forecastResult.row_count)}
                        </Badge>
                        {selectedRegressors.length > 0 && (
                          <Badge variant="outline" className="border-primary text-primary">
                            {selectedRegressors.length} regressor(s)
                          </Badge>
                        )}
                      </div>

                      <DataPreviewTable
                        columns={forecastResult.preview[0] ? Object.keys(forecastResult.preview[0]) : []}
                        rows={forecastResult.preview}
                        emptyLabel="No forecast rows to show."
                      />

                      <div className="grid gap-4 lg:grid-cols-2">
                        {forecastResult.forecast_plot && (
                          <div className="rounded-md border bg-neutral-50 p-3">
                            <p className="mb-2 text-sm font-semibold text-slate-800">
                              Forecast
                            </p>
                            <img
                              src={forecastResult.forecast_plot}
                              alt="Forecast plot"
                              className="w-full rounded-sm border bg-white"
                            />
                          </div>
                        )}
                        {forecastResult.components_plot && (
                          <div className="rounded-md border bg-neutral-50 p-3">
                            <p className="mb-2 text-sm font-semibold text-slate-800">
                              Components
                            </p>
                            <img
                              src={forecastResult.components_plot}
                              alt="Components plot"
                              className="w-full rounded-sm border bg-white"
                            />
                          </div>
                        )}
                        {forecastResult.day_breakdown_plot && (
                          <div className="rounded-md border bg-neutral-50 p-3 lg:col-span-2">
                            <p className="mb-2 text-sm font-semibold text-slate-800">
                              Actuals by day of week
                            </p>
                            <img
                              src={forecastResult.day_breakdown_plot}
                              alt="Day breakdown"
                              className="w-full rounded-sm border bg-white"
                            />
                          </div>
                        )}
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="historical" className="mt-6">
            <div className="grid gap-6 lg:grid-cols-[1fr_0.9fr]">
              <Card className="bg-white text-slate-900 border-black/10">
                <CardHeader>
                  <CardTitle>Historical data</CardTitle>
                  <CardDescription>
                    Pull a preview from the database the Flask app is reading.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">
                        Start (optional)
                      </label>
                      <Input
                        value={histStart}
                        onChange={(e) => setHistStart(e.target.value)}
                        placeholder="2021-01-01"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">
                        End (optional)
                      </label>
                      <Input
                        value={histEnd}
                        onChange={(e) => setHistEnd(e.target.value)}
                        placeholder="2022-12-31"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">
                        Resample to
                      </label>
                      <Input
                        value={histResample}
                        onChange={(e) => setHistResample(e.target.value)}
                        placeholder="H, D, W"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-slate-700">
                        Max rows (preview)
                      </label>
                      <Input
                        type="number"
                        value={maxRows}
                        onChange={(e) => setMaxRows(Number(e.target.value))}
                        min={50}
                        max={1000}
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
                      disabled={loadingHistorical}
                      className="bg-primary text-primary-foreground"
                    >
                      {loadingHistorical ? "Loading..." : "Load historical data"}
                    </Button>
                    <p className="text-sm text-slate-500">
                      Pulls directly from {tableName} via Flask.
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-white text-slate-900 border-black/10">
                <CardHeader>
                  <CardTitle>Preview</CardTitle>
                  <CardDescription>
                    Last {historical?.preview?.length ?? 0} rows (capped at {maxRows}).
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {historical && (
                    <div className="flex flex-wrap items-center gap-3 text-sm text-slate-700">
                      <Badge variant="outline">{historical.table}</Badge>
                      <Badge variant="outline" className="border-emerald-300 text-emerald-700">
                        {formatNumber(historical.row_count)} total rows
                      </Badge>
                      <Badge variant="outline" className="border-primary text-primary">
                        {historical.columns.length} columns
                      </Badge>
                    </div>
                  )}

                  <DataPreviewTable
                    columns={historical?.columns || []}
                    rows={historical?.preview || []}
                    emptyLabel="No data loaded yet. Run a fetch to see a preview."
                  />
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
