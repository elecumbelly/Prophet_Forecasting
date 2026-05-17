"use client";

import { useEffect, useState } from "react";

const STORAGE_KEY = "cwf-theme";

function readInitial(): "light" | "dark" {
  if (typeof document === "undefined") return "light";
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

export function ThemeToggle() {
  const [mode, setMode] = useState<"light" | "dark">(readInitial);

  useEffect(() => {
    const root = document.documentElement;
    if (mode === "dark") {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
    try {
      localStorage.setItem(STORAGE_KEY, mode);
    } catch {
      // localStorage may be unavailable (private mode, SSR pre-hydrate); ignore.
    }
  }, [mode]);

  const next = mode === "light" ? "dark" : "light";

  return (
    <button
      type="button"
      onClick={() => setMode(next)}
      className="cwf-eyebrow underline-offset-4 hover:underline focus-visible:underline decoration-foreground"
      aria-label={`Switch to ${next} mode`}
    >
      {mode === "light" ? "Dark ▸" : "Light ▸"}
    </button>
  );
}
