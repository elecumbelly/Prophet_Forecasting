import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Prophet · Forecasting Console",
  description:
    "Collapsing Wave Functions console for inspecting historical metrics and generating Prophet forecasts.",
};

const themeBootstrap = `
(function () {
  try {
    var stored = localStorage.getItem("cwf-theme");
    if (stored === "dark") {
      document.documentElement.classList.add("dark");
    }
  } catch (e) {}
})();
`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeBootstrap }} />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
