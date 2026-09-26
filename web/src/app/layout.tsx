import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Avellaneda-Stoikov Market Making Engine",
  description: "Interactive optimal market-making model, run against a simulated limit order book.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
