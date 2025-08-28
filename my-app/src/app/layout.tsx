import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { DocsProvider } from "@/contexts/DocsContext";
import AppShell from "@/components/AppShell";

export const metadata: Metadata = {
  title: "DocuSum",
  description: "Chat with your documents — mock frontend",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`antialiased`}>
        <DocsProvider>
          <AppShell>{children}</AppShell>
        </DocsProvider>
      </body>
    </html>
  );
}
