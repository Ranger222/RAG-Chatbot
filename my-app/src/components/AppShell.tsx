"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import PdfSidebar from "@/components/PdfSidebar";

export default function AppShell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-[#fafafa] text-[#111]">
      <div className="flex">
        {/* Sidebar */}
        <aside className={`fixed z-30 inset-y-0 left-0 w-64 bg-white border-r border-black/10 transform transition-transform duration-200 ${open ? "translate-x-0" : "-translate-x-full"} md:translate-x-0`}>
          <div className="px-4 py-4 border-b border-black/10 flex items-center justify-between">
            <Link href="/" className="font-semibold text-[#4438ca]">DocuSum</Link>
            <button className="md:hidden p-2 rounded hover:bg-black/5" onClick={() => setOpen(false)} aria-label="Close menu">
              <CloseIcon />
            </button>
          </div>
          <nav className="p-3 space-y-1">
            <NavItem href="/" active={pathname === "/"} onClick={() => setOpen(false)}>Dashboard</NavItem>
            <NavItem href="/chat" active={pathname?.startsWith("/chat") ?? false} onClick={() => setOpen(false)}>Chat</NavItem>
            <NavItem href="/saved" active={pathname?.startsWith("/saved") ?? false} onClick={() => setOpen(false)}>Saved Documents</NavItem>
            <NavItem href="/settings" active={pathname?.startsWith("/settings") ?? false} onClick={() => setOpen(false)}>Settings</NavItem>
          </nav>
        </aside>

        {/* Content area with left padding for sidebar, and room on the right for PDF viewer on lg+ */}
        <div className="flex-1 md:pl-64 lg:pr-[520px]">
          <header className="sticky top-0 z-20 bg-[#fafafa]/90 backdrop-blur border-b border-black/10">
            <div className="h-14 flex items-center justify-between px-4">
              <button className="md:hidden p-2 rounded hover:bg-black/5" onClick={() => setOpen(true)} aria-label="Open menu">
                <MenuIcon />
              </button>
              <div className="flex-1" />
              <div className="flex items-center gap-2 text-gray-500">
                <button className="p-2 rounded hover:bg-black/5" aria-label="Notifications"><BellIcon /></button>
                <button className="p-2 rounded hover:bg-black/5" aria-label="Settings"><GearIcon /></button>
              </div>
            </div>
          </header>
          <main className="p-4 md:p-6">{children}</main>
        </div>
      </div>

      {/* PDF Sidebar overlay (slides from right). On large screens we reserve space via lg:pr-[520px] */}
      <PdfSidebar />
    </div>
  );
}

function NavItem({ href, active, children, onClick }: { href: string; active?: boolean; children: React.ReactNode; onClick?: () => void }) {
  return (
    <Link
      href={href}
      onClick={onClick}
      className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${active ? "bg-[#4438ca] text-white" : "hover:bg-black/5"}`}
    >
      {children}
    </Link>
  );
}

function MenuIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M3 6h18M3 12h18M3 18h18" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M6 6l12 12M18 6L6 18"/>
    </svg>
  );
}

function BellIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M15 17h5l-1.4-1.4A2 2 0 0 1 18 14.172V11a6 6 0 1 0-12 0v3.172a2 2 0 0 1-.586 1.414L4 17h5"/>
      <path d="M9 17a3 3 0 0 0 6 0"/>
    </svg>
  );
}

function GearIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"/>
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V22a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 8 20.61a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 3.39 17 1.65 1.65 0 0 0 1.88 16H2a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.3-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 8 3.39 1.65 1.65 0 0 0 9 1.88V2a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.3 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 20.61 8c.18.55.28 1.14.28 1.74s-.1 1.19-.28 1.74c-.21.64.05.98-.21 2.52z"/>
    </svg>
  );
}