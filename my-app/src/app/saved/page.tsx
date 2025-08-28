"use client";

import FileList from "@/components/FileList";

export default function SavedPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl md:text-2xl font-semibold">Saved Documents</h1>
      <FileList />
    </div>
  );
}