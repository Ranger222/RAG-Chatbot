"use client";

import React, { useCallback, useRef, useState } from "react";
import { useDocs } from "@/contexts/DocsContext";

export default function UploadArea() {
  const { uploadDocument } = useDocs();
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) uploadDocument(file);
  }, [uploadDocument]);

  const onBrowse = () => inputRef.current?.click();

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={onDrop}
      className={`w-full h-44 rounded-xl border-2 border-dashed flex items-center justify-center text-sm text-center bg-white ${
        isDragging ? "border-[#4438ca] bg-[#f8f7ff]" : "border-black/15"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) uploadDocument(file);
        }}
      />
      <div className="space-y-1">
        <div className="mx-auto w-10 h-10 rounded-full bg-black/5 flex items-center justify-center">
          <UploadIcon />
        </div>
        <button className="text-[#4438ca] font-medium hover:underline" onClick={onBrowse}>
          Click to upload
        </button>
        <p className="text-gray-500">or drag and drop</p>
        <p className="text-gray-400 text-xs">PDF only (max 20MB)</p>
      </div>
    </div>
  );
}

function UploadIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M12 16V4M12 4l-4 4m4-4l4 4"/>
      <path d="M20 16v3a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-3"/>
    </svg>
  );
}