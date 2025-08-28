"use client";

import React, { useEffect, useMemo, useRef } from "react";
import { useDocs } from "@/contexts/DocsContext";

export default function PdfSidebar() {
  const { viewerOpen, viewerTarget, documents, closeReference } = useDocs();
  const doc = useMemo(() => documents.find(d => d.id === viewerTarget?.docId), [documents, viewerTarget]);
  const page = viewerTarget?.page ?? 1;
  const highlight = viewerTarget?.text ? encodeURIComponent(viewerTarget.text.slice(0, 120)) : undefined;
  const viewerParams = `#page=${page}&pagemode=none&zoom=page-width${highlight ? `&search=${highlight}` : ''}`;
  const src = doc?.fileUrl ? `${doc.fileUrl}${viewerParams}` : undefined;
  const panelRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!viewerOpen) return;
    panelRef.current?.focus();
  }, [viewerOpen]);

  return (
    <div
      className={`fixed right-0 top-0 h-full z-40 transition-transform duration-300 ${
        viewerOpen ? "translate-x-0" : "translate-x-full"
      } w-full sm:w-[420px] lg:w-[520px]`}
      aria-hidden={!viewerOpen}
    >
      <div className="h-full bg-white border-l border-black/10 shadow-xl flex flex-col" ref={panelRef} tabIndex={-1}>
        <div className="px-4 py-3 border-b border-black/10 flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-600">Preview</p>
            <h3 className="font-medium truncate w-64 sm:w-80">{doc?.name ?? 'Document'}</h3>
            {viewerTarget?.text && (
              <p className="text-xs text-gray-500 mt-1">Highlight: <span className="font-medium">{viewerTarget.text}</span> (p.{page})</p>
            )}
          </div>
          <button onClick={closeReference} className="p-2 rounded-md hover:bg-black/5" aria-label="Close preview">
            <CloseIcon />
          </button>
        </div>

        <div className="relative flex-1 overflow-hidden">
          {src ? (
            <>
              <iframe title="PDF Preview" src={src} className="w-full h-full" />
              {viewerTarget?.text && (
                <div className="pointer-events-none absolute left-3 top-3 max-w-[85%] rounded-md bg-yellow-200/80 px-2 py-1 text-[11px] text-gray-900 shadow">
                  Highlighted: <span className="font-semibold">{viewerTarget.text}</span> (page {page})
                </div>
              )}
            </>
          ) : (
            <div className="h-full grid place-items-center text-center text-gray-600 p-6">
              <div>
                <p className="mb-2">No local PDF file is attached to this mock document.</p>
                <p className="text-sm text-gray-500">Upload your own PDF to see an inline preview.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function CloseIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M6 6l12 12M18 6L6 18"/>
    </svg>
  );
}