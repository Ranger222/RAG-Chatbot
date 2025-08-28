"use client";

import React, { useMemo, useState } from "react";
import { useDocs, Document } from "@/contexts/DocsContext";
import { useRouter } from "next/navigation";

export default function FileList({ searchable = true }: { searchable?: boolean }) {
  const { documents, selectDocument } = useDocs();
  const [query, setQuery] = useState("");
  const router = useRouter();

  const filtered = useMemo(() => {
    const q = query.toLowerCase();
    return documents.filter((d) => d.name.toLowerCase().includes(q));
  }, [documents, query]);

  const onChat = (doc: Document) => {
    selectDocument(doc);
    router.push("/chat");
  };

  return (
    <div className="w-full">
      {searchable && (
        <div className="flex items-center justify-between mb-3">
          <div className="relative w-full md:w-96">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by file name"
              className="w-full rounded-lg border border-black/15 bg-white pl-9 pr-3 py-2 text-sm outline-none focus:ring-2 focus:ring-[#4438ca]/40"
            />
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">
              <SearchIcon />
            </span>
          </div>
        </div>
      )}

      <div className="overflow-x-auto rounded-xl border border-black/10 bg-white">
        <table className="min-w-full text-sm">
          <thead className="text-gray-500 bg-black/5">
            <tr>
              <th className="text-left font-medium px-4 py-3">File Name</th>
              <th className="text-left font-medium px-4 py-3">Uploaded on</th>
              <th className="text-left font-medium px-4 py-3">File size</th>
              <th className="text-left font-medium px-4 py-3">Action</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((doc) => (
              <tr key={doc.id} className="border-t border-black/5 hover:bg-black/2">
                <td className="px-4 py-3 flex items-center gap-2 min-w-[240px]">
                  <DocIcon />
                  <span className="truncate">{doc.name}</span>
                </td>
                <td className="px-4 py-3 text-gray-600">{doc.uploadedOn}</td>
                <td className="px-4 py-3 text-gray-600">{doc.size}</td>
                <td className="px-4 py-3">
                  <button
                    onClick={() => onChat(doc)}
                    className="inline-flex items-center gap-2 text-[#4438ca] hover:underline"
                  >
                    <ChatIcon />
                    <span>Chat</span>
                  </button>
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={4} className="px-4 py-6 text-center text-gray-500">
                  No files found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DocIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M7 2h7l5 5v15a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"/>
      <path d="M14 2v5h5"/>
    </svg>
  );
}

function ChatIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4v8z"/>
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="11" cy="11" r="7"/>
      <path d="M21 21l-4-4"/>
    </svg>
  );
}