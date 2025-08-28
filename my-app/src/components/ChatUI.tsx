"use client";

import React, { useRef, useEffect, useState } from "react";
import { useDocs } from "@/contexts/DocsContext";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function ChatUI() {
  const { chatMessages, sendMessage, clearChat, selectedDocument, openReference, provider, setProvider } = useDocs();
  const [input, setInput] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [chatMessages.length]);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text) return;
    sendMessage(text);
    setInput("");
  };

  const onCopy = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1500);
    } catch {
      // Fallback
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1500);
    }
  };

  return (
    <div className="relative">
      <div className="h-[calc(100vh-180px)] md:h-[calc(100vh-160px)] flex flex-col rounded-xl border border-black/10 bg-white">
        <div className="border-b border-black/10 px-4 py-3 flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-600">Chatting about</p>
            <h2 className="font-medium">{selectedDocument?.name ?? "All Documents"}</h2>
          </div>
          <div className="flex items-center gap-3">
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value as 'mistral' | 'gemini')}
              className="text-sm border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-[#4438ca]/40"
            >
              <option value="mistral">Mistral</option>
              <option value="gemini">Gemini</option>
            </select>
            <button onClick={clearChat} className="text-sm text-gray-600 hover:text-red-600">
              Clear chat
            </button>
          </div>
        </div>
        <div ref={listRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
          {chatMessages.length === 0 && (
            <div className="text-center text-gray-500 mt-10">
              Ask a question about your documents to get started.
            </div>
          )}
          {chatMessages.map((m) => (
            <div key={m.id} className="mb-4">
              <div className={m.sender === 'user' ? 'text-right' : ''}>
                <div className={`relative inline-block px-3 py-2 rounded-lg ${m.sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-900'} max-w-[90%]`}>
                  {m.sender === 'ai' ? (
                    <div className="prose prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-pre:my-2 prose-code:before:content-[''] prose-code:after:content-['']">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          a: (props: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
                            <a {...props} className="text-blue-600 hover:underline" target="_blank" rel="noreferrer" />
                          ),
                          code: (rawProps: any) => {
                            const { inline, className, children, ...rest } = rawProps || {};
                            return (
                              <code className={`${className || ''} ${inline ? 'bg-gray-200 px-1 py-0.5 rounded' : ''}`} {...rest}>
                                {children}
                              </code>
                            );
                          },
                        }}
                      >
                        {m.text}
                      </ReactMarkdown>
                      <button
                        onClick={() => onCopy(m.id, m.text)}
                        title={copiedId === m.id ? 'Copied!' : 'Copy reply'}
                        className="absolute top-2 right-2 text-xs text-gray-500 hover:text-gray-700 flex items-center gap-1"
                      >
                        <CopyIcon />
                        {copiedId === m.id ? 'Copied' : 'Copy'}
                      </button>
                    </div>
                  ) : (
                    <div>{m.text}</div>
                  )}
                </div>
              </div>
              {m.sender === 'ai' && m.origin && (
                <p className="mt-1 text-xs text-gray-500">
                  Source: {m.origin === 'document' ? 'Document' : m.origin === 'web' ? 'External web sources' : 'Document + External web sources'}
                </p>
              )}
              {m.sender === 'ai' && m.externalSources && m.externalSources.length > 0 && (
                <ul className="mt-2 ml-4 list-disc text-xs text-gray-600">
                  {m.externalSources.map((s, i) => (
                    <li key={i}>
                      <a href={s.url} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">{s.title || s.url}</a>
                      {s.snippet ? <span className="ml-1">— {s.snippet}</span> : null}
                    </li>
                  ))}
                </ul>
              )}
              {m.ref && (
                <button
                  onClick={() => openReference(m.ref!)}
                  title={`Open page ${m.ref.page} in ${selectedDocument?.name ?? 'document'}`}
                  className="mt-2 text-sm text-blue-600 hover:underline inline-flex items-center gap-1"
                >
                  <RefIcon />
                  View reference (p.{m.ref.page})
                </button>
              )}
            </div>
          ))}
        </div>
        <form onSubmit={onSubmit} className="p-3 border-t border-black/10 flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask anything about your documents..."
            rows={1}
            className="flex-1 resize-none rounded-lg border border-black/15 p-3 outline-none focus:ring-2 focus:ring-[#4438ca]/40"
          />
          <button
            type="submit"
            className="inline-flex items-center gap-2 rounded-lg bg-[#4438ca] text-white px-4 py-2 text-sm hover:bg-[#362fb0]"
          >
            <SendIcon />
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M22 2L11 13"/>
      <path d="M22 2l-7 20-4-9-9-4 20-7z"/>
    </svg>
  );
}

function RefIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M4 19h16V5H4v14z"/>
      <path d="M8 9h8M8 13h5"/>
    </svg>
  );
}

function CopyIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
      <rect x="9" y="9" width="10" height="10" rx="2"/>
      <rect x="5" y="5" width="10" height="10" rx="2"/>
    </svg>
  );
}