"use client";

import React, { createContext, useContext, useEffect, useMemo, useState, useCallback } from 'react';

export interface Document {
  id: string;
  name: string;
  uploadedOn: string;
  size: string;
  type: 'pdf' | 'doc' | 'txt';
  fileUrl?: string; // absolute URL for uploaded files served by backend
}

export interface ChatReference {
  docId: string;
  page: number;
  text?: string; // snippet to highlight
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'ai';
  text: string;
  ref?: ChatReference;
  origin?: 'document' | 'web' | 'hybrid';
  externalSources?: { title: string; url: string; snippet?: string }[];
}

export type Provider = 'mistral' | 'gemini';

interface DocsContextType {
  documents: Document[];
  chatMessages: ChatMessage[];
  selectedDocument: Document | null;
  uploadDocument: (file: File) => Promise<Document>;
  selectDocument: (doc: Document) => void;
  sendMessage: (message: string) => void;
  clearChat: () => void;
  // provider selection
  provider: Provider;
  setProvider: (p: Provider) => void;
  // PDF reference viewer
  viewerOpen: boolean;
  viewerTarget?: ChatReference;
  openReference: (ref: ChatReference) => void;
  closeReference: () => void;
}

const DocsContext = createContext<DocsContextType | undefined>(undefined);

// Backend base URL (set via NEXT_PUBLIC_API_BASE_URL, fallback to localhost)
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

// Local storage keys
const THREADS_KEY = 'docsum-threads'; // Record<docId, ChatMessage[]>
const LEGACY_MESSAGES_KEY = 'docsum-messages';

export function DocsProvider({ children }: { children: React.ReactNode }) {
  const [documents, setDocuments] = useState<Document[]>([]);
  // threads keyed by docId, persisted in localStorage
  const [threads, setThreads] = useState<Record<string, ChatMessage[]>>({});
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  // provider with persistence
  const [provider, setProvider] = useState<Provider>(() => {
    if (typeof window === 'undefined') return 'mistral';
    return (localStorage.getItem('provider') as Provider) || 'mistral';
  });

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('provider', provider);
    }
  }, [provider]);
  const [viewerOpen, setViewerOpen] = useState(false);
  const [viewerTarget, setViewerTarget] = useState<ChatReference | undefined>(undefined);

  // Load threads from localStorage on mount (with legacy migration)
  useEffect(() => {
    try {
      const savedThreads = localStorage.getItem(THREADS_KEY);
      if (savedThreads) {
        setThreads(JSON.parse(savedThreads));
      } else {
        const legacy = localStorage.getItem(LEGACY_MESSAGES_KEY);
        if (legacy) {
          const legacyMessages: ChatMessage[] = JSON.parse(legacy);
          setThreads({ all: legacyMessages });
          localStorage.removeItem(LEGACY_MESSAGES_KEY);
          localStorage.setItem(THREADS_KEY, JSON.stringify({ all: legacyMessages }));
        }
      }
    } catch (e) {
      console.error('Failed to load chat threads', e);
    }
  }, []);

  // Persist threads whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(THREADS_KEY, JSON.stringify(threads));
    } catch (e) {
      console.error('Failed to save chat threads', e);
    }
  }, [threads]);

  // Fetch documents from backend
  useEffect(() => {
    const fetchDocs = async () => {
      try {
        const res = await fetch(`${API_BASE}/documents`);
        if (!res.ok) throw new Error(`Failed to load documents: ${res.status}`);
        const data = await res.json();
        const mapped: Document[] = (data.documents || []).map((d: any) => ({
          id: d.id,
          name: d.name,
          uploadedOn: formatUploadedOn(d.uploadedOn),
          size: typeof d.size === 'number' ? formatFileSize(d.size) : d.size,
          type: (d.type || 'pdf') as Document['type'],
          fileUrl: d.fileUrl ? absolutizeUrl(d.fileUrl) : undefined,
        }));
        setDocuments(mapped);
        if (mapped.length && !selectedDocument) setSelectedDocument(mapped[0]);
      } catch (e) {
        console.error(e);
      }
    };
    fetchDocs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const uploadDocument = async (file: File) => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || 'Upload failed');
    }
    const data = await res.json();
    const d = data.document;
    const newDoc: Document = {
      id: d.id,
      name: d.name,
      uploadedOn: formatUploadedOn(d.uploadedOn),
      size: typeof d.size === 'number' ? formatFileSize(d.size) : d.size,
      type: (d.type || 'pdf') as Document['type'],
      fileUrl: d.fileUrl ? absolutizeUrl(d.fileUrl) : undefined,
    };
    setDocuments(prev => [newDoc, ...prev]);
    setSelectedDocument(newDoc);
    return newDoc;
  };

  const selectDocument = (doc: Document) => {
    setSelectedDocument(doc);
  };

  const currentThreadId = selectedDocument?.id || 'all';
  const chatMessages = useMemo(() => threads[currentThreadId] || [], [threads, currentThreadId]);

  const sendMessage = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    if (!selectedDocument) {
      const aiMessage: ChatMessage = {
        id: Date.now().toString() + '-ai',
        sender: 'ai',
        text: 'Please select a document first to chat about it.',
      };
      setThreads(prev => ({ ...prev, all: [...(prev.all || []), aiMessage] }));
      return;
    }

    const userMessage: ChatMessage = {
      id: Date.now().toString() + '-user',
      sender: 'user',
      text: trimmed,
    };
    const currentThreadId = selectedDocument?.id || 'all';
    setThreads(prev => ({ ...prev, [currentThreadId]: [...(prev[currentThreadId] || []), userMessage] }));

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ docId: selectedDocument.id, message: trimmed, max_chunks: 4, provider }),
      });
      if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
      const data = await res.json();
      const aiMessage: ChatMessage = {
        id: Date.now().toString() + '-ai',
        sender: 'ai',
        text: data.answer || 'No answer returned.',
        ref: data.reference ? { docId: data.reference.docId, page: data.reference.page, text: data.reference.text } : undefined,
        origin: data.origin || 'document',
        externalSources: data.sources || [],
      };
      setThreads(prev => ({ ...prev, [currentThreadId]: [...(prev[currentThreadId] || []), aiMessage] }));
    } catch (err: any) {
      console.error(err);
      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
        text: 'Sorry, something went wrong. Please try again.',
      };
      setThreads(prev => ({ ...prev, [currentThreadId]: [...(prev[currentThreadId] || []), aiMessage] }));
    }
  }, [selectedDocument, provider]);

  const clearChat = () => {
    setThreads(prev => ({ ...prev, [currentThreadId]: [] }));
  };

  const openReference = (ref: ChatReference) => {
    setViewerTarget(ref);
    setViewerOpen(true);
  };

  const closeReference = () => {
    setViewerOpen(false);
    setViewerTarget(undefined);
  };

  return (
    <DocsContext.Provider
      value={{
        documents,
        chatMessages,
        selectedDocument,
        uploadDocument,
        selectDocument,
        sendMessage,
        clearChat,
        provider,
        setProvider,
        viewerOpen,
        viewerTarget,
        openReference,
        closeReference,
      }}
    >
      {children}
    </DocsContext.Provider>
  );
}

export function useDocs() {
  const context = useContext(DocsContext);
  if (context === undefined) {
    throw new Error('useDocs must be used within a DocsProvider');
  }
  return context;
}

// Helper functions
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function formatUploadedOn(input: string): string {
  // Keep ISO string as-is for now; could format for display if needed
  try {
    const d = new Date(input);
    if (!isNaN(d.getTime())) {
      return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });
    }
    return input;
  } catch {
    return input;
  }
}

function absolutizeUrl(url: string): string {
  if (!url) return url;
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  return `${API_BASE}${url.startsWith('/') ? url : `/${url}`}`;
}