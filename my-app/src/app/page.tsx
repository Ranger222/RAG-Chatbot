"use client";

import UploadArea from "@/components/UploadArea";
import FileList from "@/components/FileList";

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl md:text-2xl font-semibold">Choose a document to summarize</h1>
        <p className="text-gray-600 text-sm mt-1 max-w-2xl">
          Transform lengthy documents into concise summaries and chat about it.
        </p>
      </div>

      <section className="space-y-3">
        <h2 className="text-sm font-medium">Upload new document</h2>
        <UploadArea />
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-medium">Or, Select from recent files</h2>
        <FileList />
      </section>
    </div>
  );
}
