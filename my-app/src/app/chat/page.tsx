"use client";

import ChatUI from "@/components/ChatUI";

export default function ChatPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl md:text-2xl font-semibold">Chat</h1>
      <ChatUI />
    </div>
  );
}