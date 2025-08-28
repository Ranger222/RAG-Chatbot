"use client";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-xl md:text-2xl font-semibold">Settings</h1>
      <div className="rounded-xl border border-black/10 bg-white p-4">
        <div className="flex items-center justify-between py-2">
          <div>
            <p className="font-medium">Dark mode</p>
            <p className="text-gray-600 text-sm">Coming soon</p>
          </div>
          <button className="px-3 py-1 rounded-md border border-black/15 text-sm bg-black/5 text-gray-600">Off</button>
        </div>
        <div className="h-px bg-black/10 my-2" />
        <div className="py-2 text-gray-600 text-sm">More settings will appear here.</div>
      </div>
    </div>
  );
}