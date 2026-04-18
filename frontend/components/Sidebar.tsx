'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  FileText,
  Zap,
  Database,
  ClipboardCheck,
  Cpu,
  MessageSquare,
  Telescope,
  ChevronRight,
  ChevronLeft,
  Factory,
} from 'lucide-react';

const NAV_ITEMS = [
  { href: '/dokumenty', icon: FileText, label: 'Dokumenty' },
  { href: '/autopilot', icon: Zap, label: 'AutoPilot' },
  { href: '/dataset', icon: Database, label: 'Dataset' },
  { href: '/review', icon: ClipboardCheck, label: 'Review' },
  { href: '/trening', icon: Cpu, label: 'Trening' },
  { href: '/chatbot', icon: MessageSquare, label: 'Chatbot' },
  { href: '/scout', icon: Telescope, label: 'Gap Scout' },
];

// Use absolute URL so the browser calls the API directly (avoids build-time proxy baking)
const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8080';

export default function Sidebar() {
  const pathname = usePathname();
  const [expanded, setExpanded] = useState(false);
  const [healthy, setHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function checkHealth() {
      try {
        const res = await fetch(`${API}/health`, { cache: 'no-store' });
        if (!cancelled) setHealthy(res.ok);
      } catch {
        if (!cancelled) setHealthy(false);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 30_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const sidebarWidth = expanded ? 'w-[220px]' : 'w-[64px]';

  return (
    <aside
      className={`
        flex flex-col flex-shrink-0 h-full
        bg-bg-surface border-r border-border
        transition-all duration-200 ease-in-out
        ${sidebarWidth}
        relative z-20
      `}
    >
      {/* Logo */}
      <div className="flex items-center h-14 px-4 border-b border-border overflow-hidden">
        <Factory className="w-6 h-6 text-accent flex-shrink-0" />
        {expanded && (
          <span className="ml-3 font-semibold text-text whitespace-nowrap text-sm">
            Foundry Studio
          </span>
        )}
      </div>

      {/* Nav Links */}
      <nav className="flex-1 py-3 overflow-hidden">
        {NAV_ITEMS.map(({ href, icon: Icon, label }) => {
          const isActive = pathname === href || pathname.startsWith(href + '/');
          return (
            <Link
              key={href}
              href={href}
              title={!expanded ? label : undefined}
              className={`
                flex items-center h-10 px-4 mx-2 my-0.5 rounded-md
                transition-colors duration-150 group
                ${isActive
                  ? 'bg-accent/15 text-accent'
                  : 'text-text-muted hover:bg-bg-surface2 hover:text-text'
                }
              `}
            >
              <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-accent' : ''}`} />
              {expanded && (
                <span className="ml-3 text-sm whitespace-nowrap overflow-hidden">
                  {label}
                </span>
              )}
              {isActive && !expanded && (
                <span className="absolute left-[60px] bg-bg-surface2 text-text text-xs px-2 py-1 rounded border border-border opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap transition-opacity">
                  {label}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Bottom: health + toggle */}
      <div className="border-t border-border p-3 space-y-2">
        {/* Health indicator */}
        <div
          className="flex items-center gap-2 px-1 overflow-hidden"
          title="Status API"
        >
          <span
            className={`
              w-2.5 h-2.5 rounded-full flex-shrink-0
              ${healthy === true
                ? 'bg-success'
                : healthy === false
                ? 'bg-error'
                : 'bg-text-muted'
              }
            `}
          />
          {expanded && (
            <span className="text-xs text-text-muted whitespace-nowrap">
              {healthy === true ? 'API online' : healthy === false ? 'API offline' : 'Sprawdzanie...'}
            </span>
          )}
        </div>

        {/* Toggle button */}
        <button
          onClick={() => setExpanded((v) => !v)}
          className="
            flex items-center justify-center w-full h-8 rounded-md
            text-text-muted hover:text-text hover:bg-bg-surface2
            transition-colors duration-150
          "
          title={expanded ? 'Zwiń panel' : 'Rozwiń panel'}
        >
          {expanded ? (
            <ChevronLeft className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
        </button>
      </div>
    </aside>
  );
}
