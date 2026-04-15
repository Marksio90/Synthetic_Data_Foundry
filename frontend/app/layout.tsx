import type { Metadata } from 'next';
import './globals.css';
import Sidebar from '@/components/Sidebar';

export const metadata: Metadata = {
  title: 'Foundry Studio',
  description: 'Synthetic Data Foundry – interfejs zarządzania',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="pl" className="dark">
      <body className="flex h-screen overflow-hidden bg-bg-base text-text">
        <Sidebar />
        <main className="flex-1 overflow-y-auto min-w-0">
          <div className="p-6 max-w-full">
            {children}
          </div>
        </main>
      </body>
    </html>
  );
}
