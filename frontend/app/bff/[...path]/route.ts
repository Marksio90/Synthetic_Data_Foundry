import { NextRequest, NextResponse } from 'next/server';

const API_BASE =
  process.env.API_URL ??
  process.env.NEXT_PRIVATE_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  'http://localhost:8080';

const ADMIN_API_KEY =
  process.env.ADMIN_API_KEY ??
  process.env.NEXT_PRIVATE_ADMIN_API_KEY ??
  '';

function buildTargetUrl(pathSegments: string[], request: NextRequest): string {
  const path = pathSegments.join('/');
  const query = request.nextUrl.search;
  return `${API_BASE.replace(/\/$/, '')}/${path}${query}`;
}

function buildForwardHeaders(request: NextRequest): Headers {
  const headers = new Headers();

  request.headers.forEach((value, key) => {
    const lower = key.toLowerCase();
    // Hop-by-hop headers are controlled by fetch/runtime.
    if (['host', 'connection', 'content-length'].includes(lower)) {
      return;
    }
    headers.set(key, value);
  });

  if (ADMIN_API_KEY) {
    headers.set('X-API-Key', ADMIN_API_KEY);
  }

  return headers;
}

async function proxy(request: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const targetUrl = buildTargetUrl(path, request);

  const method = request.method.toUpperCase();
  const headers = buildForwardHeaders(request);

  const init: RequestInit = {
    method,
    headers,
    cache: 'no-store',
  };

  if (!['GET', 'HEAD'].includes(method)) {
    init.body = await request.arrayBuffer();
  }

  const upstream = await fetch(targetUrl, init);
  const responseHeaders = new Headers(upstream.headers);
  responseHeaders.delete('content-encoding');
  responseHeaders.delete('content-length');

  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

export async function GET(request: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(request, ctx);
}

export async function POST(request: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(request, ctx);
}

export async function PUT(request: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(request, ctx);
}

export async function DELETE(request: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(request, ctx);
}

export async function PATCH(request: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(request, ctx);
}
