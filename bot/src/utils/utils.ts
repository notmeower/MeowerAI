import fetch from "node-fetch";
import https from "https";
import http from "http";
import { FetchResponse } from "../types";

export function formatNumber(num: number | undefined | null): string {
  if (num === undefined || num === null) return "unknown";

  const withCommas = num.toLocaleString();
  let shortForm = "";

  if (num >= 1_000_000_000) {
    shortForm = `${(num / 1_000_000_000).toFixed(2)}B`;
  } else if (num >= 1_000_000) {
    shortForm = `${(num / 1_000_000).toFixed(2)}M`;
  } else if (num >= 1_000) {
    shortForm = `${(num / 1_000).toFixed(2)}K`;
  }

  return shortForm ? `${withCommas} (${shortForm})` : withCommas;
}

export function buildHeaders(apiKey: string): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }
  return headers;
}

export async function fetchJson<T = any>(
  url: string,
  apiKey: string,
  agent: https.Agent | http.Agent | null,
  opts: any = {},
): Promise<FetchResponse<T>> {
  console.log(`[json] Fetching: ${url}`);

  const merged: any = {
    ...opts,
    headers: {
      ...(opts.headers || {}),
      ...buildHeaders(apiKey),
    },
  };

  if (agent) {
    merged.agent = agent;
  }

  try {
    const res = await fetch(url, merged);
    const text = await res.text();

    let parsed: any;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }

    console.log(`[json] Response ${res.status}: ${text.substring(0, 100)}...`);

    return { ok: res.ok, status: res.status, body: parsed };
  } catch (err: any) {
    console.error(`[json] Fetch error for ${url}:`, err.message);
    throw err;
  }
}
