// Copyright (c) Xiangyu Li. MIT licensed; see ../../licenses/MIT.txt.
const DAY_MS = 86_400_000;
const RETENTION_DAYS = 400;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const BOT = /bot\b|crawler|spider|headless|preview|facebookexternalhit|curl|wget/i;
const PERSONAL_SITE = { id: "personal", store: "site-v1" };
const PROJECT_ORIGIN = "https://air-embodied-brain.github.io";
const PROJECTS = [
  { id: "oxygen", store: "oxygen-v1", path: "/OxyGen/" },
  { id: "steerdistill", store: "steerdistill-v1", path: "/SteerDistill/" },
];

function dateAt(time) {
  return new Date(time).toISOString().slice(0, 10);
}

function personalOrigins(env) {
  const origins = (env.ALLOWED_ORIGINS || "")
    .split(",").map(value => value.trim()).filter(Boolean);
  if (env.ALLOW_LOCALHOST === "true") {
    origins.push("http://127.0.0.1:1111", "http://localhost:1111");
  }
  return origins;
}

function eventSite(origin, path) {
  if (origin !== PROJECT_ORIGIN) return PERSONAL_SITE;
  return PROJECTS.find(site => path === site.path.slice(0, -1) || path.startsWith(site.path));
}

function normalizePath(raw, origin) {
  if (typeof raw !== "string" || !raw.startsWith("/") || raw.startsWith("//")) return null;
  const url = new URL(raw, origin);
  if (url.origin !== origin || url.pathname.length > 240) return null;
  if (/^\/(?:zh\/)?analytics(?:\/|$)/.test(url.pathname)) return null;
  return url.pathname;
}

function coordinate(value, limit) {
  if (value == null || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) && Math.abs(number) <= limit ? Math.round(number) : null;
}

function geography(cf = {}) {
  return {
    country: /^[A-Z]{2}$/.test(cf.country || "") && cf.country !== "XX" ? cf.country : "",
    city: typeof cf.city === "string" ? cf.city.replace(/[\x00-\x1f]/g, "").trim().slice(0, 80) : "",
    lat: coordinate(cf.latitude, 90),
    lng: coordinate(cf.longitude, 180),
  };
}

function referrerHost(raw, origins) {
  try {
    const url = new URL(raw);
    if (!["https:", "http:"].includes(url.protocol) || origins.includes(url.origin)) return "";
    return url.hostname.toLowerCase().slice(0, 160);
  } catch {
    return "";
  }
}

async function smallJson(request) {
  if (!request.headers.get("content-type")?.startsWith("application/json")) return null;
  const reader = request.body?.getReader();
  if (!reader) return null;
  const chunks = [];
  let size = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.length;
    if (size > 2048) {
      await reader.cancel();
      return null;
    }
    chunks.push(value);
  }
  try {
    return JSON.parse(await new Blob(chunks).text());
  } catch {
    return null;
  }
}

function cors(response, origin, revision) {
  const headers = new Headers(response.headers);
  if (origin) headers.set("Access-Control-Allow-Origin", origin);
  headers.set("Vary", "Origin");
  headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  headers.set("Access-Control-Allow-Headers", "Content-Type");
  headers.set("Access-Control-Max-Age", "86400");
  headers.set("X-Content-Type-Options", "nosniff");
  headers.set("X-Analytics-Revision", revision || "local");
  return new Response(response.body, { status: response.status, headers });
}

function reportError(error, operation, env) {
  console.error({
    event: "analytics-error",
    operation,
    error: error instanceof Error ? error.name : "UnknownError",
    version: env.CF_VERSION_METADATA?.id || "local",
    revision: env.CF_VERSION_METADATA?.tag || "local",
  });
}

export default {
  async fetch(request, env, context) {
    const url = new URL(request.url);
    const origin = request.headers.get("Origin");
    const origins = personalOrigins(env);
    if (origin && origin !== PROJECT_ORIGIN && !origins.includes(origin)) return new Response("Forbidden", { status: 403 });
    const respond = response => cors(response, origin, env.CF_VERSION_METADATA?.tag);
    if (request.method === "OPTIONS") return respond(new Response(null, { status: 204 }));
    if (url.pathname === "/" && request.method === "GET") {
      return respond(Response.redirect("https://xxxxyu.github.io/analytics/", 302));
    }
    let operation = "store-binding";
    try {
      if (url.pathname === "/api/events" && request.method === "POST") {
        operation = "track";
        if (!origin) return respond(new Response(null, { status: 403 }));
        const agent = request.headers.get("User-Agent") || "";
        if (request.headers.get("DNT") === "1" || request.headers.get("Sec-GPC") === "1" || !agent || BOT.test(agent)) {
          return respond(new Response(null, { status: 204 }));
        }
        const payload = await smallJson(request);
        const path = normalizePath(payload?.path, origin);
        if (!path || ![payload?.visitorId, payload?.eventId].every(id => typeof id === "string" && UUID.test(id))) {
          return respond(Response.json({ error: "invalid-event" }, { status: 400 }));
        }
        const site = eventSite(origin, path);
        if (!site) return respond(Response.json({ error: "invalid-site-path" }, { status: 400 }));
        const store = env.ANALYTICS.getByName(site.store);
        const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(payload.visitorId));
        const visitor = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
        return respond(await store.fetch("https://store/events", {
          method: "POST",
          body: JSON.stringify({
            eventId: payload.eventId,
            visitor,
            path,
            referrer: referrerHost(payload.referrer, site === PERSONAL_SITE ? origins : [origin]),
            ...geography(request.cf),
          }),
        }));
      }
      if (url.pathname === "/api/summary" && request.method === "GET") {
        operation = "summary";
        const id = url.searchParams.get("site") ?? PERSONAL_SITE.id;
        const site = id === PERSONAL_SITE.id ? PERSONAL_SITE : PROJECTS.find(project => project.id === id);
        if (!site) return respond(Response.json({ error: "invalid-site" }, { status: 400 }));
        const days = Number(url.searchParams.get("days") || 30);
        if (![7, 30, 90].includes(days)) return respond(Response.json({ error: "invalid-range" }, { status: 400 }));
        const key = new Request(`${url.origin}/api/summary?site=${site.id}&days=${days}&version=${encodeURIComponent(env.CF_VERSION_METADATA?.id || "local")}`);
        const cache = globalThis.caches?.default;
        const refresh = url.searchParams.get("refresh") === "1";
        let response = refresh ? null : await cache?.match(key);
        if (!response) {
          const store = env.ANALYTICS.getByName(site.store);
          response = await store.fetch(`https://store/summary?days=${days}`);
          if (response.ok && cache) {
            context.waitUntil(cache.put(key, response.clone()).catch(error => reportError(error, "cache-write", env)));
          }
        }
        if (refresh) {
          response = new Response(response.body, response);
          response.headers.set("Cache-Control", "no-store");
        }
        return respond(response);
      }
      return respond(Response.json({ error: "not-found" }, { status: 404 }));
    } catch (error) {
      reportError(error, operation, env);
      return respond(Response.json({ error: "temporarily-unavailable" }, { status: 503 }));
    }
  },
};

export class AnalyticsStore {
  constructor(ctx) {
    this.storage = ctx.storage;
    this.sql = ctx.storage.sql;
    this.sql.exec(`
      CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        day TEXT NOT NULL,
        visitor TEXT NOT NULL,
        path TEXT NOT NULL,
        referrer TEXT NOT NULL,
        country TEXT NOT NULL,
        city TEXT NOT NULL,
        lat INTEGER,
        lng INTEGER
      );
      CREATE INDEX IF NOT EXISTS events_day ON events(day);
      CREATE TABLE IF NOT EXISTS visitors (id TEXT PRIMARY KEY, last_seen TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    `);
  }

  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === "/events" && request.method === "POST") {
      this.record(await request.json());
      return new Response(null, { status: 204 });
    }
    return Response.json(this.summary(Number(url.searchParams.get("days") || 30)), {
      headers: { "Cache-Control": "public, max-age=60" },
    });
  }

  record(event, now = Date.now()) {
    const day = dateAt(now);
    this.storage.transactionSync(() => {
      this.sql.exec(
        "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        event.eventId, day, event.visitor, event.path, event.referrer,
        event.country, event.city, event.lat, event.lng,
      );
      if (!this.sql.exec("SELECT changes() AS added").toArray()[0].added) return;
      const returning = this.sql.exec("SELECT 1 FROM visitors WHERE id = ?", event.visitor).toArray().length;
      this.sql.exec(
        "INSERT INTO visitors VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET last_seen = excluded.last_seen",
        event.visitor, day,
      );
      this.increment("views");
      if (!returning) this.increment("visitors");
      this.sql.exec("INSERT OR IGNORE INTO meta VALUES ('started_at', ?)", new Date(now).toISOString());
      const lastCleanup = this.sql.exec("SELECT value FROM meta WHERE key = 'cleanup'").toArray()[0]?.value;
      if (lastCleanup !== day) {
        this.sql.exec("DELETE FROM events WHERE day < ?", dateAt(now - (RETENTION_DAYS - 1) * DAY_MS));
        this.sql.exec("DELETE FROM visitors WHERE last_seen < ?", dateAt(now - 366 * DAY_MS));
        this.sql.exec("INSERT OR REPLACE INTO meta VALUES ('cleanup', ?)", day);
      }
    });
  }

  increment(key) {
    this.sql.exec(
      "INSERT INTO meta VALUES (?, '1') ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1",
      key,
    );
  }

  summary(dayCount, now = Date.now()) {
    const to = dateAt(now);
    const from = dateAt(now - (dayCount - 1) * DAY_MS);
    const rows = (query, ...bindings) => this.sql.exec(query, ...bindings).toArray();
    const period = rows(`SELECT COUNT(*) AS views, COUNT(DISTINCT visitor) AS visitors,
      COUNT(DISTINCT NULLIF(country, '')) AS countries FROM events WHERE day >= ?`, from)[0];
    const today = rows("SELECT COUNT(*) AS views, COUNT(DISTINCT visitor) AS visitors FROM events WHERE day = ?", to)[0];
    const daily = rows(`SELECT day AS date, COUNT(*) AS views, COUNT(DISTINCT visitor) AS visitors
      FROM events WHERE day >= ? GROUP BY day ORDER BY day`, from);
    const byDate = new Map(daily.map(row => [row.date, row]));
    const days = Array.from({ length: dayCount }, (_, index) => {
      const date = dateAt(now - (dayCount - 1 - index) * DAY_MS);
      return byDate.get(date) || { date, views: 0, visitors: 0 };
    });
    const meta = Object.fromEntries(rows("SELECT key, value FROM meta").map(row => [row.key, row.value]));
    return {
      generatedAt: new Date(now).toISOString(),
      startedAt: meta.started_at || null,
      period: { days: dayCount, from, to, ...period },
      lifetime: { views: Number(meta.views || 0), visitors: Number(meta.visitors || 0) },
      today,
      days,
      countries: rows(`SELECT country AS code, COUNT(*) AS views FROM events
        WHERE day >= ? AND country != '' GROUP BY country ORDER BY views DESC, country LIMIT 100`, from),
      cities: rows(`SELECT city, country, COUNT(*) AS views FROM events
        WHERE day >= ? AND city != '' GROUP BY city, country ORDER BY views DESC, city LIMIT 100`, from),
      pages: rows(`SELECT path, COUNT(*) AS views FROM events WHERE day >= ?
        GROUP BY path ORDER BY views DESC, path LIMIT 20`, from),
      referrers: rows(`SELECT referrer AS host, COUNT(*) AS views FROM events WHERE day >= ?
        GROUP BY referrer ORDER BY views DESC, referrer LIMIT 20`, from),
      points: rows(`SELECT lat, lng, COUNT(*) AS views FROM events
        WHERE day >= ? AND lat IS NOT NULL AND lng IS NOT NULL
        GROUP BY lat, lng ORDER BY views DESC, lat, lng LIMIT 1500`, from),
    };
  }
}
