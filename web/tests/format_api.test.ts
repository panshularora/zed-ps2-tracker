import { api, ApiError } from "../src/api";
import { DASH, int, num, pct, seconds } from "../src/lib/format";

describe("format", () => {
  it("shows an em dash for anything not measured", () => {
    expect(num(null)).toBe(DASH);
    expect(num(undefined)).toBe(DASH);
    expect(num(Number.NaN)).toBe(DASH);
    expect(pct(null)).toBe(DASH);
    expect(int(null)).toBe(DASH);
    expect(seconds(null)).toBe(DASH);
    expect(num(59.84)).toBe("59.8");
    expect(pct(0.2667)).toBe("26.7 %");
    expect(int(1234)).toBe("1,234");
    expect(seconds(90)).toBe("1 min 30 s");
  });
});

describe("api client", () => {
  afterEach(() => vi.restoreAllMocks());
  it("turns structured errors into ApiError", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ error: { code: "clip_not_found", message: "no clip 'x'" }, detail: "no clip 'x'" }), { status: 404 }),
    );
    await expect(api.clips()).rejects.toMatchObject({ status: 404, code: "clip_not_found", message: "no clip 'x'" });
  });
  it("reports network failures", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    const err = await api.health().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe("network");
  });
  it("posts jobs as JSON", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ id: "abc" }), { status: 202 }));
    await api.createJob({ clip_id: "walk", stride: 2 });
    const [url, init] = spy.mock.calls[0];
    expect(url).toBe("/api/jobs");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ clip_id: "walk", stride: 2 });
  });
});
