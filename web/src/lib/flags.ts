// Feature flags. Off unless switched on at build time (VITE_FLAGS=harder,...) or, for local
// checks, with ?ff=harder in the URL.
export type Flag = "harder";

export function flagOn(flag: Flag, env: string | undefined, search: string): boolean {
  const list = [...(env ?? "").split(","), ...(new URLSearchParams(search).get("ff") ?? "").split(",")];
  return list.map((x) => x.trim()).includes(flag);
}

export const flag = (f: Flag): boolean =>
  flagOn(f, import.meta.env?.VITE_FLAGS as string | undefined, typeof window !== "undefined" ? window.location.search : "");
