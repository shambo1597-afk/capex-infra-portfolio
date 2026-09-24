"""
One-off diagnostic for niftyindices.com browser automation (not part of the pipeline).

Runs the navigation matrix (engine x headless x wait_until x context options),
records status codes, redirect chains, bot-protection headers, stalled requests
and screenshots; then, on the first successful load, dumps the real DOM of the
historical-data dropdowns so the selectors in fetch_benchmark_tri_automated()
can be written against what the site actually renders.

Usage:
    pip install playwright && playwright install chromium firefox webkit
    python diagnose_niftyindices.py                 # full matrix
    python diagnose_niftyindices.py --engines chromium --quick
Outputs go to ./tri_diagnostics/ (report.json, *.png, dom_*.html).
"""
import argparse
import json
import time
from pathlib import Path

URL = "https://niftyindices.com/reports/historical-data"
BOT_HEADER_HINTS = ("server", "cf-ray", "cf-mitigated", "akamai", "x-akamai", "x-cache",
                    "x-iinfo", "x-sucuri", "set-cookie", "location", "via", "x-powered-by")
REALISTIC_CONTEXT = dict(
    viewport={"width": 1366, "height": 768},
    locale="en-IN",
    timezone_id="Asia/Kolkata",
    extra_http_headers={"Accept-Language": "en-IN,en;q=0.9"},
)

DOM_PROBE_JS = r"""
() => {
  const out = {selects: [], text_hits: [], iframes: document.querySelectorAll('iframe').length,
               title: document.title, body_text_len: (document.body && document.body.innerText.length) || 0,
               has_jquery: typeof window.jQuery === 'function',
               frameworks: {react: !!document.querySelector('[data-reactroot],#root,#__next'),
                            angular: !!document.querySelector('[ng-version],[ng-app]'),
                            vue: !!document.querySelector('[data-v-app],#app')}};
  for (const s of document.querySelectorAll('select')) {
    const r = s.getBoundingClientRect();
    out.selects.push({id: s.id, name: s.name, cls: s.className, visible: r.width > 0 && r.height > 0,
      options: [...s.options].slice(0, 40).map(o => [o.value, o.text.trim()])});
  }
  const needles = ['Historical Index Data', 'Total returns Index Values', 'Select an Index Type',
                   'Select a Sub-Index', 'Select an Index', 'Submit', 'csv'];
  const all = document.querySelectorAll('body *');
  for (const n of needles) {
    let best = null;
    for (const el of all) {
      const t = (el.innerText || el.value || '').trim();
      if (t && t.toLowerCase().includes(n.toLowerCase()) && (!best || t.length <= best.t.length)) best = {el, t};
    }
    if (best) {
      const el = best.el, p = el.parentElement;
      out.text_hits.push({needle: n, tag: el.tagName, id: el.id, cls: String(el.className),
        role: el.getAttribute('role'), onclick: el.getAttribute('onclick'),
        outer: el.outerHTML.slice(0, 600), parent_outer: p ? p.outerHTML.slice(0, 1500) : null});
    } else out.text_hits.push({needle: n, found: false});
  }
  out.inputs = [...document.querySelectorAll('input')].slice(0, 40)
     .map(i => ({id: i.id, name: i.name, type: i.type, cls: i.className, placeholder: i.placeholder}));
  return out;
}
"""


def attempt(p, engine, headless, wait_until, realistic, timeout_ms, outdir, dump_dom):
    tag = f"{engine}_{'headless' if headless else 'headed'}_{wait_until}_{'ctx' if realistic else 'bare'}"
    rec = {"tag": tag, "engine": engine, "headless": headless, "wait_until": wait_until,
           "realistic_context": realistic}
    launcher = getattr(p, engine)
    t0 = time.time()
    try:
        browser = launcher.launch(headless=headless)
    except Exception as e:
        rec["launch_error"] = str(e).splitlines()[0]
        return rec
    inflight, failed, main_chain = {}, [], []
    try:
        ctx = browser.new_context(**(REALISTIC_CONTEXT if realistic else {}))
        page = ctx.new_page()
        page.on("request", lambda r: inflight.__setitem__(id(r), r.url))
        page.on("requestfinished", lambda r: inflight.pop(id(r), None))
        page.on("requestfailed", lambda r: (inflight.pop(id(r), None),
                                            failed.append([r.url[:150], r.failure])))

        def on_response(resp):
            if resp.request.resource_type == "document":
                main_chain.append({"url": resp.url, "status": resp.status,
                                   "headers": {k: v[:200] for k, v in resp.headers.items()
                                               if any(h in k.lower() for h in BOT_HEADER_HINTS)}})
        page.on("response", on_response)
        try:
            resp = page.goto(URL, wait_until=wait_until, timeout=timeout_ms)
            rec["goto"] = "ok"
            rec["status"] = resp.status if resp else None
        except Exception as e:
            rec["goto"] = "error"
            rec["goto_error"] = str(e).splitlines()[0]
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec["document_responses"] = main_chain
        rec["failed_requests"] = failed[:20]
        rec["still_inflight_at_end"] = sorted(set(inflight.values()))[:30]
        try:
            rec["title"] = page.title()
            html = page.content()
            rec["html_len"] = len(html)
            low = html.lower()
            rec["challenge_markers"] = [m for m in ("captcha", "cf-challenge", "challenge-platform",
                                                    "access denied", "akamai", "reference #",
                                                    "_abck", "bm_sz", "incapsula", "just a moment")
                                        if m in low]
            page.screenshot(path=str(outdir / f"{tag}.png"), full_page=True, timeout=15000)
            if dump_dom and rec.get("goto") == "ok":
                (outdir / f"dom_{tag}.html").write_text(html, encoding="utf-8")
                rec["dom_probe"] = page.evaluate(DOM_PROBE_JS)
        except Exception as e:
            rec["post_error"] = str(e).splitlines()[0]
    finally:
        browser.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="chromium,firefox,webkit")
    ap.add_argument("--timeout", type=int, default=90000)
    ap.add_argument("--quick", action="store_true", help="only domcontentloaded + realistic ctx, headless")
    ap.add_argument("--headed", action="store_true", help="also try headless=False (needs a display/xvfb)")
    ap.add_argument("--out", default="tri_diagnostics")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    from playwright.sync_api import sync_playwright
    waits = ["domcontentloaded"] if args.quick else ["commit", "domcontentloaded", "load", "networkidle"]
    ctxs = [True] if args.quick else [False, True]
    modes = [True, False] if args.headed else [True]
    results, dom_dumped = [], False
    with sync_playwright() as p:
        for engine in args.engines.split(","):
            for headless in modes:
                for realistic in ctxs:
                    for w in waits:
                        r = attempt(p, engine, headless, w, realistic, args.timeout, outdir,
                                    dump_dom=not dom_dumped)
                        dom_dumped = dom_dumped or "dom_probe" in r
                        results.append(r)
                        print(f"{r['tag']:55s} goto={r.get('goto', r.get('launch_error'))} "
                              f"status={r.get('status')} t={r.get('elapsed_s')}s "
                              f"title={r.get('title', '')[:40]!r} markers={r.get('challenge_markers')} "
                              f"err={r.get('goto_error', '')[:90]}")
    (outdir / "report.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nFull report: {outdir / 'report.json'}  (send this file + the PNGs back)")


if __name__ == "__main__":
    main()
