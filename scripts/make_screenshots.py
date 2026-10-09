"""Regenerate the README screenshots in docs/images/ (offline, reproducible).

    pip install -e ".[all]" ansi2html nbconvert playwright
    python scripts/make_screenshots.py

Every terminal screenshot is the *real* output of the command shown in its title bar,
run with the offline provider so no API key is needed.
"""

from __future__ import annotations

import html
import os
import subprocess
import sys
from pathlib import Path

from ansi2html import Ansi2HTMLConverter
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
ENV = {**os.environ, "FORCE_COLOR": "1", "PY_COLORS": "1", "LLM_PROVIDER": "offline", "DATA_DIR": "data", "COLUMNS": "120"}

TERMINAL_CSS = """
body{margin:0;padding:28px;background:#e9edf2;font-family:-apple-system,Segoe UI,sans-serif}
.win{display:inline-block;min-width:860px;max-width:1180px;border-radius:10px;overflow:hidden;
     box-shadow:0 12px 32px rgba(15,23,42,.28);background:#0f172a}
.bar{background:#1e293b;padding:10px 14px;display:flex;align-items:center;gap:8px}
.dot{width:12px;height:12px;border-radius:50%}
.title{color:#94a3b8;font-size:13px;margin-left:10px;font-family:ui-monospace,Consolas,monospace}
pre{margin:0;padding:18px 22px;color:#e2e8f0;font:14px/1.55 ui-monospace,'Cascadia Code',Consolas,monospace;
    white-space:pre-wrap;word-break:break-word}
.prompt{color:#22c55e}
"""


def run(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, cwd=ROOT, env=ENV, capture_output=True, text=True)
    return (result.stdout + result.stderr).rstrip()


def terminal_html(command: str, output: str, title: str) -> str:
    conv = Ansi2HTMLConverter(inline=True, dark_bg=True)
    body = conv.convert(output, full=False)
    return f"""<html><head><meta charset="utf-8"><style>{TERMINAL_CSS}</style></head><body>
<div class="win"><div class="bar"><span class="dot" style="background:#ef4444"></span>
<span class="dot" style="background:#f59e0b"></span><span class="dot" style="background:#22c55e"></span>
<span class="title">{html.escape(title)}</span></div>
<pre><span class="prompt">$</span> {html.escape(command)}\n{body}</pre></div></body></html>"""


def shoot(page, html_doc: str, name: str, selector: str = ".win") -> None:
    page.set_content(html_doc, wait_until="networkidle")
    page.locator(selector).first.screenshot(path=str(OUT / name))
    print("  ✓", name)


ARCHITECTURE = """<html><head><meta charset="utf-8"><style>
body{margin:0;padding:28px;background:#e9edf2;font-family:-apple-system,Segoe UI,sans-serif}
.card{display:inline-block;background:#fff;border-radius:12px;padding:22px 26px;box-shadow:0 12px 32px rgba(15,23,42,.18)}
h2{margin:0 0 4px;font-size:20px;color:#0f172a}p{margin:0 0 14px;color:#475569;font-size:14px}
text{font-family:-apple-system,Segoe UI,sans-serif} .mono text{font-family:ui-monospace,Consolas,monospace}
</style></head><body><div class="card">
<h2>DataWizard: how the agent analyzes your CSV files</h2>
<p>The LLM never sees the data. It calls tools by name; the tools load CSVs once into a cache and return small summaries and metrics.</p>
<svg width="1080" height="420" viewBox="0 0 1080 420" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>
<rect x="10" y="40" width="190" height="96" rx="12" fill="#e0f2fe" stroke="#0284c7" stroke-width="2"/>
<text x="105" y="74" text-anchor="middle" font-size="16" font-weight="700" fill="#075985">User question</text>
<text x="105" y="98" text-anchor="middle" font-size="12.5" fill="#0c4a6e">"Is each dataset</text>
<text x="105" y="115" text-anchor="middle" font-size="12.5" fill="#0c4a6e">classification or regression?"</text>
<rect x="260" y="20" width="250" height="140" rx="12" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
<text x="385" y="52" text-anchor="middle" font-size="16" font-weight="700" fill="#92400e">Agent loop (create_agent)</text>
<text x="385" y="78" text-anchor="middle" font-size="13" fill="#78350f">LLM: Gemini · OpenAI · offline</text>
<text x="385" y="100" text-anchor="middle" font-size="12.5" fill="#78350f">reason → call tool → observe</text>
<text x="385" y="120" text-anchor="middle" font-size="12.5" fill="#78350f">→ repeat → answer</text>
<text x="385" y="144" text-anchor="middle" font-size="11.5" fill="#92400e">(the lab's AgentExecutor / ReAct)</text>
<rect x="600" y="10" width="470" height="230" rx="12" fill="#f3e8ff" stroke="#9333ea" stroke-width="2"/>
<text x="835" y="40" text-anchor="middle" font-size="16" font-weight="700" fill="#6b21a8">6 tools (@tool)</text>
<g class="mono" font-size="12.5" fill="#581c87" style="white-space:pre" xml:space="preserve">
<text x="620" y="70">1 list_csv_files</text><text x="862" y="70" fill="#7e22ce">what data is there?</text>
<text x="620" y="96">2 preload_datasets</text><text x="862" y="96" fill="#7e22ce">load CSVs into cache</text>
<text x="620" y="122">3 get_dataset_summaries</text><text x="862" y="122" fill="#7e22ce">columns, dtypes, uniques</text>
<text x="620" y="148">4 call_dataframe_method</text><text x="862" y="148" fill="#7e22ce">head/describe/info/...</text>
<text x="620" y="174">5 evaluate_classification</text><text x="862" y="174" fill="#7e22ce">accuracy, F1, baseline</text>
<text x="620" y="200">6 evaluate_regression</text><text x="862" y="200" fill="#7e22ce">R², MSE, MAE</text>
<text x="620" y="226" font-size="11.5" fill="#7e22ce">read-only allow-list · errors returned as data</text></g>
<rect x="600" y="290" width="220" height="110" rx="12" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="710" y="322" text-anchor="middle" font-size="15" font-weight="700" fill="#166534">DatasetStore cache</text>
<text x="710" y="346" text-anchor="middle" font-size="12.5" fill="#14532d">{"house-prices.csv": DataFrame,</text>
<text x="710" y="364" text-anchor="middle" font-size="12.5" fill="#14532d"> "customer-churn.csv": ...}</text>
<text x="710" y="386" text-anchor="middle" font-size="11.5" fill="#166534">loaded once, referenced by name</text>
<rect x="850" y="290" width="220" height="110" rx="12" fill="#e2e8f0" stroke="#475569" stroke-width="2"/>
<text x="960" y="322" text-anchor="middle" font-size="15" font-weight="700" fill="#1e293b">data/*.csv + sklearn</text>
<text x="960" y="348" text-anchor="middle" font-size="12.5" fill="#334155">pandas · get_dummies · dropna</text>
<text x="960" y="368" text-anchor="middle" font-size="12.5" fill="#334155">RandomForest classifier /</text>
<text x="960" y="386" text-anchor="middle" font-size="12.5" fill="#334155">regressor, 80/20 split</text>
<rect x="260" y="290" width="250" height="110" rx="12" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="385" y="322" text-anchor="middle" font-size="16" font-weight="700" fill="#166534">Plain-language answer</text>
<text x="385" y="348" text-anchor="middle" font-size="12.5" fill="#14532d">"churned → classification, 74% acc.</text>
<text x="385" y="366" text-anchor="middle" font-size="12.5" fill="#14532d">price_k → regression, R² 0.88"</text>
<line x1="200" y1="88" x2="256" y2="88" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<path d="M510,70 C550,60 565,60 596,62" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<text x="522" y="54" font-size="11.5" fill="#334155">tool call</text>
<path d="M600,150 C560,150 545,130 514,128" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<text x="524" y="168" font-size="11.5" fill="#334155">result (JSON)</text>
<line x1="710" y1="240" x2="710" y2="286" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<line x1="820" y1="345" x2="846" y2="345" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<line x1="385" y1="160" x2="385" y2="286" stroke="#16a34a" stroke-width="2" marker-end="url(#a)"/>
<text x="395" y="230" font-size="11.5" fill="#166534">no more tool calls</text>
</svg></div></body></html>"""


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.getenv("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        shoot(page, ARCHITECTURE, "architecture.png", ".card")

        py = sys.executable
        shots = [
            ("tools-direct.png", "Tools called directly (no LLM)", f"{py} examples/tools_demo.py", "python examples/tools_demo.py"),
            ("agent-trace-analysis.png", "Agent trace: classification or regression?",
             'data-agent "Is each dataset a classification or a regression problem? Train and evaluate a model for each." -p offline -t', None),
            ("agent-trace-inspect.png", "Agent trace: summaries and inspections",
             'data-agent "Tell me about the datasets" -p offline -t && data-agent "How many missing values does house-prices.csv have?" -p offline -t', None),
            ("classic-agent-executor.png", "The lab's pattern: one-step agent vs AgentExecutor (langchain-classic)",
             f"{py} examples/classic_agent_executor.py offline", "python examples/classic_agent_executor.py offline"),
            ("quickstart.png", "Quickstart example", f"{py} examples/quickstart.py offline", "python examples/quickstart.py offline"),
            ("pytest-results.png", "Test suite", f"{py} -m pytest -v --color=yes -p no:cacheprovider", "pytest -v"),
            ("pytest-coverage.png", "Coverage",
             f"{py} -m pytest -q --color=yes -p no:cacheprovider --cov=data_agent --cov-report=term-missing",
             "pytest --cov=data_agent --cov-report=term-missing"),
        ]
        for name, title, cmd, shown in shots:
            shoot(page, terminal_html(shown or cmd, run(cmd), title), name)

        nb_html = run(f"{py} -m jupyter nbconvert --to html --stdout notebooks/walkthrough.ipynb 2>/dev/null")
        page.set_viewport_size({"width": 1100, "height": 900})
        page.set_content(nb_html, wait_until="networkidle")
        page.screenshot(path=str(OUT / "notebook-walkthrough.png"), full_page=True)
        print("  ✓ notebook-walkthrough.png")
        browser.close()


if __name__ == "__main__":
    main()
