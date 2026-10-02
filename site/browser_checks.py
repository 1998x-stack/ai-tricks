"""真实 Chromium 浏览器验收；自行启动静态服务器，无模型或数据下载。"""
from __future__ import annotations

import argparse
import functools
import http.server
import threading
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def check(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}/ai-tricks/'
    errors = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('response', lambda response: errors.append(f'HTTP {response.status}: {response.url}') if response.status >= 400 else None)
            page.goto(base, wait_until='networkidle')
            expect(page.locator('h1')).to_contain_text('让每一次调参')
            page.screenshot(path=str(output/'home-desktop.png'), full_page=True)
            page.locator('[data-open-search]').click()
            expect(page.locator('#dialog-search')).to_be_focused()
            page.locator('#dialog-search').fill('梯度累积')
            expect(page.locator('#search-dialog .search-results li').first).to_be_visible()
            expect(page.locator('#search-dialog [role="status"]')).to_contain_text('找到')
            expect(page.locator('label[for="dialog-search"]')).to_have_text('术语、方法或实验编号')
            page.screenshot(path=str(output/'search-desktop.png'))
            page.keyboard.press('Escape')
            expect(page.locator('#search-dialog')).not_to_be_visible()
            expect(page.locator('[data-open-search]')).to_be_focused()

            page.goto(base+'experiments/', wait_until='networkidle')
            expect(page.locator('#experiment-rows tr:visible')).to_have_count(30)
            first = page.locator('#experiment-rows tr:visible').first.inner_text()
            page.locator('#next-page').click()
            assert page.locator('#experiment-rows tr:visible').first.inner_text() != first
            page.locator('#experiment-category').select_option('llm-sft')
            page.locator('#experiment-status').select_option('compatible')
            visible = page.locator('#experiment-rows tr:visible')
            assert visible.count() > 0
            assert visible.evaluate_all("rows => rows.every(r => r.dataset.category === 'llm-sft' && r.dataset.status === 'compatible')")
            page.reload(wait_until='networkidle')
            expect(page.locator('#experiment-category')).to_have_value('llm-sft')
            page.locator('#experiment-query').fill('definitely-no-experiment-xyz')
            expect(page.locator('#empty-experiments')).to_be_visible()
            page.locator('#reset-filters').click()
            expect(page.locator('#experiment-rows tr:visible')).to_have_count(30)
            page.screenshot(path=str(output/'experiments-desktop.png'), full_page=True)

            page.goto(base+'wiki/batch-size/gradient-accumulation.html', wait_until='networkidle')
            expect(page.locator('.prose pre')).not_to_have_count(0)
            page.screenshot(path=str(output/'wiki-desktop.png'), full_page=True)
            page.goto(base+'wiki/llm-sft/lora.html', wait_until='networkidle')
            expect(page.locator('math').first).to_be_visible()
            page.screenshot(path=str(output/'formulas-desktop.png'), full_page=True)
            for width in (390, 320):
                page.set_viewport_size({'width': width, 'height': 844})
                for route in ('', 'experiments/', 'wiki/batch-size/gradient-accumulation.html', 'methodology/', 'sources/', 'wiki/llm-sft/lora.html'):
                    page.goto(base+route, wait_until='networkidle')
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, route)
                    if width == 390:
                        name = route.strip('/').replace('/', '-') or 'home'
                        page.screenshot(path=str(output/f'{name}-mobile.png'), full_page=True)
                page.locator('.menu-toggle').click()
                expect(page.locator('#main-nav')).to_be_visible()
                expect(page.locator('.menu-toggle')).to_have_attribute('aria-expanded', 'true')
            page.goto(base+'search/?q=LoRA', wait_until='networkidle')
            expect(page.locator('#page-results li').first).to_be_visible()
            page.locator('#page-search').fill('<img src=x onerror=alert(1)>')
            expect(page.locator('#page-search-count')).to_contain_text('没有找到')
            expect(page.locator('#page-results img')).to_have_count(0)
            no_js = browser.new_context(java_script_enabled=False)
            fallback = no_js.new_page()
            fallback.goto(base+'experiments/', wait_until='networkidle')
            expect(fallback.locator('#experiment-rows tr:visible')).to_have_count(517)
            fallback.locator('#experiment-rows a').first.click()
            expect(fallback.locator('article')).to_be_visible()
            no_js.close()
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    print('Browser checks passed: search, filters, pagination, URL state, keyboard, 320/390px layouts, no-JS; no browser/HTTP errors.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.site-preview'))
    parser.add_argument('--output', type=Path, default=Path('site-test-results'))
    args = parser.parse_args()
    check(args.root, args.output)
