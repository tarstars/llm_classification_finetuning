"""Exercise the generated poster in Chrome; screenshots stay with private artifacts."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
ROOT = Path.home() / 'database/llm_classification_finetuning/eda/attempt_002/poster'
p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:8765/')
p.add_argument('--browser', default='/usr/bin/google-chrome')
a = p.parse_args()
out = ROOT / 'qa'
out.mkdir(exist_ok=True)
errors = []
checks = []

with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path=a.browser, headless=True, args=['--no-sandbox'])
    context = browser.new_context(viewport={'width': 1440, 'height': 1100}, accept_downloads=True)
    page = context.new_page()
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(a.url)
    page.wait_for_selector('.card')
    assert page.evaluate('DATA.cases.length') > 100
    assert page.evaluate('window.posterState.visible.length') == 20
    assert page.evaluate('filtered.every((c,i)=>!i||filtered[i-1].loss>=c.loss)')
    checks.append('initial rendering and descending calibrated loss')
    if page.evaluate("'train' in COHORTS"):
        assert page.evaluate('COHORTS.train.rows') == 2048
        assert not page.evaluate('DATA.preview')
        page.locator('#cohort').select_option('train')
        page.locator('#category').select_option('order_sensitive')
        assert page.evaluate('filtered.length') > 0
        assert page.evaluate('filtered.every(c=>c.diagnostic.order_flip)')
        page.locator('.card').first.get_by_text('Compare answer-order diagnostics', exact=True).click()
        assert page.locator('.card').first.locator('.diag-table tr').count() == 4
        page.locator('#reset').click()
        checks.append('complete training sample and paired-order diagnostics')
    page.screenshot(path=str(out / 'desktop.png'))

    page.locator('#cohort').select_option('test')
    assert page.evaluate("filtered.every(c=>c.cohort==='test')")
    assert abs(page.evaluate("COHORTS.test.metrics.log_loss") - .9623631122994142) < 1e-6
    page.locator('#category').select_option('truncated')
    assert page.evaluate("filtered.every(c=>c.tags.includes('truncated'))")
    assert page.locator('.omitted').count() > 0
    page.locator('#omit-toggle').uncheck()
    assert page.locator('body').evaluate("e=>e.classList.contains('hide-omitted')")
    page.locator('#omit-toggle').check()
    checks.append('cohort/category filters, confirmation calibration and omission toggle')

    page.locator('#reset').click()
    page.locator('#wrong-only').check()
    assert page.evaluate('filtered.every(c=>c.label!==c.prediction)')
    page.locator('#sort').select_option('confidence')
    assert page.evaluate('filtered.every((c,i)=>!i||filtered[i-1].confidence>=c.confidence)')
    page.locator('#reset').click()
    page.locator('#next').click()
    assert page.evaluate('window.posterState.page') == 1
    page.locator('#previous').click()
    assert page.evaluate('window.posterState.page') == 0
    checks.append('wrong-only, alternate sort and pagination')

    key = page.evaluate('window.posterState.visible[0]')
    rid = page.evaluate('filtered[0].id')
    page.locator('#search').fill(str(rid))
    assert page.evaluate('filtered.some(c=>c.id===Number(document.getElementById("search").value))')
    card = page.locator(f'[data-key="{key}"]')
    card.get_by_role('button', name='Expand full text', exact=True).click()
    assert card.evaluate("e=>e.classList.contains('expanded')")
    card.get_by_role('button', name='Collapse long text', exact=True).click()
    card.get_by_text('Inspect the exact input the model saw', exact=True).click()
    assert card.locator('.packed').count() == 2
    area = card.get_by_role('textbox', name=f'Review notes for row {rid}')
    area.fill('QA: check the instruction against both responses.')
    card.get_by_role('combobox', name=f'Review category for row {rid}').select_option('instruction')
    page.reload()
    page.locator('#search').fill(str(rid))
    assert page.locator(f'[data-key="{key}"] textarea').input_value().startswith('QA:')
    with page.expect_download() as download:
        page.locator('#export-notes').click()
    exported = json.loads(Path(download.value.path()).read_text())
    assert exported['notes'][key]['category'] == 'instruction'
    checks.append('search, full text, exact input, saved notes and JSON export')

    page.goto(a.url + '#case-' + key)
    assert page.locator(f'#case-{key}').count() == 1
    page.goto(a.url + '#case-nonexistent')
    assert page.locator('.card').count() > 0
    page.locator('a[href="#case-test-2313946712"]').evaluate('e=>e.click()')
    page.wait_for_selector('#case-test-2313946712')
    assert page.locator('#cohort').input_value() == 'test'
    checks.append('valid and invalid example deep links')

    page.set_viewport_size({'width': 390, 'height': 844})
    page.goto(a.url)
    page.wait_for_selector('.card')
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'), 'mobile horizontal overflow'
    page.screenshot(path=str(out / 'mobile.png'), full_page=False)
    checks.append('390px mobile layout without horizontal overflow')
    assert not errors, errors
    browser.close()

result = {'passed': True, 'checks': checks, 'javascript_errors': errors,
          'browser': a.browser, 'private_screenshots': str(out)}
(out / 'browser_checks.json').write_text(json.dumps(result, indent=2) + '\n')
(REPO / 'results/error_poster_002/browser_checks.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
