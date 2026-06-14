// Visual QA pass: drive the production build (served by server.py on :8000)
// through Edge headless and screenshot every page, including a live Arena
// run with a PatchCore and a GAN variant. Console/page errors are collected
// and the script exits non-zero if any page fails to render.
import { chromium } from 'playwright';
import fs from 'node:fs';

const BASE = 'http://127.0.0.1:8000';
const OUT = 'visual-shots';
fs.mkdirSync(OUT, { recursive: true });

const errors = [];
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.on('console', (m) => {
  if (m.type() === 'error') errors.push(`[console ${page.url()}] ${m.text()}`);
});
page.on('pageerror', (e) => errors.push(`[pageerror ${page.url()}] ${e.message}`));

async function shot(name, opts = {}) {
  await page.waitForTimeout(opts.settle ?? 800);
  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: opts.full ?? false });
  console.log('shot:', name);
}

async function go(path) {
  await page.goto(BASE + path, { waitUntil: 'networkidle', timeout: 30000 });
}

// "mean ms" only appears in the final summary panel, never during the live run
const summaryChip = () => page.getByText('mean ms', { exact: false }).first();

// Home - let the count-up stats finish
await go('/');
await shot('01-home', { full: true, settle: 2600 });

// Evaluation Lab + metric switch re-sort
await go('/evaluation');
await shot('02-evaluation', { full: true, settle: 1600 });
const metricBtn = page.getByRole('button', { name: /auprc/i }).first();
if (await metricBtn.count()) {
  await metricBtn.click();
  await shot('03-evaluation-auprc', { settle: 1000 });
} else {
  console.log('note: no AUPRC switch found');
}

// Models gallery + first detail page
await go('/models');
await shot('04-models', { full: true, settle: 1200 });
const firstCard = page.locator('a[href^="/models/"]').first();
if (await firstCard.count()) {
  await firstCard.click();
  await page.waitForLoadState('networkidle');
  await shot('05-model-detail', { full: true, settle: 1200 });
}

// Arena - config with all 5 variants, then a live production run
await go('/arena?category=bottle&variant=production&n=25&seed=7');
await shot('06-arena-config', { full: true, settle: 1200 });
const ganCard = page.getByText(/OCGAN final/).first();
console.log('gan card visible:', await ganCard.count());

await page.getByRole('button', { name: /run batch/i }).click();
await summaryChip().waitFor({ timeout: 120000 });
await shot('07-arena-production-done', { full: true, settle: 1200 });

// GAN run (model should be warm in the server's single-slot cache)
await ganCard.click();
await page.getByRole('button', { name: /run batch/i }).click();
await summaryChip().waitFor({ state: 'detached', timeout: 15000 }).catch(() => {});
await summaryChip().waitFor({ timeout: 300000 });
await shot('09-arena-gan-done', { full: true, settle: 1200 });

// open a result modal on the last grid cell (first button imgs are the
// category thumbnails in the config panel) - fetches the GAN heatmap live
const cell = page.locator('main button img').last();
if (await cell.count()) {
  await cell.click();
  await shot('08-arena-result-modal', { settle: 4000 });
  await page.keyboard.press('Escape');
}

// Dataset explorer + category with mask overlay
await go('/dataset');
await shot('10-dataset', { full: true, settle: 1500 });
await go('/dataset/bottle');
await page.waitForTimeout(1500);
const defectTab = page.getByRole('button', { name: /broken_large/i }).first();
if (await defectTab.count()) await defectTab.click();
await shot('11-dataset-bottle', { full: true, settle: 1500 });

// Methodology
await go('/methodology');
await shot('12-methodology', { full: true, settle: 1000 });

await browser.close();

if (errors.length) {
  console.log('\nBROWSER ERRORS:');
  for (const e of errors) console.log(' -', e);
  process.exit(2);
}
console.log('\nVISUAL CHECK OK - no console/page errors');
