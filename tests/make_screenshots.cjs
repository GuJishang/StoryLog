/**
 * 采集 README 用的界面截图（本地服务 -> 无头 Chrome -> 截图到 assets/screenshots/）
 *
 * 用法：
 *   node tests/make_screenshots.cjs
 * 需要：npm i -D playwright，且本机已安装 Chrome（channel: 'chrome'）。
 */
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const ROOT = path.resolve(__dirname, '..');
const PY = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
const PORT = Number(process.env.SHOT_PORT || 8095);
const BASE = `http://127.0.0.1:${PORT}`;
const OUT = path.join(ROOT, 'assets', 'screenshots');

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function probe() {
  return new Promise((resolve) => {
    const req = http.get(BASE + '/api/index/status', (res) => { res.resume(); resolve(res.statusCode === 200); });
    req.on('error', () => resolve(false));
    req.setTimeout(1000, () => { req.destroy(); resolve(false); });
  });
}

async function shot(page, name) {
  const file = path.join(OUT, name);
  await page.screenshot({ path: file, fullPage: true });
  console.log('已保存 ' + file);
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  // 注意：不主动删 LLM_API_KEY —— 配了 Key 时 RAG/Agent 截图展示真实生成链路，
  // 没配 Key 时自动降级为纯检索模式，截图依然可用。
  const env = { ...process.env };
  const server = spawn(PY, ['server.py', String(PORT)], { cwd: ROOT, env, stdio: 'ignore' });

  let browser;
  try {
    let ready = false;
    for (let i = 0; i < 60; i++) { if (await probe()) { ready = true; break; } await sleep(300); }
    if (!ready) throw new Error('服务未就绪，请确认端口 ' + PORT + ' 未被占用');

    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
    await page.goto(`${BASE}/index.html`, { waitUntil: 'load' });
    await page.waitForSelector('.game-card', { timeout: 10000 });
    await sleep(700);
    await shot(page, 'view-library.png');            // 游戏库

    await page.locator('.game-card', { hasText: '鸣潮' }).first().click();
    await page.waitForSelector('.version-card', { timeout: 10000 });
    await page.locator('.version-card').first().click();
    await sleep(700);
    await shot(page, 'view-episodes.png');           // 版本 / 剧集

    // 视图之间没有统一的返回按钮，直接重新加载回游戏库最稳妥
    await page.goto(`${BASE}/index.html`, { waitUntil: 'load' });
    await page.waitForSelector('#btn-calendar', { timeout: 10000 });
    await sleep(500);

    await page.click('#btn-calendar');
    await sleep(900);
    await shot(page, 'view-calendar.png');           // 观剧日历

    // 同上，回到游戏库再进问答视图
    await page.goto(`${BASE}/index.html`, { waitUntil: 'load' });
    await page.waitForSelector('#btn-qa', { timeout: 10000 });

    // 问答：RAG 模式
    await page.click('#btn-qa');
    await page.waitForSelector('.qa-index-bar', { timeout: 10000 });
    if (await page.locator('.qa-mode[data-mode="rag"]').count()) {
      await page.locator('.qa-mode[data-mode="rag"]').click();
    }
    await page.fill('#qa-input', '原神第七章有哪些幕？');
    await page.click('#qa-send');
    await page.waitForFunction(() => !document.querySelector('.qa-loading'), null, { timeout: 180000 });
    await sleep(900);
    await shot(page, 'ui_rag.png');                  // 路径 C：RAG 问答

    // 问答：Agent 模式
    await page.locator('.qa-mode[data-mode="agent"]').click();
    await page.fill('#qa-input', '钟离是谁？');
    await page.click('#qa-send');
    await page.waitForFunction(() => !document.querySelector('.qa-loading'), null, { timeout: 180000 });
    await sleep(900);
    await shot(page, 'ui_agent.png');                // 路径 D：Agent 问答
  } catch (e) {
    console.error('截图失败：' + (e && e.message || e));
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
})();
