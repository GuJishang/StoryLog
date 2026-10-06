/**
 * 前端「剧情问答」视图端到端测试
 * 启动本地服务 -> 用系统 Chrome 打开页面 -> 点击问答 -> 提问 -> 校验渲染与引用
 */
const { spawn } = require('child_process');
const fs = require('fs');
const http = require('http');
const path = require('path');
const { chromium } = require('playwright');

const ROOT = path.resolve(__dirname, '..');
const PY = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
const PORT = Number(process.env.UI_TEST_PORT || 8093);
const BASE = `http://127.0.0.1:${PORT}`;
// 测试产出的截图放在 tests/.tmp/，README 用的截图由 tests/make_screenshots.cjs 单独生成
const SHOT_DIR = path.join(ROOT, 'tests', '.tmp');
const SHOT = path.join(SHOT_DIR, 'ui_rag.png');
const SHOT_AGENT = path.join(SHOT_DIR, 'ui_agent.png');

const PASS = [], FAIL = [];
function check(name, cond, detail = '') {
  (cond ? PASS : FAIL).push(name);
  console.log((cond ? 'PASS' : 'FAIL') + '  ' + name + (!cond && detail ? '  | ' + detail : ''));
}

function probe() {
  return new Promise((resolve) => {
    const req = http.get(BASE + '/api/index/status', (res) => { res.resume(); resolve(res.statusCode === 200); });
    req.on('error', () => resolve(false));
    req.setTimeout(1000, () => { req.destroy(); resolve(false); });
  });
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

(async () => {
  fs.mkdirSync(SHOT_DIR, { recursive: true });
  const env = { ...process.env };
  delete env.LLM_API_KEY;                       // 走检索降级路径，快速且确定
  const server = spawn(PY, ['server.py', String(PORT)], { cwd: ROOT, env, stdio: 'ignore' });

  let browser;
  try {
    let ready = false;
    for (let i = 0; i < 60; i++) { if (await probe()) { ready = true; break; } await sleep(300); }
    check('U0 服务就绪', ready);
    if (!ready) throw new Error('server not ready');

    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const errors = [];
    page.on('pageerror', e => errors.push('pageerror: ' + String(e)));
    page.on('response', r => { if (r.status() >= 400) errors.push(`HTTP ${r.status()} ${r.url()}`); });

    await page.goto(`${BASE}/index.html`, { waitUntil: 'load' });
    await page.waitForSelector('#app', { timeout: 10000 });

    check('U1 首页有问答入口', await page.locator('#btn-qa').count() === 1);
    const gameCards = await page.locator('.game-card').count();
    check('U1 游戏库渲染正常', gameCards === 3, `cards=${gameCards}`);

    await page.click('#btn-qa');
    await page.waitForSelector('.qa-index-bar', { timeout: 10000 });
    const bar = (await page.locator('.qa-index-bar').innerText()).replace(/\s+/g, ' ');
    check('U2 索引状态栏渲染', bar.length > 5, bar);
    check('U2 状态栏显示索引就绪', /索引就绪|正在重建|尚未建立|不可用/.test(bar), bar);
    check('U2 有筛选芯片', await page.locator('.qa-chip').count() === 4);
    check('U2 有示例问题', await page.locator('.qa-sample').count() >= 3);

    // 提问
    await page.fill('#qa-input', '原神第七章有哪些幕？');
    await page.click('#qa-send');
    await page.waitForSelector('.qa-card', { timeout: 60000 });
    await page.waitForFunction(() => !document.querySelector('.qa-loading'), null, { timeout: 180000 });

    const answer = await page.locator('.qa-answer-body').first().innerText();
    check('U3 返回答案文本', answer.trim().length > 10, answer.slice(0, 60));
    const cites = await page.locator('.qa-cite').count();
    check('U3 有引用卡片', cites > 0, `cites=${cites}`);
    const head = (await page.locator('.qa-card-head').innerText()).replace(/\s+/g, ' ');
    check('U3 显示模式与置信度', /置信度/.test(head), head);
    check('U3 降级模式标识正确', /仅检索|降级/.test(head), head);

    // 游戏筛选
    await page.locator('.qa-chip', { hasText: '战双帕弥什' }).click();
    const active = (await page.locator('.qa-chip.active').innerText()).trim();
    check('U4 游戏筛选可切换', active === '战双帕弥什', active);

    // 示例问题快速提问
    await page.locator('.qa-sample').first().click();
    await page.waitForSelector('.qa-loading', { timeout: 5000 }).catch(() => {});
    await page.waitForFunction(() => !document.querySelector('.qa-loading'), null, { timeout: 180000 });
    const a2 = await page.locator('.qa-answer-body').first().innerText();
    check('U5 示例问题可提问', a2.trim().length > 10, a2.slice(0, 60));

    // ---- Agent 模式（路径 D） ----
    // 本进程未注入 LLM_API_KEY，所以 Agent 会走「降级为单次 RAG」链路。
    // 要验证的正是：① 模式可切换；② 降级结果按 Agent 卡片渲染而非空卡片；③ 会话 id 复用。
    check('U8 有 Agent 模式切换', await page.locator('.qa-mode[data-mode="agent"]').count() === 1);
    await page.locator('.qa-mode[data-mode="agent"]').click();
    const agentActive = (await page.locator('.qa-mode.active').innerText()).trim();
    check('U8 Agent 模式可切换', agentActive === 'Agent 问答', agentActive);
    const hintA = (await page.locator('.qa-mode-hint').innerText()).trim();
    check('U8 Agent 提示可见', hintA.length > 4, hintA);

    // 归零上一轮遗留的游戏筛选，避免测试之间互相污染（U4 曾把筛选留在「战双」上）
    await page.locator('.qa-chip', { hasText: '全部' }).click();
    check('U8 游戏筛选已归零', (await page.locator('.qa-chip.active').innerText()).trim() === '全部');

    await page.fill('#qa-input', '钟离是谁？');
    await page.click('#qa-send');
    await page.waitForSelector('.qa-card', { timeout: 60000 });
    await page.waitForFunction(() => !document.querySelector('.qa-loading'), null, { timeout: 180000 });

    const aHead = (await page.locator('.qa-card-head').innerText()).replace(/\s+/g, ' ');
    check('U9 Agent 卡片含降级标识', /降级/.test(aHead), aHead);
    check('U9 Agent 卡片显示工具调用次数', /工具调用/.test(aHead), aHead);
    const aBody = (await page.locator('.qa-answer-body').first().innerText()).trim();
    check('U9 Agent 降级仍有答案', aBody.length > 10, aBody.slice(0, 60));
    const aNote = await page.locator('.qa-note').first().innerText().catch(() => '');
    check('U9 Agent 说明降级原因', /降级/.test(aNote), aNote);

    const sid1 = await page.evaluate(() => qaState.sessionId);
    check('U10 前端记录会话 id', typeof sid1 === 'string' && sid1.length > 0, String(sid1));

    await page.fill('#qa-input', '那战双的剧情呢？');
    await page.click('#qa-send');
    await page.waitForFunction(() => !document.querySelector('.qa-loading'), null, { timeout: 180000 });
    const sid2 = await page.evaluate(() => qaState.sessionId);
    check('U10 同会话复用 id（多轮记忆）', sid2 === sid1, `${sid1} vs ${sid2}`);

    // 回归：Agent 检索无命中（限定到没有该内容的游戏）时，仍应渲染卡片头部而不是空白卡片
    await page.locator('.qa-chip', { hasText: '战双帕弥什' }).click();
    await page.fill('#qa-input', '钟离是谁？');
    await page.click('#qa-send');
    await page.waitForFunction(() => !document.querySelector('.qa-loading'), null, { timeout: 180000 });
    const nhHead = await page.locator('.qa-card-head').count();
    check('U10b 无命中时仍保留卡片头部', nhHead === 1, `heads=${nhHead}`);
    const nhBody = (await page.locator('.qa-answer-body').first().innerText()).trim();
    check('U10b 无命中给出可读说明', nhBody.length > 5, nhBody.slice(0, 60));
    await page.locator('.qa-chip', { hasText: '全部' }).click();

    // 切回 RAG 模式，确认结果按 RAG 卡片渲染（置信度字段回归）
    await page.locator('.qa-mode[data-mode="rag"]').click();
    await page.fill('#qa-input', '钟离是谁？');
    await page.click('#qa-send');
    await page.waitForFunction(() => !document.querySelector('.qa-loading'), null, { timeout: 180000 });
    const rHead = (await page.locator('.qa-card-head').innerText()).replace(/\s+/g, ' ');
    check('U11 切回 RAG 后渲染置信度', /置信度/.test(rHead), rHead);

    // 返回
    await page.click('#qa-back');
    await page.waitForSelector('.game-card', { timeout: 10000 });
    check('U6 可返回游戏库', await page.locator('.game-card').count() === 3);

    check('U7 无 JS 运行时错误', errors.length === 0, errors.slice(0, 3).join(' | '));

    // 回到问答页截图（RAG 模式 + Agent 模式各一张）
    await page.click('#btn-qa');
    await page.waitForSelector('.qa-index-bar', { timeout: 10000 });
    await page.locator('.qa-mode[data-mode="agent"]').click();
    await page.fill('#qa-input', '钟离是谁？');
    await page.click('#qa-send');
    await page.waitForFunction(() => !document.querySelector('.qa-loading'), null, { timeout: 180000 });
    await sleep(800);   // 等 view transition 渐入动画结束再截图
    await page.screenshot({ path: SHOT_AGENT, fullPage: true });
    console.log('截图: ' + SHOT_AGENT);

    await page.locator('.qa-mode[data-mode="rag"]').click();
    await page.fill('#qa-input', '原神第七章有哪些幕？');
    await page.click('#qa-send');
    await page.waitForFunction(() => !document.querySelector('.qa-loading'), null, { timeout: 180000 });
    await sleep(800);
    await page.screenshot({ path: SHOT, fullPage: true });
    console.log('截图: ' + SHOT);
  } catch (e) {
    check('U-异常', false, String(e && e.message || e));
  } finally {
    if (browser) await browser.close();
    server.kill();
  }

  console.log(`\n${PASS.length} passed, ${FAIL.length} failed`);
  process.exit(FAIL.length ? 1 : 0);
})();
