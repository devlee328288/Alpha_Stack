/**
 * HTML 그림을 PNG 로 굽는다. **HTML 이 정본이고 PNG 는 산출물이다.**
 *
 *     node scripts/render_html.mjs docs/아키텍처/version1.2/시스템아키텍처.html
 *     node scripts/render_html.mjs <입력.html> [출력.png] [--width 1680] [--scale 2]
 *
 * ## 왜 mermaid 가 아닌가
 *
 * `.mmd` 는 그래프를 그리는 데는 좋지만 **카드 안에 부제와 배지를 넣는 자유 배치**를
 * 표현할 수 없다. 계층 아키텍처처럼 "무엇이 무엇으로 흐르나" 가 전부인 그림은
 * 계속 mermaid 로 두고(`계층아키텍처.mmd`), 발표용 구성도만 HTML 로 그린다.
 *
 * HTML 을 정본으로 두면 `git diff` 에 무엇이 바뀌었는지 그대로 보인다. PNG 만 커밋하면
 * 이력에 바이너리 덩어리만 쌓이고 **무엇이 달라졌는지 아무도 못 읽는다.**
 *
 * ## puppeteer 는 어디서 오나
 *
 * 이 저장소는 puppeteer 를 직접 의존하지 않는다. 그림 하나 굽자고 Chrome 을 통째로
 * 내려받게 만들 이유가 없다. 대신 이미 깔려 있는 `@mermaid-js/mermaid-cli` 가 들고 있는
 * 것을 빌려 쓴다 — 그쪽은 이미 이 저장소의 `.mmd` 를 굽는 데 쓰이고 있다.
 *
 * ⚠️ 둘 다 없으면 **무엇을 해야 하는지까지** 출력하고 멈춘다. 예외만 던지면
 *    받는 사람은 막다른 길에 서게 된다.
 */

import { execSync } from 'node:child_process'
import { createRequire } from 'node:module'
import { existsSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const require = createRequire(import.meta.url)

/** puppeteer 를 찾는다. 없으면 어떻게 깔면 되는지까지 말하고 멈춘다. */
async function loadPuppeteer() {
  try {
    return (await import('puppeteer')).default
  } catch {
    /* 아래에서 mermaid-cli 것을 빌린다 */
  }

  let npmRoot = ''
  try {
    npmRoot = execSync('npm root -g', { encoding: 'utf8' }).trim()
  } catch {
    npmRoot = ''
  }

  const borrowed = join(
    npmRoot, '@mermaid-js', 'mermaid-cli', 'node_modules', 'puppeteer', 'lib', 'cjs',
    'puppeteer', 'puppeteer.js',
  )
  if (npmRoot && existsSync(join(npmRoot, '@mermaid-js', 'mermaid-cli', 'node_modules', 'puppeteer'))) {
    const mod = require(join(npmRoot, '@mermaid-js', 'mermaid-cli', 'node_modules', 'puppeteer'))
    return mod.default ?? mod
  }

  console.error(
    'puppeteer 를 찾지 못했습니다.\n' +
    '  왜 필요한가: HTML 그림을 PNG 로 구우려면 브라우저가 있어야 합니다.\n' +
    '  할 일 (둘 중 하나):\n' +
    '    npm i -g @mermaid-js/mermaid-cli    # .mmd 도 이걸로 굽습니다. 권장\n' +
    '    npm i -g puppeteer\n' +
    `  찾아본 곳: ${borrowed}`,
  )
  process.exit(1)
}

function parseArgs(argv) {
  const positional = []
  const opts = { width: 1680, scale: 2 }
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--width') { opts.width = Number(argv[++i]) }
    else if (argv[i] === '--scale') { opts.scale = Number(argv[++i]) }
    else { positional.push(argv[i]) }
  }
  opts.input = positional[0]
  opts.output = positional[1] ?? (positional[0] ?? '').replace(/\.html?$/i, '.png')
  return opts
}

async function main() {
  const opts = parseArgs(process.argv.slice(2))
  if (!opts.input) {
    console.error('쓰는 법: node scripts/render_html.mjs <입력.html> [출력.png] [--width 1680] [--scale 2]')
    process.exit(1)
  }
  const input = resolve(opts.input)
  if (!existsSync(input)) {
    console.error(`입력 파일이 없습니다: ${input}`)
    process.exit(1)
  }

  const puppeteer = await loadPuppeteer()
  const browser = await puppeteer.launch({ headless: 'shell' })
  try {
    const page = await browser.newPage()
    // deviceScaleFactor 를 올려야 글자가 흐려지지 않는다. 발표 화면과 인쇄 양쪽에 쓴다.
    await page.setViewport({ width: opts.width, height: 900, deviceScaleFactor: opts.scale })
    await page.goto(pathToFileURL(input).href, { waitUntil: 'networkidle0' })
    // 웹폰트를 안 쓰지만, 시스템 폰트 치환이 끝나기 전에 찍으면 줄바꿈이 달라진다.
    await page.evaluate(() => document.fonts.ready)

    // 페이지가 실제로 차지한 높이만큼만 찍는다. fullPage 는 body 여백까지 담는다.
    const box = await page.evaluate(() => {
      const el = document.querySelector('.sheet') ?? document.body
      const r = el.getBoundingClientRect()
      return { width: Math.ceil(r.width), height: Math.ceil(r.height) }
    })
    await page.setViewport({
      width: box.width, height: box.height, deviceScaleFactor: opts.scale,
    })
    const output = resolve(opts.output)
    await page.screenshot({ path: output, type: 'png' })
    console.log(`${output}  (${box.width}×${box.height} · ×${opts.scale})`)
  } finally {
    await browser.close()
  }
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
