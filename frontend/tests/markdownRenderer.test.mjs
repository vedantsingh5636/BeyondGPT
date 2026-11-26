import assert from 'node:assert/strict'
import test from 'node:test'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

let MarkdownRenderer
let viteServer

test.before(async () => {
  viteServer = await createServer({
    server: { middlewareMode: true },
    appType: 'custom',
    optimizeDeps: { noDiscovery: true },
  })
  const mod = await viteServer.ssrLoadModule('/src/components/MarkdownRenderer.jsx')
  MarkdownRenderer = mod.default
})

test.after(async () => {
  if (viteServer) {
    await viteServer.close()
  }
})

test('MarkdownRenderer renders compact filename-only citation pills without page numbers', () => {
  const html = renderToStaticMarkup(
    React.createElement(MarkdownRenderer, {
      content:
        'See [very_long_architecture_specification_2026.pdf, Page 12], 【summary.docx, Page 3】, and [notes.pdf].',
    })
  )

  assert.match(html, /class="md-citation-pill"/)
  assert.doesNotMatch(html, /md-citation-page/)
  assert.doesNotMatch(html, /Page 12/)
  assert.doesNotMatch(html, /Page 3/)
  assert.match(
    html,
    /<span class="md-citation-pill" title="very_long_architecture_specification_2026\.pdf"><span class="md-citation-name">very_long_architecture_specification_2026\.pdf<\/span><\/span>/
  )
  assert.match(
    html,
    /<span class="md-citation-pill" title="summary\.docx"><span class="md-citation-name">summary\.docx<\/span><\/span>/
  )
  assert.match(
    html,
    /<span class="md-citation-pill" title="notes\.pdf"><span class="md-citation-name">notes\.pdf<\/span><\/span>/
  )
})

test('MarkdownRenderer converts <br> tags inside table cells into real line breaks', () => {
  const md = [
    '| Feature | Details |',
    '| --- | --- |',
    '| Ingestion | Line one<br>Line two<br />**Bold** line |',
    '| Inline Code | Literal `<br>` preserved |',
  ].join('\n')

  const html = renderToStaticMarkup(
    React.createElement(MarkdownRenderer, { content: md })
  )

  assert.doesNotMatch(html, /&lt;br\s*\/?&gt;Line/)
  assert.match(html, /Line one<br\/>Line two<br\/><strong>Bold<\/strong> line/)
  assert.match(html, /<code class="md-inline-code">&lt;br&gt;<\/code>/)
})

test('MarkdownRenderer preserves headings, lists, links, and syntax-highlighted code blocks', () => {
  const md = [
    '## Overview',
    '',
    '- Point with [Docs](https://example.com)',
    '',
    '```python',
    'def hello():',
    '    return "ok"',
    '```',
  ].join('\n')

  const html = renderToStaticMarkup(
    React.createElement(MarkdownRenderer, { content: md })
  )

  assert.match(html, /<h2 class="md-heading md-h2">Overview<\/h2>/)
  assert.match(html, /<a[^>]+href="https:\/\/example\.com"/)
  assert.match(html, /class="tok-keyword">def<\/span>/)
  assert.match(html, /class="tok-string">&quot;ok&quot;<\/span>/)
})

test('MarkdownRenderer removes table Source column and repeated row citations into compact Sources section', () => {
  const md = [
    '| Module | Responsibility | Source |',
    '| --- | --- | --- |',
    '| Retriever | Hybrid search [architecture.pdf] | [architecture.pdf] |',
    '| Ingestion | Parses PDFs [pdf_loader.pdf]<br>Chunks text [architecture.pdf] | [pdf_loader.pdf], notes.docx |',
  ].join('\n')

  const completeHtml = renderToStaticMarkup(
    React.createElement(MarkdownRenderer, { content: md })
  )

  assert.doesNotMatch(completeHtml, /<th>Source<\/th>/)
  assert.match(completeHtml, /<th>Module<\/th><th>Responsibility<\/th>/)
  assert.match(completeHtml, /<td>Hybrid search<\/td>/)
  assert.match(completeHtml, /<td>Parses PDFs<br\/>Chunks text<\/td>/)
  assert.match(completeHtml, /<div class="md-sources-section" aria-label="Sources">/)
  assert.match(completeHtml, /<strong class="md-sources-label">Sources<\/strong>/)

  // Each unique filename should appear exactly once in the rendered table response
  assert.equal(completeHtml.match(/>architecture\.pdf</g)?.length ?? 0, 1)
  assert.equal(completeHtml.match(/>pdf_loader\.pdf</g)?.length ?? 0, 1)
  assert.equal(completeHtml.match(/>notes\.docx</g)?.length ?? 0, 1)

  // Streaming preview hides the Sources footer until the response is complete
  const streamingHtml = renderToStaticMarkup(
    React.createElement(MarkdownRenderer, { content: md, showSources: false })
  )
  assert.doesNotMatch(streamingHtml, /md-sources-section/)
  assert.doesNotMatch(streamingHtml, /<th>Source<\/th>/)
})
