import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildRequestHistory,
  parseMarkdownBlocks,
  parseSseChunk,
  prepareMarkdownWithSources,
  safeLoadChats,
  stripTableCellCitations,
  tokenizeCodeLine,
  UPLOAD_TIMEOUT_MS,
  uploadDocumentRequest,
} from '../src/utils/chatHelpers.js'

test('safeLoadChats handles malformed and valid localStorage payloads safely', () => {
  assert.deepEqual(safeLoadChats(null), [])
  assert.deepEqual(safeLoadChats('{invalid-json'), [])
  assert.deepEqual(safeLoadChats('{"not": "an array"}'), [])

  const loaded = safeLoadChats(
    JSON.stringify([
      null,
      { id: 1, title: '  ', messages: [{ role: 'user', content: 'Hi' }, { role: 'bad', content: 123 }] },
      { id: 2, title: 'Report', messages: [{ role: 'assistant', content: 'Err', isError: true }] },
    ])
  )

  assert.equal(loaded.length, 2)
  assert.equal(loaded[0].title, 'Untitled Chat')
  assert.deepEqual(loaded[0].messages, [{ role: 'user', content: 'Hi' }])
  assert.deepEqual(loaded[1].messages, [{ role: 'assistant', content: 'Err', isError: true }])
})

test('buildRequestHistory excludes error messages and trims to recent turns', () => {
  const messages = [
    { role: 'user', content: 'Q1' },
    { role: 'assistant', content: 'Failed', isError: true },
    { role: 'assistant', content: 'A1' },
    { role: 'user', content: 'Q2' },
    { role: 'assistant', content: 'A2' },
  ]

  const history = buildRequestHistory(messages, 3)
  assert.deepEqual(history, [
    { role: 'assistant', content: 'A1' },
    { role: 'user', content: 'Q2' },
    { role: 'assistant', content: 'A2' },
  ])
})

test('parseSseChunk parses split SSE chunks cleanly', () => {
  const events = []
  let buffer = ''

  buffer = parseSseChunk(buffer, 'data: {"type":"token","content":"Hel', (e) => events.push(e))
  assert.equal(events.length, 0)

  buffer = parseSseChunk(
    buffer,
    'lo "}\n\ndata: {"type":"done","response":"Hello world"}\n\n',
    (e) => events.push(e)
  )
  assert.equal(buffer, '')
  assert.deepEqual(events, [
    { type: 'token', content: 'Hello ' },
    { type: 'done', response: 'Hello world' },
  ])
})

test('parseMarkdownBlocks parses headings, lists, tables, blockquotes, and code', () => {
  const md = [
    '# Summary',
    '',
    'Paragraph with **bold** and [report.pdf, Page 2].',
    '',
    '- Item one',
    '- Item two',
    '',
    '| Metric | Value |',
    '| --- | --- |',
    '| Accuracy | 98% |',
    '',
    '> Cited note',
    '',
    '```python',
    'def greet():',
    '    return "hi" # ok',
    '```',
  ].join('\n')

  const blocks = parseMarkdownBlocks(md)
  assert.deepEqual(
    blocks.map((b) => b.type),
    ['heading', 'paragraph', 'list', 'table', 'blockquote', 'code']
  )
  assert.equal(blocks[0].level, 1)
  assert.deepEqual(blocks[2].items, ['Item one', 'Item two'])
  assert.deepEqual(blocks[3].headers, ['Metric', 'Value'])
  assert.deepEqual(blocks[3].rows, [['Accuracy', '98%']])
  assert.equal(blocks[5].lang, 'python')

  const tokens = tokenizeCodeLine('return "hi" # ok')
  assert.ok(tokens.some((t) => t.type === 'keyword' && t.text === 'return'))
  assert.ok(tokens.some((t) => t.type === 'string' && t.text === '"hi"'))
  assert.ok(tokens.some((t) => t.type === 'comment' && t.text === '# ok'))
})

test('uploadDocumentRequest handles success, errors, and browser timeout safely', async () => {
  assert.equal(UPLOAD_TIMEOUT_MS, 180000)

  const fakeFile = new Blob(['%PDF-1.4 test'], { type: 'application/pdf' })

  const okResult = await uploadDocumentRequest({
    apiBaseUrl: 'http://127.0.0.1:8000',
    file: fakeFile,
    timeoutMs: 500,
    fetchImpl: async () => ({
      ok: true,
      json: async () => ({ filename: 'report.pdf', chunks_created: 5, duplicate: false }),
    }),
  })
  assert.equal(okResult.chunks_created, 5)

  await assert.rejects(
    uploadDocumentRequest({
      apiBaseUrl: 'http://127.0.0.1:8000',
      file: fakeFile,
      timeoutMs: 500,
      fetchImpl: async () => ({
        ok: false,
        json: async () => ({ detail: 'Only PDF and DOCX files are allowed.' }),
      }),
    }),
    /Only PDF and DOCX files are allowed\./
  )

  await assert.rejects(
    uploadDocumentRequest({
      apiBaseUrl: 'http://127.0.0.1:8000',
      file: fakeFile,
      timeoutMs: 15,
      fetchImpl: (_url, { signal }) =>
        new Promise((_resolve, reject) => {
          signal.addEventListener('abort', () => {
            const err = new Error('The operation was aborted.')
            err.name = 'AbortError'
            reject(err)
          })
        }),
    }),
    /timed out in the browser.*finish indexing on the server/i
  )
})

test('prepareMarkdownWithSources strips Source columns and table citations while deduplicating filenames', () => {
  const md = [
    'Summary overview [intro.pdf].',
    '',
    '| Topic | Details | Source |',
    '| --- | --- | --- |',
    '| Latency | Low latency [bench.pdf], [guide.pdf] | [bench.pdf] |',
    '| Config | Uses `[keep_code.py]` properly [guide.pdf, Page 2]. | intro.pdf |',
    '',
    'Follow-up [Report.docx].',
  ].join('\n')

  const prepared = prepareMarkdownWithSources(md)
  assert.equal(prepared.hasTable, true)
  assert.deepEqual(prepared.sources, ['intro.pdf', 'bench.pdf', 'guide.pdf', 'Report.docx'])

  const tableBlock = prepared.blocks.find((b) => b.type === 'table')
  assert.deepEqual(tableBlock.headers, ['Topic', 'Details'])
  assert.deepEqual(tableBlock.rows, [
    ['Latency', 'Low latency'],
    ['Config', 'Uses `[keep_code.py]` properly.'],
  ])

  const cellResult = stripTableCellCitations('Metric 95% [a.pdf], [A.pdf].')
  assert.equal(cellResult.text, 'Metric 95%.')
  assert.deepEqual(cellResult.sources, ['a.pdf'])
})
