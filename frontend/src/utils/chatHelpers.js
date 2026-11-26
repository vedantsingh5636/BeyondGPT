const KEYWORDS = new Set([
  'async', 'await', 'break', 'case', 'catch', 'class', 'const', 'continue',
  'def', 'default', 'elif', 'else', 'export', 'extends', 'false', 'finally',
  'for', 'from', 'function', 'if', 'import', 'in', 'interface', 'let', 'none',
  'null', 'pass', 'raise', 'return', 'self', 'static', 'struct', 'switch',
  'throw', 'true', 'try', 'type', 'typeof', 'undefined', 'var', 'while',
  'with', 'yield', 'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE',
])

export const UPLOAD_TIMEOUT_MS = 180000

export async function uploadDocumentRequest({
  apiBaseUrl,
  file,
  fetchImpl = fetch,
  timeoutMs = UPLOAD_TIMEOUT_MS,
}) {
  const formData = new FormData()
  formData.append('file', file)

  const controller = new AbortController()
  let timedOut = false
  const timerId = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)

  try {
    const response = await fetchImpl(`${apiBaseUrl}/upload`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    })

    let data = null
    try {
      data = await response.json()
    } catch {
      data = null
    }

    if (!response.ok) {
      throw new Error(data?.detail || 'File upload failed.')
    }

    return data
  } catch (error) {
    if (timedOut || error?.name === 'AbortError') {
      throw new Error(
        'Upload request timed out in the browser after 3 minutes. Large documents may still finish indexing on the server — please verify before re-uploading.'
      )
    }
    throw error
  } finally {
    clearTimeout(timerId)
  }
}

export function safeLoadChats(rawValue) {
  if (!rawValue || typeof rawValue !== 'string') return []
  try {
    const parsed = JSON.parse(rawValue)
    if (!Array.isArray(parsed)) return []

    return parsed
      .filter((item) => item && typeof item === 'object' && item.id != null)
      .map((chat) => ({
        id: chat.id,
        title: typeof chat.title === 'string' && chat.title.trim() ? chat.title : 'Untitled Chat',
        messages: Array.isArray(chat.messages)
          ? chat.messages
              .filter(
                (msg) =>
                  msg &&
                  typeof msg === 'object' &&
                  (msg.role === 'user' || msg.role === 'assistant') &&
                  typeof msg.content === 'string'
              )
              .map((msg) => ({
                role: msg.role,
                content: msg.content,
                ...(msg.isError ? { isError: true } : {}),
              }))
          : [],
      }))
  } catch {
    return []
  }
}

export function buildRequestHistory(messages, maxMessages = 10) {
  if (!Array.isArray(messages)) return []
  return messages
    .filter(
      (msg) =>
        msg &&
        !msg.isError &&
        (msg.role === 'user' || msg.role === 'assistant') &&
        typeof msg.content === 'string' &&
        msg.content.trim()
    )
    .slice(-maxMessages)
    .map((msg) => ({ role: msg.role, content: msg.content }))
}

export function parseSseChunk(buffer, chunkText, onEvent) {
  const combined = buffer + chunkText
  const parts = combined.split(/\r?\n\r?\n/)
  const remainder = parts.pop() ?? ''

  for (const rawEvent of parts) {
    const dataLines = rawEvent
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trim())

    if (dataLines.length === 0) continue
    const dataStr = dataLines.join('\n')
    try {
      const parsed = JSON.parse(dataStr)
      onEvent(parsed)
    } catch {
      // Ignore malformed/keepalive SSE comment frames
    }
  }

  return remainder
}

const isTableRowLine = (line) => {
  const trimmed = line.trim()
  return trimmed.startsWith('|') && trimmed.endsWith('|') && trimmed.includes('|')
}

const isTableSeparatorLine = (line) =>
  /^\|\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)*\s*\|?$/.test(line.trim())

const splitTableRow = (line) => {
  const trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '')
  return trimmed.split(/(?<!\\)\|/).map((cell) => cell.replace(/\\\|/g, '|').trim())
}

export function parseMarkdownBlocks(markdown) {
  if (!markdown || typeof markdown !== 'string') return []
  const lines = markdown.replace(/\r\n/g, '\n').split('\n')
  const blocks = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]
    const trimmed = line.trim()

    if (!trimmed) {
      i += 1
      continue
    }

    // Fenced code block
    const fenceMatch = trimmed.match(/^```([\w.+-]*)\s*$/)
    if (fenceMatch) {
      const lang = fenceMatch[1] || ''
      const codeLines = []
      i += 1
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i])
        i += 1
      }
      if (i < lines.length && lines[i].trim().startsWith('```')) {
        i += 1
      }
      blocks.push({ type: 'code', lang, content: codeLines.join('\n') })
      continue
    }

    // Heading (# to ####)
    const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/)
    if (headingMatch) {
      blocks.push({
        type: 'heading',
        level: headingMatch[1].length,
        content: headingMatch[2].trim(),
      })
      i += 1
      continue
    }

    // GFM Table
    if (
      i + 1 < lines.length &&
      isTableRowLine(lines[i]) &&
      isTableSeparatorLine(lines[i + 1])
    ) {
      const headers = splitTableRow(lines[i])
      i += 2
      const rows = []
      while (i < lines.length && isTableRowLine(lines[i])) {
        rows.push(splitTableRow(lines[i]))
        i += 1
      }
      blocks.push({ type: 'table', headers, rows })
      continue
    }

    // Blockquote
    if (trimmed.startsWith('>')) {
      const quoteLines = []
      while (i < lines.length && lines[i].trim().startsWith('>')) {
        quoteLines.push(lines[i].trim().replace(/^>\s?/, ''))
        i += 1
      }
      blocks.push({ type: 'blockquote', content: quoteLines.join('\n') })
      continue
    }

    // Unordered list
    if (/^[-*+]\s+/.test(trimmed)) {
      const items = []
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        items.push(lines[i].trim().replace(/^[-*+]\s+/, ''))
        i += 1
      }
      blocks.push({ type: 'list', ordered: false, items })
      continue
    }

    // Ordered list
    if (/^\d+\.\s+/.test(trimmed)) {
      const items = []
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ''))
        i += 1
      }
      blocks.push({ type: 'list', ordered: true, items })
      continue
    }

    // Paragraph (stop at next special block)
    const paraLines = []
    while (i < lines.length) {
      const current = lines[i].trim()
      if (
        !current ||
        current.startsWith('```') ||
        /^#{1,4}\s+/.test(current) ||
        current.startsWith('>') ||
        /^[-*+]\s+/.test(current) ||
        /^\d+\.\s+/.test(current) ||
        (i + 1 < lines.length && isTableRowLine(lines[i]) && isTableSeparatorLine(lines[i + 1]))
      ) {
        break
      }
      paraLines.push(lines[i].trimEnd())
      i += 1
    }
    if (paraLines.length > 0) {
      blocks.push({ type: 'paragraph', content: paraLines.join('\n') })
    }
  }

  return blocks
}

const CODE_TOKEN_RE = /(#[^\n]*|\/\/[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b\d+(?:\.\d+)?\b|\b[A-Za-z_][A-Za-z0-9_]*\b)/g

export const CITATION_EXTENSIONS = 'pdf|docx?|txt|md|csv|xlsx?|pptx?|py|jsx?|tsx?'

export const CITATION_TOKEN_PATTERN_SOURCE =
  `[\\[【][^\\]】\\n]+\\.(?:${CITATION_EXTENSIONS})(?:\\s*[,·]\\s*Pages?\\s+(?:\\d+(?:\\s*[-–]\\s*\\d+)?|N\\/A))?[】\\]]`

export const INLINE_TOKEN_PATTERN_SOURCE =
  `(\`[^\`]+\`|\\*\\*[^*]+\\*\\*|\\*[^*]+\\*|\\[[^\\]]+\\]\\(https?:\\/\\/[^)\\s]+\\)|${CITATION_TOKEN_PATTERN_SOURCE})`

const CITATION_SCAN_PATTERN_SOURCE =
  `(\`[^\`]+\`|\\[[^\\]]+\\]\\(https?:\\/\\/[^)\\s]+\\)|${CITATION_TOKEN_PATTERN_SOURCE})`

const PLAIN_FILENAME_RE = new RegExp(`\\b[\\w.-]+\\.(?:${CITATION_EXTENSIONS})\\b`, 'gi')
const WHOLE_FILENAME_RE = new RegExp(`^[^<>|/\\[\\]\`]+?\\.(?:${CITATION_EXTENSIONS})$`, 'i')
const SOURCE_COLUMN_HEADER_RE =
  /^(?:sources?(?:\(s\))?|citations?(?:\(s\))?|source\s+(?:files?|docs?|documents?)|references?)$/i
const PAGE_SUFFIX_RE = /\s*[,·]\s*Pages?\s+(?:\d+(?:\s*[-–]\s*\d+)?|N\/A)/gi
const ONLY_SEPARATORS_OR_BREAKS_RE = /^(?:\s|<br\s*\/?>|[,;·•/])+$/i

const isMarkdownLinkToken = (token) => /^\[[^\]]+\]\(https?:\/\/[^)\s]+\)$/i.test(token)

export function parseCitationToken(token) {
  const inner = token.slice(1, -1).trim()
  const filename = inner
    .replace(/\s*[,·]\s*Pages?\s+(?:\d+(?:\s*[-–]\s*\d+)?|N\/A)$/i, '')
    .trim()
  return { filename }
}

const addUniqueSource = (sources, seenLower, rawName) => {
  const cleaned = (rawName || '').trim()
  if (!cleaned) return
  const key = cleaned.toLowerCase()
  if (seenLower.has(key)) return
  seenLower.add(key)
  sources.push(cleaned)
}

export function extractInlineCitations(text) {
  if (!text || typeof text !== 'string') return []
  const scanRe = new RegExp(CITATION_SCAN_PATTERN_SOURCE, 'gi')
  const sources = []
  const seenLower = new Set()

  for (const match of text.matchAll(scanRe)) {
    const token = match[0]
    if ((token.startsWith('`') && token.endsWith('`')) || isMarkdownLinkToken(token)) {
      continue
    }
    const { filename } = parseCitationToken(token)
    addUniqueSource(sources, seenLower, filename)
  }

  return sources
}

function extractSourcesFromSourceCell(cellText) {
  if (!cellText || typeof cellText !== 'string') return []
  const sources = []
  const seenLower = new Set()

  for (const filename of extractInlineCitations(cellText)) {
    addUniqueSource(sources, seenLower, filename)
  }

  const scanRe = new RegExp(CITATION_SCAN_PATTERN_SOURCE, 'gi')
  const remainder = cellText
    .replace(scanRe, (token) =>
      (token.startsWith('`') && token.endsWith('`')) || isMarkdownLinkToken(token) ? token : ' '
    )
    .replace(/`/g, '')
    .replace(PAGE_SUFFIX_RE, '')

  for (const part of remainder.split(/<br\s*\/?>|[;,\n]/i)) {
    const trimmed = part.trim()
    if (!trimmed) continue
    if (WHOLE_FILENAME_RE.test(trimmed)) {
      addUniqueSource(sources, seenLower, trimmed)
      continue
    }
    for (const fileMatch of trimmed.matchAll(PLAIN_FILENAME_RE)) {
      addUniqueSource(sources, seenLower, fileMatch[0])
    }
  }

  return sources
}

export function stripTableCellCitations(cellText) {
  if (!cellText || typeof cellText !== 'string') {
    return { text: '', sources: [] }
  }

  const scanRe = new RegExp(CITATION_SCAN_PATTERN_SOURCE, 'gi')
  const parts = []
  const sources = []
  const seenLower = new Set()
  let lastIndex = 0
  let removedAny = false

  for (const match of cellText.matchAll(scanRe)) {
    const idx = match.index ?? 0
    if (idx > lastIndex) {
      parts.push({ text: cellText.slice(lastIndex, idx), removed: false })
    }
    const token = match[0]
    if ((token.startsWith('`') && token.endsWith('`')) || isMarkdownLinkToken(token)) {
      parts.push({ text: token, removed: false })
    } else {
      const { filename } = parseCitationToken(token)
      addUniqueSource(sources, seenLower, filename)
      parts.push({ text: '', removed: true })
      removedAny = true
    }
    lastIndex = idx + token.length
  }

  if (lastIndex < cellText.length) {
    parts.push({ text: cellText.slice(lastIndex), removed: false })
  }

  if (!removedAny) {
    return { text: cellText.trim(), sources: [] }
  }

  for (let i = 1; i < parts.length - 1; i += 1) {
    if (!parts[i].removed && parts[i - 1].removed && parts[i + 1].removed) {
      if (/^[\s,;·•/]+$/.test(parts[i].text)) {
        parts[i].text = ''
      }
    }
  }

  for (let i = 0; i < parts.length; i += 1) {
    if (parts[i].removed) continue
    const nextPart = parts.slice(i + 1).find((p) => p.removed || p.text)
    if (nextPart?.removed) {
      const afterRemoved = parts.slice(i + 2).find((p) => !p.removed && p.text)
      if (!afterRemoved || /^(?:[.,;:!?*)\]]|<br\s*\/?>|\s)/i.test(afterRemoved.text)) {
        parts[i].text = parts[i].text.replace(/[ \t]+$/, '')
      }
    }
    const prevPart = i > 0 ? parts[i - 1] : null
    if (prevPart?.removed && i === 1) {
      parts[i].text = parts[i].text.replace(/^[\s,;·•/]+/, '')
    }
  }

  const joined = parts
    .map((p) => p.text)
    .join('')
    .replace(/[ \t]+(?=<br\s*\/?>)/gi, '')
    .replace(/(<br\s*\/?>)[ \t]+/gi, '$1')
    .replace(/[ \t]{2,}/g, ' ')
    .trim()

  return {
    text: ONLY_SEPARATORS_OR_BREAKS_RE.test(joined) ? '' : joined,
    sources,
  }
}

function cleanTableBlockCitations(tableBlock) {
  const { headers = [], rows = [] } = tableBlock
  const sources = []
  const seenLower = new Set()

  const sourceColIndexes = new Set()
  const columnExtracted = headers.map((_, colIdx) => {
    const colSources = []
    for (const row of rows) {
      colSources.push(...extractSourcesFromSourceCell(row[colIdx] ?? ''))
    }
    return colSources
  })

  headers.forEach((header, colIdx) => {
    const cleanHeader = header.replace(/[*_`]/g, '').trim()
    const isSourceHeader = SOURCE_COLUMN_HEADER_RE.test(cleanHeader)
    const remainingCols = headers.length - sourceColIndexes.size
    if (isSourceHeader && columnExtracted[colIdx].length > 0 && remainingCols > 1) {
      sourceColIndexes.add(colIdx)
      for (const file of columnExtracted[colIdx]) {
        addUniqueSource(sources, seenLower, file)
      }
    }
  })

  const cleanedHeaders = []
  const keptIndexes = []
  headers.forEach((header, colIdx) => {
    if (sourceColIndexes.has(colIdx)) return
    const stripped = stripTableCellCitations(header)
    for (const file of stripped.sources) {
      addUniqueSource(sources, seenLower, file)
    }
    cleanedHeaders.push(stripped.text)
    keptIndexes.push(colIdx)
  })

  const cleanedRows = rows.map((row) =>
    keptIndexes.map((colIdx) => {
      const stripped = stripTableCellCitations(row[colIdx] ?? '')
      for (const file of stripped.sources) {
        addUniqueSource(sources, seenLower, file)
      }
      return stripped.text
    })
  )

  return {
    block: { type: 'table', headers: cleanedHeaders, rows: cleanedRows },
    sources,
  }
}

export function prepareMarkdownWithSources(markdown) {
  const rawBlocks = parseMarkdownBlocks(markdown)
  const blocks = []
  const uniqueSources = []
  const seenLower = new Set()
  let hasTable = false

  for (const block of rawBlocks) {
    if (block.type === 'table') {
      hasTable = true
      const cleaned = cleanTableBlockCitations(block)
      blocks.push(cleaned.block)
      for (const file of cleaned.sources) {
        addUniqueSource(uniqueSources, seenLower, file)
      }
    } else if (block.type === 'heading' || block.type === 'paragraph' || block.type === 'blockquote') {
      blocks.push(block)
      for (const file of extractInlineCitations(block.content)) {
        addUniqueSource(uniqueSources, seenLower, file)
      }
    } else if (block.type === 'list') {
      blocks.push(block)
      for (const item of block.items) {
        for (const file of extractInlineCitations(item)) {
          addUniqueSource(uniqueSources, seenLower, file)
        }
      }
    } else {
      blocks.push(block)
    }
  }

  return {
    blocks,
    hasTable,
    sources: hasTable ? uniqueSources : [],
  }
}

export function tokenizeCodeLine(line) {
  if (!line) return [{ type: 'plain', text: '' }]
  const tokens = []
  let lastIndex = 0

  for (const match of line.matchAll(CODE_TOKEN_RE)) {
    const idx = match.index ?? 0
    if (idx > lastIndex) {
      tokens.push({ type: 'plain', text: line.slice(lastIndex, idx) })
    }
    const token = match[0]
    let type = 'plain'
    if (token.startsWith('#') || token.startsWith('//')) {
      type = 'comment'
    } else if (token.startsWith('"') || token.startsWith("'")) {
      type = 'string'
    } else if (/^\d/.test(token)) {
      type = 'number'
    } else if (KEYWORDS.has(token) || KEYWORDS.has(token.toLowerCase())) {
      type = 'keyword'
    }
    tokens.push({ type, text: token })
    lastIndex = idx + token.length
  }

  if (lastIndex < line.length) {
    tokens.push({ type: 'plain', text: line.slice(lastIndex) })
  }

  return tokens
}
