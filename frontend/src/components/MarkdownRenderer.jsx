import { useState } from 'react'
import {
  INLINE_TOKEN_PATTERN_SOURCE,
  parseCitationToken,
  prepareMarkdownWithSources,
  tokenizeCodeLine,
} from '../utils/chatHelpers'

const INLINE_PATTERN = new RegExp(INLINE_TOKEN_PATTERN_SOURCE, 'gi')

const HTML_BR_RE = /<br\s*\/?>/gi

function renderPlainWithBreaks(text, keyPrefix) {
  if (!text) return null
  const segments = text.split(HTML_BR_RE)
  if (segments.length === 1) {
    return text
  }

  const nodes = []
  segments.forEach((segment, idx) => {
    if (segment) {
      nodes.push(segment)
    }
    if (idx < segments.length - 1) {
      nodes.push(<br key={`${keyPrefix}-br-${idx}`} />)
    }
  })
  return nodes
}

function renderInline(text) {
  if (!text) return null
  const parts = []
  let lastIndex = 0

  for (const match of text.matchAll(INLINE_PATTERN)) {
    const idx = match.index ?? 0
    if (idx > lastIndex) {
      parts.push(
        renderPlainWithBreaks(text.slice(lastIndex, idx), `pre-${idx}`)
      )
    }

    const token = match[0]
    const key = `${idx}-${token.slice(0, 12)}`

    if (token.startsWith('`') && token.endsWith('`')) {
      parts.push(
        <code key={key} className="md-inline-code">
          {token.slice(1, -1)}
        </code>
      )
    } else if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(<strong key={key}>{renderInline(token.slice(2, -2))}</strong>)
    } else if (token.startsWith('*') && token.endsWith('*')) {
      parts.push(<em key={key}>{renderInline(token.slice(1, -1))}</em>)
    } else if (token.includes('](')) {
      const linkMatch = token.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/)
      if (linkMatch) {
        parts.push(
          <a
            key={key}
            href={linkMatch[2]}
            target="_blank"
            rel="noopener noreferrer"
            className="md-link"
          >
            {linkMatch[1]}
          </a>
        )
      } else {
        parts.push(token)
      }
    } else {
      const { filename } = parseCitationToken(token)
      parts.push(
        <span key={key} className="md-citation-pill" title={filename}>
          <span className="md-citation-name">{filename}</span>
        </span>
      )
    }

    lastIndex = idx + token.length
  }

  if (lastIndex < text.length) {
    parts.push(renderPlainWithBreaks(text.slice(lastIndex), `tail-${lastIndex}`))
  }

  return parts
}

function CodeBlock({ lang, content }) {
  const [copied, setCopied] = useState(false)
  const lines = content.split('\n')

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }

  return (
    <div className="md-code-block">
      <div className="md-code-header">
        <span>{lang || 'code'}</span>
        <button type="button" className="md-copy-btn" onClick={handleCopy}>
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="md-code-pre">
        <code>
          {lines.map((line, lineIdx) => (
            <span key={lineIdx} className="md-code-line">
              {tokenizeCodeLine(line).map((tok, tokIdx) => (
                <span key={tokIdx} className={`tok-${tok.type}`}>
                  {tok.text}
                </span>
              ))}
              {lineIdx < lines.length - 1 ? '\n' : ''}
            </span>
          ))}
        </code>
      </pre>
    </div>
  )
}

export default function MarkdownRenderer({ content, showSources = true }) {
  const { blocks, sources } = prepareMarkdownWithSources(content)

  if (blocks.length === 0) {
    return <span>{renderInline(content)}</span>
  }

  return (
    <div className="md-root">
      {blocks.map((block, index) => {
        if (block.type === 'heading') {
          const Tag = `h${Math.min(4, Math.max(1, block.level))}`
          return (
            <Tag key={index} className={`md-heading md-h${block.level}`}>
              {renderInline(block.content)}
            </Tag>
          )
        }

        if (block.type === 'code') {
          return <CodeBlock key={index} lang={block.lang} content={block.content} />
        }

        if (block.type === 'table') {
          return (
            <div key={index} className="md-table-wrap">
              <table className="md-table">
                <thead>
                  <tr>
                    {block.headers.map((header, hIdx) => (
                      <th key={hIdx}>{renderInline(header)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {block.rows.map((row, rIdx) => (
                    <tr key={rIdx}>
                      {block.headers.map((_, cIdx) => (
                        <td key={cIdx}>{renderInline(row[cIdx] ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        }

        if (block.type === 'blockquote') {
          return (
            <blockquote key={index} className="md-blockquote">
              {renderInline(block.content)}
            </blockquote>
          )
        }

        if (block.type === 'list') {
          const ListTag = block.ordered ? 'ol' : 'ul'
          return (
            <ListTag key={index} className="md-list">
              {block.items.map((item, itemIdx) => (
                <li key={itemIdx}>{renderInline(item)}</li>
              ))}
            </ListTag>
          )
        }

        const lines = block.content.split('\n')
        return (
          <p key={index} className="md-paragraph">
            {lines.map((line, lIdx) => (
              <span key={lIdx}>
                {renderInline(line)}
                {lIdx < lines.length - 1 ? <br /> : null}
              </span>
            ))}
          </p>
        )
      })}

      {showSources && sources.length > 0 && (
        <div className="md-sources-section" aria-label="Sources">
          <strong className="md-sources-label">Sources</strong>
          <div className="md-sources-list">
            {sources.map((filename) => (
              <span key={filename} className="md-citation-pill" title={filename}>
                <span className="md-citation-name">{filename}</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
