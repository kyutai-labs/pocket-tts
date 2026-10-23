package org.pockettts.android.speech

/**
 * Turns Markdown into something worth listening to.
 *
 * Feeding raw Markdown to a TTS engine gets you "hash hash Getting started",
 * "asterisk asterisk important asterisk asterisk", and forty seconds of a URL
 * being spelled out. This strips the syntax and keeps the prose, and adds the
 * sentence punctuation that headings and list items normally get away without
 * so the voice falls at the end of them instead of running on.
 *
 * This is deliberately a plain-text pipeline rather than a full CommonMark
 * parse: it has to be forgiving of the half-Markdown people actually paste into
 * a scratchpad, where a stray asterisk is far more likely to be a typo than an
 * unclosed emphasis span.
 */
object MarkdownSpeech {

    data class Options(
        /**
         * Fenced code read aloud is almost never what you want, but "almost" is
         * doing real work there - reading a short snippet back to yourself is a
         * legitimate thing to want.
         */
        val speakCodeBlocks: Boolean = false,
        /** Spoken in place of a skipped code block. Empty means say nothing. */
        val codeBlockPlaceholder: String = "",
        /** URLs are unlistenable. Spoken in place of one; empty means drop it. */
        val linkPlaceholder: String = "",
    )

    private val HTML_COMMENT = Regex("""<!--.*?-->""", RegexOption.DOT_MATCHES_ALL)
    private val FENCE = Regex("""^ {0,3}(`{3,}|~{3,})\s*(\S*).*$""")
    private val ATX_HEADING = Regex("""^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$""")
    private val SETEXT_UNDERLINE = Regex("""^ {0,3}(=+|-+)\s*$""")
    private val THEMATIC_BREAK = Regex("""^ {0,3}([-*_])\s*(?:\1\s*){2,}$""")
    private val BLOCKQUOTE = Regex("""^ {0,3}(?:>\s?)+""")
    private val UNORDERED_ITEM = Regex("""^\s*[-+*]\s+""")
    private val ORDERED_ITEM = Regex("""^\s*(\d{1,9})[.)]\s+""")
    private val TASK_MARKER = Regex("""^\[[ xX]]\s+""")
    private val TABLE_DELIMITER =
        Regex("""^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$""")
    private val LINK_REFERENCE_DEF = Regex("""^ {0,3}\[[^]]+]:\s*\S+.*$""")

    /** A line break of any flavour; matched rather than normalised, to keep offsets true. */
    private val BREAK = Regex("""\r\n|\r|\n""")

    /** Private-use codepoint; stands in for a backslash-escaped character. */
    private const val SENTINEL = '\uE000'

    /**
     * Speakable text, and where each piece of it came from.
     *
     * Following along in the editor needs the second half. A chunk's offsets
     * are into the *stripped* text, and stripping deletes syntax, so those
     * offsets say nothing about the document the user is looking at. The map is
     * kept per block rather than per character: a block comes from a known run
     * of source lines, which is cheap to track and is the granularity a reader
     * cares about anyway.
     */
    data class Speakable(val text: String, val spans: List<Span>) {

        /**
         * Where [start] until [end] of [text] came from in the source, or null
         * if it cannot be placed.
         *
         * Narrowed to the exact sentence when that sentence survived stripping
         * unchanged, which is the common case for ordinary prose. A line that
         * was rewritten - emphasis removed, a link collapsed to its text -
         * falls back to the block it belongs to, which is still a useful thing
         * to highlight and is never a *wrong* thing to highlight.
         */
        fun sourceRange(start: Int, end: Int, source: String): IntRange? {
            val touched = spans.filter { it.speakableStart < end && it.speakableEnd > start }
            if (touched.isEmpty()) return null
            val from = touched.minOf { it.sourceStart }
            val to = touched.maxOf { it.sourceEnd }
            if (from >= to || to > source.length) return null

            val spoken = text.substring(start.coerceIn(0, text.length), end.coerceIn(0, text.length))
                .trim()
            if (spoken.isNotEmpty()) {
                val exact = source.indexOf(spoken, startIndex = from)
                if (exact in from until to) return exact until (exact + spoken.length)
            }
            return from until to
        }
    }

    /** One block of speakable text, and the source it was made from. */
    data class Span(
        val speakableStart: Int,
        val speakableEnd: Int,
        val sourceStart: Int,
        val sourceEnd: Int,
    )

    /**
     * @return speakable plain text, paragraphs separated by a blank line so a
     *   downstream chunker can turn them into pauses.
     */
    fun toSpeakable(markdown: String, options: Options = Options()): String =
        toSpeakableWithSource(markdown, options).text

    /** As [toSpeakable], but also reporting where each block came from. */
    fun toSpeakableWithSource(markdown: String, options: Options = Options()): Speakable {
        if (markdown.isBlank()) return Speakable("", emptyList())

        // Comments are blanked rather than removed, and lines are located by
        // scanning rather than by rewriting the string, because every offset
        // recorded below has to point into the document the user can see. A
        // replacement that changes length shifts everything after it.
        val blanked = HTML_COMMENT.replace(markdown) { " ".repeat(it.value.length) }
        val lines = mutableListOf<String>()
        val lineStarts = mutableListOf<Int>()
        var cursor = 0
        while (cursor <= blanked.length) {
            val breakAt = BREAK.find(blanked, cursor)
            val lineEnd = breakAt?.range?.first ?: blanked.length
            lines += blanked.substring(cursor, lineEnd)
            lineStarts += cursor
            if (breakAt == null) break
            cursor = breakAt.range.last + 1
        }

        val out = Blocks()
        val paragraph = StringBuilder()
        var paragraphFrom = 0
        var paragraphTo = 0
        // Code is accumulated separately from prose: inside a fence the markup
        // characters are literal, so this text must not be run through the
        // inline stripper on its way out.
        val code = StringBuilder()
        var codeFrom = 0

        fun sourceEndOf(line: Int) = lineStarts[line] + lines[line].length

        fun emit(text: String, from: Int, to: Int) {
            val speakable = inline(text, options)
            if (speakable.isNotBlank()) out.add(speakable.ensureSentenceEnd(), from, to)
        }

        fun flushParagraph() {
            if (paragraph.isEmpty()) return
            val text = paragraph.toString().trim()
            paragraph.setLength(0)
            val speakable = inline(text, options)
            // A paragraph usually punctuates itself; only rescue the ones that
            // do not, so ordinary prose keeps whatever ending it had.
            if (speakable.isNotBlank()) {
                out.add(speakable.ensureSentenceEnd(), paragraphFrom, paragraphTo)
            }
        }

        var index = skipFrontMatter(lines)
        var fenceMarker: String? = null

        while (index < lines.size) {
            val raw = lines[index]
            val lineFrom = lineStarts[index]
            val lineTo = sourceEndOf(index)

            // Fenced code: the closing fence has to be at least as long as the
            // opening one and made of the same character, so a ``` block that
            // contains ~~~ survives intact.
            val fence = FENCE.matchEntire(raw)
            if (fenceMarker != null) {
                val marker = fenceMarker
                val closes = fence != null &&
                    fence.groupValues[1][0] == marker[0] &&
                    fence.groupValues[1].length >= marker.length &&
                    fence.groupValues[2].isEmpty()
                if (closes) {
                    fenceMarker = null
                    if (code.isNotEmpty()) {
                        out.add(code.toString().ensureSentenceEnd(), codeFrom, lineTo)
                        code.setLength(0)
                    }
                } else if (options.speakCodeBlocks) {
                    code.appendSpaced(raw.trim())
                }
                index++
                continue
            }
            if (fence != null) {
                flushParagraph()
                fenceMarker = fence.groupValues[1]
                codeFrom = lineFrom
                if (!options.speakCodeBlocks && options.codeBlockPlaceholder.isNotEmpty()) {
                    out.add(options.codeBlockPlaceholder.ensureSentenceEnd(), lineFrom, lineTo)
                }
                index++
                continue
            }

            val line = raw.replaceFirst(BLOCKQUOTE, "")

            if (line.isBlank()) {
                flushParagraph()
                index++
                continue
            }

            if (THEMATIC_BREAK.matches(line)) {
                flushParagraph()
                index++
                continue
            }

            if (LINK_REFERENCE_DEF.matches(line)) {
                index++
                continue
            }

            val heading = ATX_HEADING.matchEntire(line)
            if (heading != null) {
                flushParagraph()
                emit(heading.groupValues[2], lineFrom, lineTo)
                index++
                continue
            }

            // A setext underline is only an underline if there is paragraph
            // text directly above it to underline.
            val next = lines.getOrNull(index + 1)
            if (next != null && SETEXT_UNDERLINE.matches(next) && !THEMATIC_BREAK.matches(next)) {
                flushParagraph()
                emit(line, lineFrom, sourceEndOf(index + 1))
                index += 2
                continue
            }

            if (line.contains('|')) {
                // A table: a header row, then a delimiter row, then rows
                // until one has no pipe. Read the way a screen reader does,
                // not as a run of cells (issue #10).
                val delimiter = lines.getOrNull(index + 1)
                if (delimiter != null && TABLE_DELIMITER.matches(delimiter) && delimiter.contains('|')) {
                    flushParagraph()
                    val header = cellsOf(line, options)
                    var end = index + 2
                    val rows = mutableListOf<IndexedValue<List<String>>>()
                    while (end < lines.size && lines[end].contains('|') && lines[end].isNotBlank()) {
                        rows += IndexedValue(end, cellsOf(lines[end], options))
                        end++
                    }
                    val headerTo = sourceEndOf(index + 1)
                    val spoken = speakTable(header, rows.map { it.value })
                    for ((text, rowLine) in spoken.zip(listOf(-1) + rows.map { it.index })) {
                        when {
                            text == null -> {}
                            rowLine < 0 -> out.add(text, lineFrom, headerTo)
                            else -> out.add(text, lineStarts[rowLine], sourceEndOf(rowLine))
                        }
                    }
                    index = end
                    continue
                }
                if (TABLE_DELIMITER.matches(line) && line.contains('-')) {
                    index++
                    continue
                }
                if (line.trimStart().startsWith("|") && line.trimEnd().endsWith("|")) {
                    flushParagraph()
                    val cells = line.trim().trim('|').split('|')
                        .map { inline(it.trim(), options) }
                        .filter { it.isNotBlank() }
                    if (cells.isNotEmpty()) {
                        out.add(cells.joinToString(", ").ensureSentenceEnd(), lineFrom, lineTo)
                    }
                    index++
                    continue
                }
            }

            val ordered = ORDERED_ITEM.find(line)
            val unordered = if (ordered == null) UNORDERED_ITEM.find(line) else null
            val itemMatch = ordered ?: unordered
            if (itemMatch != null) {
                flushParagraph()
                val body = line.substring(itemMatch.value.length).replaceFirst(TASK_MARKER, "")
                // Keeping the ordinal is the point of an ordered list; a bullet
                // carries no information worth speaking.
                val prefix = ordered?.let { "${it.groupValues[1]}. " } ?: ""
                val speakable = inline(body, options)
                if (speakable.isNotBlank()) {
                    out.add((prefix + speakable).ensureSentenceEnd(), lineFrom, lineTo)
                }
                index++
                continue
            }

            if (paragraph.isEmpty()) paragraphFrom = lineFrom
            paragraphTo = lineTo
            paragraph.appendSpaced(line.trim())
            index++
        }
        flushParagraph()
        // An unterminated fence still has content worth speaking.
        if (code.isNotEmpty()) {
            out.add(code.toString().ensureSentenceEnd(), codeFrom, blanked.length)
        }

        return out.build()
    }

    /**
     * The widest table read row by row. Each row is read with its column
     * names, so four columns is already a long sentence; wider ones become a
     * stream of fragments with nothing to say where the listener is.
     */
    const val MAX_READ_COLUMNS = 4

    private fun cellsOf(line: String, options: Options): List<String> {
        var row = line.trim()
        if (row.startsWith("|")) row = row.substring(1)
        if (row.endsWith("|")) row = row.dropLast(1)
        return row.split('|').map { inline(it.trim(), options).trim() }
    }

    /**
     * What a table says, as a screen reader would put it: first how big it
     * is, then each row as its first cell followed by each other cell under
     * its column's name - "Alpha: before, one; after, two." A table wider than
     * [MAX_READ_COLUMNS] is announced with its column names and not read.
     *
     * The first element is the announcement; when the rows are read, one
     * follows per row, null for a row with nothing in it, so each stays
     * aligned with its source line.
     */
    internal fun speakTable(header: List<String>, rows: List<List<String>>): List<String?> {
        val columns = maxOf(header.size, rows.maxOfOrNull { it.size } ?: 0)
        val count = if (rows.size == 1) "1 row" else "${rows.size} rows"
        val names = header.filter { it.isNotBlank() }
        if (columns > MAX_READ_COLUMNS) {
            val headed = if (names.isEmpty()) "" else ", headed ${spokenList(names)}"
            return listOf("Table not read: $count, $columns columns$headed.")
        }
        val spoken = rows.map { cells ->
            val label = cells.getOrElse(0) { "" }
            val pairs = (1 until columns).mapNotNull { j ->
                val value = cells.getOrElse(j) { "" }
                val name = header.getOrElse(j) { "" }
                when {
                    value.isBlank() -> null
                    name.isBlank() -> value
                    else -> "$name, $value"
                }
            }
            when {
                label.isBlank() && pairs.isEmpty() -> null
                label.isBlank() -> pairs.joinToString("; ")
                pairs.isEmpty() -> label
                else -> "$label: ${pairs.joinToString("; ")}"
            }?.ensureSentenceEnd()
        }
        return listOf("Table, $count.") + spoken
    }

    /** "a", "a and b", "a, b and c". */
    private fun spokenList(items: List<String>): String =
        if (items.size <= 1) items.joinToString() else items.dropLast(1).joinToString(", ") + " and " + items.last()

    /** Collects blocks with the source range each was made from. */
    private class Blocks {
        private val texts = mutableListOf<String>()
        private val from = mutableListOf<Int>()
        private val to = mutableListOf<Int>()

        fun add(text: String, sourceStart: Int, sourceEnd: Int) {
            texts += text
            from += sourceStart
            to += sourceEnd
        }

        fun build(): Speakable {
            val joined = texts.joinToString(SEPARATOR)
            val spans = mutableListOf<Span>()
            var at = 0
            for (i in texts.indices) {
                spans += Span(at, at + texts[i].length, from[i], to[i])
                at += texts[i].length + SEPARATOR.length
            }
            return Speakable(joined, spans)
        }

        private companion object {
            const val SEPARATOR = "\n\n"
        }
    }

    /** Strips inline Markdown from a single run of text. */
    fun inline(text: String, options: Options = Options()): String {
        if (text.isEmpty()) return text

        // Backslash escapes are resolved first and parked behind a sentinel so
        // that a literal escaped asterisk is not then read back as emphasis.
        val escapes = mutableListOf<Char>()
        var s = Regex("""\\([\\`*_{}\[\]()#+\-.!>~|])""").replace(text) { m ->
            escapes += m.groupValues[1][0]
            "$SENTINEL${escapes.size - 1}$SENTINEL"
        }

        // Inline code. The backtick run is matched greedily-innermost so that a
        // double-backtick span containing a single backtick survives.
        s = Regex("""(`{1,3})([^`]|[^`].*?[^`])\1""").replace(s) { it.groupValues[2].trim() }

        s = Regex("""!\[([^]]*)]\([^)]*\)""").replace(s) { it.groupValues[1] }
        s = Regex("""\[([^]]+)]\([^)]*\)""").replace(s) { it.groupValues[1] }
        s = Regex("""\[\^[^]]+]""").replace(s, "")
        s = Regex("""\[([^]]+)]\[[^]]*]""").replace(s) { it.groupValues[1] }
        s = Regex("""<https?://[^>\s]+>""").replace(s, options.linkPlaceholder)
        s = Regex("""(?<![\w/])https?://\S+""").replace(s, options.linkPlaceholder)

        s = Regex("""~~(?=\S)(.+?)(?<=\S)~~""").replace(s) { it.groupValues[1] }
        s = Regex("""==(?=\S)(.+?)(?<=\S)==""").replace(s) { it.groupValues[1] }
        s = Regex("""\*\*\*(?=\S)(.+?)(?<=\S)\*\*\*""").replace(s) { it.groupValues[1] }
        s = Regex("""\*\*(?=\S)(.+?)(?<=\S)\*\*""").replace(s) { it.groupValues[1] }
        s = Regex("""\*(?=\S)([^*]+?)(?<=\S)\*""").replace(s) { it.groupValues[1] }
        // Underscore emphasis only outside words, so snake_case_names survive.
        s = Regex("""(?<!\w)__(?=\S)(.+?)(?<=\S)__(?!\w)""").replace(s) { it.groupValues[1] }
        s = Regex("""(?<!\w)_(?=\S)([^_]+?)(?<=\S)_(?!\w)""").replace(s) { it.groupValues[1] }

        s = Regex("""<br\s*/?>""", RegexOption.IGNORE_CASE).replace(s, " ")
        s = Regex("""</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>""").replace(s, "")

        s = s.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")

        s = Regex("""$SENTINEL(\d+)$SENTINEL""").replace(s) { m ->
            escapes.getOrElse(m.groupValues[1].toInt()) { ' ' }.toString()
        }

        return s.replace(Regex("""[ \t]+"""), " ").trim()
    }

    private fun StringBuilder.appendSpaced(text: String) {
        if (text.isEmpty()) return
        if (isNotEmpty()) append(' ')
        append(text)
    }

    private fun skipFrontMatter(lines: List<String>): Int {
        if (lines.firstOrNull()?.trim() != "---") return 0
        for (i in 1 until lines.size) {
            val trimmed = lines[i].trim()
            if (trimmed == "---" || trimmed == "...") return i + 1
        }
        return 0
    }

    private fun String.ensureSentenceEnd(): String {
        val trimmed = trimEnd()
        if (trimmed.isEmpty()) return trimmed
        return if (trimmed.last() in ".!?:;,…") trimmed else "$trimmed."
    }
}
