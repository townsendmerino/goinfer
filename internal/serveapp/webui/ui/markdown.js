"use strict";
// markdown.js — renders model output as Markdown by BUILDING DOM NODES, never by assembling HTML.
//
// WHY THIS FILE EXISTS AS IT DOES (docs/tasks/task-web-ui-2026-09.md W1, §6.2). The page is served
// from the API's own origin with the user's key in a field on it, and the text rendered here comes
// from a model — possibly one pulled from a stranger's Hugging Face repo minutes earlier. So:
//
//   - Every element is made with document.createElement from a FIXED allow-list, and all text goes
//     in through createTextNode. There is no innerHTML/outerHTML/insertAdjacentHTML anywhere in the
//     page; a Go test (TestWebUI_noHTMLStringSinks) fails the build if one appears.
//   - Raw HTML in the model's output is not interpreted. "<script>" is shown as the text "<script>".
//   - Links are live only for http: and https: followed by two slashes, and for mailto:. Anything else
//     (javascript:, data:, vbscript:, file:, relative paths, and "http:path", which a browser resolves against
//     this page's own address) is rendered as plain text, not as a link.
//   - Images are NEVER loaded. A Markdown image becomes a link to it. Loading one would make the
//     offline page reach the network, and a model could use it as a tracking pixel.
//
// Supported: paragraphs (single newlines kept as line breaks, as chat output expects), ATX headings,
// fenced code blocks (``` or ~~~, with a language label; an UNCLOSED fence renders as a code block so
// streaming output does not flicker), inline code, **strong**, *em*/_em_, ~~strike~~, links,
// bare http(s) URLs, blockquotes, ordered/unordered lists with nesting, GFM tables, horizontal rules,
// and backslash escapes. That covers what chat models actually emit; it is not a full CommonMark
// implementation, and malformed input degrades to text rather than to anything executable.
const Markdown = (() => {
  // The complete element set this renderer may create. The browser gate asserts the rendered DOM
  // never contains anything else.
  const TAGS = new Set(["p", "br", "strong", "em", "del", "code", "pre", "a", "ul", "ol", "li",
    "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "table", "thead", "tbody", "tr", "th",
    "td", "div", "span", "button"]);
  const LINK_OK = /^(https?:\/\/|mailto:)/i;
  const LANG_OK = /^[A-Za-z0-9_+#.-]{1,32}$/;

  function el(tag, cls) {
    if (!TAGS.has(tag)) throw new Error("markdown: refusing to create <" + tag + ">");
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    return e;
  }
  const text = s => document.createTextNode(s);

  // ---- inline ---------------------------------------------------------------------------------
  // Works on a list of segments, each a string (not yet parsed) or a Node (finished). Each pass
  // splits only the string segments, so text already turned into a node — the inside of a code span,
  // an escaped character — is never re-interpreted by a later pass.
  function pass(segs, re, make) {
    const out = [];
    for (const s of segs) {
      if (typeof s !== "string") { out.push(s); continue; }
      let last = 0;
      re.lastIndex = 0;
      let m;
      while ((m = re.exec(s)) !== null) {
        if (m[0] === "") { re.lastIndex++; continue; }
        if (m.index > last) out.push(s.slice(last, m.index));
        const node = make(m);
        if (Array.isArray(node)) out.push(...node); else out.push(node);
        last = m.index + m[0].length;
      }
      if (last < s.length) out.push(s.slice(last));
    }
    return out;
  }

  function linkNode(label, url) {
    const href = url.trim();
    if (!LINK_OK.test(href)) return null;
    const a = el("a");
    a.href = href;
    a.rel = "noopener noreferrer nofollow";
    a.target = "_blank";
    a.append(...label);
    return a;
  }

  // INLINE_MAX bounds one inline parse. An opener that never closes is searched for to the end of the text from every
  // opener, so one parse costs the square of its length. Longer text is parsed in pieces, cut after a line break or a
  // space where there is one; emphasis that would span a cut stays literal.
  const INLINE_MAX = 4096;

  function inline(src) {
    if (src.length <= INLINE_MAX) return inlinePiece(src);
    const out = [];
    for (let at = 0; at < src.length;) {
      let end = Math.min(at + INLINE_MAX, src.length);
      if (end < src.length) {
        const piece = src.slice(at, end);
        let cut = piece.lastIndexOf("\n");
        if (cut < 0) cut = piece.lastIndexOf(" ");
        if (cut >= 0) end = at + cut + 1;
      }
      out.push(...inlinePiece(src.slice(at, end)));
      at = end;
    }
    return out;
  }

  function inlinePiece(src) {
    let segs = [src];
    // backslash escapes become literal text nodes first, so "\*" can never open emphasis
    segs = pass(segs, /\\([\\`*_{}\[\]()#+\-.!~|>])/g, m => text(m[1]));
    // code spans: content is literal
    segs = pass(segs, /(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g, m => {
      const c = el("code");
      c.textContent = m[2].replace(/^ (.*) $/s, "$1");
      return c;
    });
    // images: never loaded — rendered as a link to the image, or plain text if the URL is unsafe
    segs = pass(segs, /!\[([^\]\n]*)\]\(\s*<?([^\s<>()]+)>?(?:\s+"[^"\n]*")?\s*\)/g, m => {
      const label = [text("image: " + (m[1] || m[2]))];
      return linkNode(label, m[2]) || text(m[0]);
    });
    // links
    segs = pass(segs, /\[([^\]\n]+)\]\(\s*<?([^\s<>()]+)>?(?:\s+"[^"\n]*")?\s*\)/g, m => {
      return linkNode(emphasis([m[1]]), m[2]) || [...emphasis([m[1]]), text(" (" + m[2] + ")")];
    });
    // bare URLs
    segs = pass(segs, /\bhttps?:\/\/[^\s<>()\[\]"']*[^\s<>()\[\]"'.,;:!?]/g, m => linkNode([text(m[0])], m[0]) || text(m[0]));
    segs = emphasis(segs);
    // remaining single newlines are line breaks
    segs = pass(segs, /\n/g, () => el("br"));
    return segs.map(s => typeof s === "string" ? text(s) : s);
  }

  function emphasis(segs) {
    const wrap = tag => m => { const e = el(tag); e.append(...emphasis([m[1] ?? m[2]]).map(s => typeof s === "string" ? text(s) : s)); return e; };
    segs = pass(segs, /\*\*(?=\S)([\s\S]*?\S)\*\*|(?<![A-Za-z0-9])__(?=\S)([\s\S]*?\S)__(?![A-Za-z0-9])/g, wrap("strong"));
    segs = pass(segs, /~~(?=\S)([\s\S]*?\S)~~/g, wrap("del"));
    segs = pass(segs, /(?<![*A-Za-z0-9])\*(?=[^\s*])([^*\n]*?[^\s*])\*(?![*A-Za-z0-9])|(?<![_A-Za-z0-9])_(?=[^\s_])([^_\n]*?[^\s_])_(?![_A-Za-z0-9])/g, wrap("em"));
    return segs;
  }

  // ---- blocks ---------------------------------------------------------------------------------
  // A fence, a heading and a table separator are read with index loops, not patterns: each is tested on every line of
  // every streamed frame, and a pattern with two adjacent runs that can share the same characters takes time in the
  // square of a long line. The patterns that remain are anchored and have no such pair.
  const isSpTab = c => c === " " || c === "\t";

  // fenceOf reads a code fence opener: up to three spaces, a run of three or more backticks or tildes, and an info
  // string with no backtick in it. It returns the run and the info string's first word, or null.
  function fenceOf(line) {
    let p = 0;
    while (p < 3 && line[p] === " ") p++;
    const c = line[p];
    if (c !== "`" && c !== "~") return null;
    let q = p;
    while (line[q] === c) q++;
    if (q - p < 3 || line.indexOf("`", q) !== -1) return null;
    return { marker: line.slice(p, q), lang: /^\s*(\S*)/.exec(line.slice(q))[1] };
  }

  // headingOf reads an ATX heading: up to three spaces, one to six #, a space or tab, the text, and an optional closing
  // run of # that a space or tab sets off from the text. It returns the level and the text, or null.
  function headingOf(line) {
    let p = 0;
    while (p < 3 && line[p] === " ") p++;
    let q = p;
    while (q - p < 6 && line[q] === "#") q++;
    if (q === p || !isSpTab(line[q])) return null;
    let a = q;
    while (isSpTab(line[a])) a++;
    let e = line.length;
    while (e > a && isSpTab(line[e - 1])) e--;
    let h = e;
    while (h > a && line[h - 1] === "#") h--;
    if (h < e && h > a && isSpTab(line[h - 1])) {
      e = h;
      while (e > a && isSpTab(line[e - 1])) e--;
    }
    return { level: q - p, text: line.slice(a, e) };
  }

  // isTableSep reads a table's separator row: two or more cells of dashes, each with an optional colon at either end
  // and spaces around it, between optional outer pipes.
  function isTableSep(line) {
    let a = 0, e = line.length;
    while (a < e && line[a] === " ") a++;
    while (e > a && line[e - 1] === " ") e--;
    if (line[a] === "|") a++;
    if (e > a && line[e - 1] === "|") e--;
    const cols = line.slice(a, e).split("|");
    return cols.length >= 2 && cols.every(c => {
      let x = 0, y = c.length;
      while (x < y && c[x] === " ") x++;
      while (y > x && c[y - 1] === " ") y--;
      return SEP_CELL.test(c.slice(x, y));
    });
  }
  const SEP_CELL = /^:?-+:?$/;

  const HR = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
  const QUOTE = /^ {0,3}>[ ]?/;
  const ITEM = /^( *)([-*+]|\d{1,9}[.)])[ \t]+(.*)$/;

  // MAX_DEPTH bounds how deep quotes and lists nest: each level is a call, and a line of fifty thousand ">" would
  // otherwise overflow the stack. Anything deeper is shown as text.
  const MAX_DEPTH = 64;

  const isBlank = l => /^\s*$/.test(l);
  const startsBlock = l => !!fenceOf(l) || !!headingOf(l) || HR.test(l) || QUOTE.test(l) || ITEM.test(l);

  function cells(line) {
    let s = line.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
    return s.split(/(?<!\\)\|/).map(c => c.trim());
  }

  function blocks(lines, into, depth = 0) {
    if (depth > MAX_DEPTH) {
      const p = el("p");
      p.append(text(lines.join("\n")));
      into.appendChild(p);
      return;
    }
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (isBlank(line)) { i++; continue; }

      let m;
      if ((m = fenceOf(line))) {
        const marker = m.marker;
        const lang = LANG_OK.test(m.lang) ? m.lang : "";
        const body = [];
        i++;
        const close = new RegExp("^ {0,3}" + marker[0] + "{" + marker.length + ",}\\s*$");
        while (i < lines.length && !close.test(lines[i])) body.push(lines[i++]);
        if (i < lines.length) i++; // consume the closing fence; an unclosed one runs to the end
        const wrap = el("div", "md-code");
        // Header row: the language label (if any) and a Copy button (W2). The button carries NO
        // handler — this block is rebuilt on every streamed frame, so app.js handles clicks by
        // delegation on the log instead of binding each button.
        const head = el("div", "md-code-head");
        if (lang) {
          const label = el("span", "md-lang");
          label.textContent = lang;
          head.appendChild(label);
        }
        const copy = el("button", "md-copy");
        copy.type = "button";
        copy.textContent = "Copy";
        copy.setAttribute("aria-label", "Copy code");
        head.appendChild(copy);
        // W25: a rendered preview, offered only for HTML — the one fenced language that is itself a
        // renderable document. Also handled by delegation (app.js); hidden while the reply is still
        // streaming (renderReply toggles it), since this whole block is rebuilt from scratch on every
        // frame and an open preview would just flicker away on the next token.
        if (lang === "html") {
          const preview = el("button", "md-preview");
          preview.type = "button";
          preview.textContent = "Preview";
          preview.setAttribute("aria-label", "Preview this code, sandboxed");
          head.appendChild(preview);
        }
        wrap.appendChild(head);
        const pre = el("pre");
        const code = el("code", lang ? "language-" + lang : "");
        code.textContent = body.join("\n");
        pre.appendChild(code);
        wrap.appendChild(pre);
        into.appendChild(wrap);
        continue;
      }
      if ((m = headingOf(line))) {
        const h = el("h" + m.level);
        h.append(...inline(m.text));
        into.appendChild(h);
        i++;
        continue;
      }
      if (HR.test(line)) { into.appendChild(el("hr")); i++; continue; }
      if (QUOTE.test(line)) {
        const inner = [];
        while (i < lines.length && !isBlank(lines[i]) && (QUOTE.test(lines[i]) || !startsBlock(lines[i]))) {
          inner.push(lines[i].replace(QUOTE, ""));
          i++;
        }
        const bq = el("blockquote");
        blocks(inner, bq, depth + 1);
        into.appendChild(bq);
        continue;
      }
      if (ITEM.test(line)) { i = list(lines, i, into, depth); continue; }
      if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        i = table(lines, i, into);
        continue;
      }
      // paragraph: runs until a blank line or the start of another block
      const para = [];
      while (i < lines.length && !isBlank(lines[i]) && !(para.length && startsBlock(lines[i])) &&
             !(lines[i].includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1]) && para.length)) {
        para.push(lines[i].replace(/^ {0,3}/, ""));
        i++;
      }
      const p = el("p");
      p.append(...inline(para.join("\n")));
      into.appendChild(p);
    }
  }

  function list(lines, i, into, depth) {
    const first = ITEM.exec(lines[i]);
    const indent = first[1].length;
    const ordered = /\d/.test(first[2]);
    const listEl = el(ordered ? "ol" : "ul");
    if (ordered) {
      const start = parseInt(first[2], 10);
      if (start !== 1 && start < 1e9) listEl.start = start;
    }
    let loose = false;
    const items = [];
    while (i < lines.length) {
      const m = ITEM.exec(lines[i]);
      if (!m || m[1].length !== indent || /\d/.test(m[2]) !== ordered) break;
      const content = [m[3]];
      const contentIndent = indent + m[2].length + 1;
      i++;
      while (i < lines.length) {
        const l = lines[i];
        if (isBlank(l)) {
          // a blank line stays inside the item only if the item continues after it
          const next = lines[i + 1];
          if (next !== undefined && (next.length - next.trimStart().length) >= contentIndent) {
            content.push("");
            loose = true;
            i++;
            continue;
          }
          break;
        }
        const lead = l.length - l.trimStart().length;
        const sib = ITEM.exec(l);
        if (sib && sib[1].length <= indent) break;
        if (lead < contentIndent && startsBlock(l) && !sib) break;
        content.push(lead >= contentIndent ? l.slice(contentIndent) : l.trimStart());
        i++;
      }
      items.push(content);
      // blank line between items makes the list loose
      if (i < lines.length && isBlank(lines[i])) {
        let j = i;
        while (j < lines.length && isBlank(lines[j])) j++;
        const nxt = j < lines.length ? ITEM.exec(lines[j]) : null;
        if (nxt && nxt[1].length === indent && /\d/.test(nxt[2]) === ordered) { loose = true; i = j; }
      }
    }
    for (const content of items) {
      const li = el("li");
      blocks(content, li, depth + 1);
      if (!loose && li.childNodes.length && li.firstChild.nodeName === "P") {
        const p = li.firstChild;
        while (p.firstChild) li.insertBefore(p.firstChild, p);
        p.remove();
      }
      listEl.appendChild(li);
    }
    into.appendChild(listEl);
    return i;
  }

  function table(lines, i, into) {
    const head = cells(lines[i]);
    const aligns = cells(lines[i + 1]).map(c => c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : c.startsWith(":") ? "left" : "");
    const t = el("table", "md-table");
    const thead = el("thead"), tr = el("tr");
    head.forEach((h, k) => { const th = el("th", aligns[k] ? "al-" + aligns[k] : ""); th.append(...inline(h)); tr.appendChild(th); });
    thead.appendChild(tr);
    t.appendChild(thead);
    const tbody = el("tbody");
    i += 2;
    while (i < lines.length && !isBlank(lines[i]) && lines[i].includes("|")) {
      const row = el("tr");
      cells(lines[i]).slice(0, head.length).forEach((c, k) => {
        const td = el("td", aligns[k] ? "al-" + aligns[k] : "");
        td.append(...inline(c));
        row.appendChild(td);
      });
      tbody.appendChild(row);
      i++;
    }
    t.appendChild(tbody);
    into.appendChild(t);
    return i;
  }

  // render replaces container's children with the rendered Markdown of src.
  function render(container, src) {
    const frag = document.createDocumentFragment();
    blocks(String(src).replace(/\r\n?/g, "\n").split("\n"), frag);
    container.replaceChildren(frag);
  }

  return { render, TAGS };
})();
