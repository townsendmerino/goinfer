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
//   - Links are live only for http:, https: and mailto:. Anything else (javascript:, data:, vbscript:,
//     file:, relative paths) is rendered as plain text, not as a link.
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
  const LINK_OK = /^(https?:|mailto:)/i;
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

  function inline(src) {
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
  const FENCE = /^ {0,3}(`{3,}|~{3,})\s*([^\s`]*)[^`]*$/;
  const HEADING = /^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$/;
  const HR = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
  const QUOTE = /^ {0,3}>[ ]?/;
  const ITEM = /^( *)([-*+]|\d{1,9}[.)])[ \t]+(.*)$/;
  const TABLE_SEP = /^ *\|? *:?-{1,}:? *(\| *:?-{1,}:? *)+\|? *$/;

  const isBlank = l => /^\s*$/.test(l);
  const startsBlock = l => FENCE.test(l) || HEADING.test(l) || HR.test(l) || QUOTE.test(l) || ITEM.test(l);

  function cells(line) {
    let s = line.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
    return s.split(/(?<!\\)\|/).map(c => c.trim());
  }

  function blocks(lines, into) {
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (isBlank(line)) { i++; continue; }

      let m;
      if ((m = FENCE.exec(line))) {
        const marker = m[1];
        const lang = LANG_OK.test(m[2]) ? m[2] : "";
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
        wrap.appendChild(head);
        const pre = el("pre");
        const code = el("code", lang ? "language-" + lang : "");
        code.textContent = body.join("\n");
        pre.appendChild(code);
        wrap.appendChild(pre);
        into.appendChild(wrap);
        continue;
      }
      if ((m = HEADING.exec(line))) {
        const h = el("h" + m[1].length);
        h.append(...inline(m[2]));
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
        blocks(inner, bq);
        into.appendChild(bq);
        continue;
      }
      if (ITEM.test(line)) { i = list(lines, i, into); continue; }
      if (line.includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
        i = table(lines, i, into);
        continue;
      }
      // paragraph: runs until a blank line or the start of another block
      const para = [];
      while (i < lines.length && !isBlank(lines[i]) && !(para.length && startsBlock(lines[i])) &&
             !(lines[i].includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1]) && para.length)) {
        para.push(lines[i].replace(/^ {0,3}/, ""));
        i++;
      }
      const p = el("p");
      p.append(...inline(para.join("\n")));
      into.appendChild(p);
    }
  }

  function list(lines, i, into) {
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
      blocks(content, li);
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
