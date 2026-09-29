/* goinfer.dev: the machine picker, the ledger filters and the copy buttons. Every page is complete without this file;
   it only chooses which per-machine variant is showing, and hides ledger rows. */
(function () {
  "use strict";
  var root = document.documentElement, KEY = "gi-machine", VALID = ["mac", "cuda", "cpu", "other"];
  function $(s, c) { return (c || document).querySelector(s); }
  function $$(s, c) { return Array.prototype.slice.call((c || document).querySelectorAll(s)); }

  /* The fit verdict for a machine we have not measured (mirrors site/internal/site/model.go FitFor). */
  function otherFit(bytes, moe, gpu, mem, g) {
    var s = bytes / 1e9;
    if (g === "apple") {
      if (s <= mem * 0.7) return ["fits", gpu ? "" : "on the CPU"];
      if (moe && s < mem * 1.6) return ["tight", "pages experts from disk"];
      return ["no", ""];
    }
    if (g === "nv8" || g === "nv16") {
      var v = g === "nv8" ? 8 : 14;
      if (gpu && s <= v * 0.85) return ["fits", ""];
      if (gpu && moe && s < mem) return ["fits", "experts streamed to the card"];
      if (s <= mem * 0.6) return ["fits", "on the CPU"];
      if (s < mem * 0.9) return ["tight", "on the CPU"];
      return ["no", ""];
    }
    if (s <= mem * 0.6) return ["fits", ""];
    if (s <= mem * 0.9 || (moe && s < mem * 1.6)) return ["tight", moe ? "pages experts from disk" : ""];
    return ["no", ""];
  }
  function span(cls, text) { var e = document.createElement("span"); e.className = cls; e.textContent = text; return e; }
  function otherName() {
    var g = { apple: "Apple Silicon", nv8: "an 8 GB NVIDIA card", nv16: "a 12–16 GB NVIDIA card", none: "no GPU" }[$("#o-gpu").value];
    return $("#o-mem").value + " GB with " + g;
  }
  function fillOther() {
    var mem = +$("#o-mem").value, g = $("#o-gpu").value;
    $$(".fitslot").forEach(function (slot) {
      var t = $(".fitother", slot); if (!t) return;
      var f = otherFit(+slot.dataset.bytes, slot.dataset.moe === "1", slot.dataset.gpu === "1", mem, g);
      t.textContent = "";
      t.appendChild(span("fit " + f[0], f[0] === "no" ? "won't fit" : f[0]));
      if (f[1]) { t.appendChild(document.createTextNode(" ")); t.appendChild(span("fitnote", f[1])); }
      if (slot.dataset.on) { t.appendChild(document.createTextNode(" ")); t.appendChild(span("fitnote", "on " + otherName())); }
    });
  }
  function setMachine(m, store) {
    if (VALID.indexOf(m) < 0) m = "mac";
    root.dataset.machine = m;
    $$("#mseg button").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.m === m)); });
    var o = $("#other"); if (o) o.hidden = m !== "other";
    if (m === "other") fillOther();
    if (store) { try { localStorage.setItem(KEY, m); } catch (e) {} }
  }
  var start = root.dataset.machine || "mac";
  setMachine(start, false);
  $$("#mseg button").forEach(function (b) { b.addEventListener("click", function () { setMachine(b.dataset.m, true); }); });
  ["#o-mem", "#o-gpu"].forEach(function (s) { var e = $(s); if (e) e.addEventListener("change", fillOther); });

  /* the ledger filters (the Models page only) */
  var ledger = $("#ledger");
  if (ledger) {
    var F = { task: "any", proof: "any" }, rows = $$("li.lrow", ledger), total = rows.length;
    function apply() {
      var n = 0;
      rows.forEach(function (li) {
        var d = li.dataset, ok = true;
        if (F.task !== "any" && d.tasks.split(" ").indexOf(F.task) < 0) ok = false;
        if (F.proof !== "any" && d.tier !== F.proof) ok = false;
        if ($("#f-exp").checked && d.exp === "1") ok = false;
        if ($("#f-pull").checked && d.gguf !== "1") ok = false;
        if ($("#f-gpu").checked && d.gpu !== "1") ok = false;
        if ($("#f-moe").checked && d.moe !== "1") ok = false;
        if ($("#f-ck").checked && d.ck !== "1") ok = false;
        li.hidden = !ok; if (ok) n++;
      });
      $("#shown").textContent = "Showing " + n + " of " + total + " families";
      var e = $("#empty");
      e.textContent = F.task === "vision" ? "Nothing matches. Only two families take images today: Gemma 4 and Qwen2.5-VL." : "Nothing matches all of those. Loosen a filter.";
      e.hidden = n !== 0;
    }
    [["#f-task", "task"], ["#f-proof", "proof"]].forEach(function (p) {
      var seg = $(p[0]); if (!seg) return;
      $$("button", seg).forEach(function (b) {
        b.addEventListener("click", function () {
          F[p[1]] = b.dataset.v;
          $$("button", seg).forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
          apply();
        });
      });
    });
    ["#f-exp", "#f-pull", "#f-gpu", "#f-moe", "#f-ck"].forEach(function (s) { $(s).addEventListener("change", apply); });
  }

  /* copy buttons */
  $$(".copy").forEach(function (box) {
    var code = $("code", box), btn = $("button", box);
    if (!btn) return;
    btn.addEventListener("click", function () {
      var done = function () { btn.textContent = "copied"; setTimeout(function () { btn.textContent = "copy"; }, 1400); };
      function sel() { var r = document.createRange(); r.selectNodeContents(code); var s = getSelection(); s.removeAllRanges(); s.addRange(r); }
      try { navigator.clipboard.writeText(code.textContent).then(done, sel); } catch (e) { sel(); }
    });
  });
})();
