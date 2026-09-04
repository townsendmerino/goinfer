# Task — restyle the web chat UI: AmbientCSS, light surface, blue accent

**Status:** proposed, try-and-revert. Filed 2026-09-02.
**Venue:** any. No measurement, no hardware.
**Disposition:** this is a taste change, not a correctness one. Land it on a branch, look at it,
revert without ceremony if it doesn't work. Nothing downstream depends on it.

---

## Why

The current web UI uses a warm cream palette — `#faf9f5`, `#f0eee6`, `#e3e0d5` in
`demo/agent/cmd/agent-web/index.html:11-12` — which reads as Claude's colour scheme. goinfer is
not a Claude product and shouldn't look like one. Separately, a project with a visibility problem
gains something from not looking like every other tool in the category.

The proposed direction is **AmbientCSS** (https://github.com/kikkupico/ambientcss): a CSS library
that models directional lighting rather than flat fills — convex and concave surfaces, chamfered
or filleted edges, elevation, a key light and a fill light with settable direction, hue and
intensity.

It suits an inference engine specifically. The look is physical hardware — a control panel — which
is the right register for software whose whole pitch is that it runs locally on the user's own
machine, one binary, no cloud.

---

## What to build

Restyle the browser chat UI to a light ambient surface with a **bright blue** accent.

### Palette (starting point, not gospel)

```
surface       #e9ecf1     one background colour for everything
text primary  #28303a
text muted    #69737f     metadata, mono rows
accent        #0b6fd4     bright blue
highlight     rgba(255,255,255,0.92)      key light, top-left
shadow        rgba(146,158,176,0.42)      fill shadow, bottom-right
```

Blue was chosen over the warm alternative for a concrete reason, not only taste: at the accent
saturation that looked right, orange could not carry white text on a solid fill, so the primary
action had to stay a tinted outline. Blue at `#0b6fd4` takes white comfortably, which means the
send button can be the brightest thing on screen — a real fix for this style's main accessibility
weakness, not a cosmetic one.

### The four decisions that carry the look

These matter more than the accent, which is the cheapest thing to change later.

1. **One background colour throughout.** No cards on a different fill, no borders. Depth comes
   only from light direction. Introducing a second surface colour breaks the effect.
2. **Consistent light direction.** Key light top-left, shadow bottom-right, everywhere, no
   exceptions. Inconsistent lighting is what makes this style look broken rather than soft.
3. **Inset for input, raised for output.** What the user types into is recessed; what the engine
   produces sits proud. This replaces the coloured-bubble convention entirely and is most of what
   stops it reading as a chat-app clone.
4. **Runtime stats as a first-class element, not chrome.** Model, quantization, backend and tok/s
   in the header; token count, wall time and context use under each response, in mono. Nothing
   else in this category surfaces that, and it is the truest thing about this project.

### Scope

Chat UI only. Do **not** apply this to anything dense and text-heavy — benchmark tables, the
capability matrix, generated docs. The physical-panel treatment suits controls and chrome; it
actively hurts data.

---

## Constraints

**Vendor the CSS. Do not use the CDN.** AmbientCSS's own example loads `ambient.css` from
jsdelivr. A `serve -web` UI that fetches its stylesheet over the network breaks the offline claim
the README leads with — someone running an embedded model on a disconnected machine gets an
unstyled page. Pull the file into the repo and `go:embed` it with the rest of the web assets. This
also retires the dependency risk: once vendored, we own a CSS file, which is a mild thing to own.

**Check contrast before landing.** This style descends from neumorphism, whose standard criticism
is that low-contrast soft shadows on same-tone backgrounds fail WCAG AA — controls stop looking
like controls. The library exposes light intensity and highlight colour, so it should be tunable,
but verify rather than assume. The weakest elements are known in advance:

- the muted mono metadata rows (`#69737f` on `#e9ecf1`) — check at actual rendered size
- the inset user message, which has the least separation from the background of anything on
  screen. If it reads as too faint, **deepen the inner shadow rather than tinting the fill** —
  tinting breaks decision 1 above.

**Keep the diff revertible.** Both existing web UIs put all their CSS in a single `<style>` block
(`demo/agent/cmd/agent-web/index.html`, 347 lines; `demo/gemma-web/index.html`, 241 lines, dark
theme). Keep that structure. One commit, one file where practical, so a revert is `git revert` and
not an archaeology exercise.

**Restraint on the light settings.** AmbientCSS's demo values are tuned to show the effect off.
Dial the key and fill intensities down from their defaults. The house preference for public-facing
material is a humble register, and that applies to visual tone as much as to prose.

---

## First move

Identify the actual target file. The `serve -web` browser UI landed 2026-09-02 and is not in the
tree this doc was written against — the two files named above are the older demo UIs. Confirm
whether `serve -web` has its own HTML/CSS or reuses one of them, and restyle whichever is the one
users actually see. Say which you picked.

Then: vendor `ambient.css`, restyle one screen, look at it, and stop. Do not restyle everything
before the first look.

---

## Non-goals

- No dark mode in this pass. `demo/gemma-web/index.html` is already dark and can stay as it is;
  a light ambient surface and a dark one are two different tuning problems.
- No component library adoption (`@ambientcss/components`). CSS only.
- No build step. Whatever is done here must survive the no-toolchain constraint.
- No accent bikeshedding after landing. The accent is one hex value and trivially changed; the
  four decisions above are the expensive part.

---

## Acceptance

- The UI no longer uses the Claude cream palette.
- `ambient.css` is vendored and embedded; no network fetch at runtime, verified by loading the
  page with networking disabled.
- Contrast checked on the two weak elements named above, with the result recorded.
- The change is one revertible commit.
- Do not use the words "honest" or "honesty" in anything written for this task.

Leave uncommitted for review.
