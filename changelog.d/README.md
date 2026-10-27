# Unreleased changes, one file each

A change users would notice gets one file here instead of an edit to `CHANGELOG.md`, so two pull requests that each add
an entry never conflict.

- **Name:** `YYYY-MM-DD-<slug>.md`, the date you write it and a few lowercase words (`2026-10-27-kv-share-flag.md`).
  Entries list newest first, by file name.
- **Content:** the entry exactly as it should read in the CHANGELOG: one `- **...**` bullet, continuation lines indented
  two spaces, ending in a newline. Name any new route, `serve` flag, model family or aikit pin it brings, as
  `gate changelog` checks (RELEASING.md, step 9).
- **Preview:** `python3 scripts/changelog.py render` prints every entry as the next release will show it.
- **At release:** `python3 scripts/changelog.py release vX.Y.Z --date YYYY-MM-DD` writes `CHANGELOG.md`'s `[vX.Y.Z]`
  section from these files and removes them (RELEASING.md).
