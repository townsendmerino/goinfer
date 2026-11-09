- **A JSON Schema string's `format` is now enforced for `date`, `time`, `date-time`, `uuid`, `ipv4` and `email`.**
  Constrained output (`response_format` `json_schema`, `constrain.JSONSchema`, `GrammarFromStruct`) can no longer write
  `"2026-13-01"` for a date or a malformed address for an email: the grammar holds the value to RFC 3339's full-date,
  full-time and date-time, RFC 4122's UUID text, a dotted-quad IPv4 and a dot-atom email. Any other `format` stays an
  annotation, as JSON Schema 2020-12 defines it. Before, every `format` was accepted and ignored.
