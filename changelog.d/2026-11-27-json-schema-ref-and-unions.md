- **Structured output takes `$ref`, nullable values and unions.** A `json_schema` (`response_format`, `/v1/responses`'
  `text.format`, `/v1/messages`' `output_config.format`, and `constrain.JSONSchema` in Go) now compiles `$defs` or
  `definitions` with a `$ref` into the same document, recursive ones included; `type` given as a list; and `anyOf` and
  `oneOf`, between alternatives a value's first character tells apart or between objects, with a `discriminator` or
  without. Those are the shapes Pydantic gives a nested model, an `Optional` field, a recursive model and a discriminated
  union, each of which was a 400 before. What the grammar cannot enforce is still an error that names it: `allOf`, `not`,
  `pattern`, `maxLength`, `maximum`, a `$ref` out of the document, a union it cannot split (two arrays with different
  `items`), and a `oneOf` whose alternatives are not shown to exclude each other. A schema that compiled before accepts
  the documents it did, with one exception: a `type` list beside `properties` was read as an object alone and is now read
  as written. **Tool calls are held tighter:** in a tool's parameters these keywords left a value free to be any JSON
  value and are now followed (`oneOf` as `anyOf`), so a parameter declared `["array", "null"]` takes an array or `null`
  and nothing else. `constrain.TemplateFromSchema` follows `$ref` and unions too, so a GLM-OCR extraction schema may use
  them. `docs/server.md` lists the subset under "What a `json_schema` may contain".
