- **Tool calls are constrained for real agent tool lists.** A tool whose parameter schema used a keyword the grammar does not
  check (opencode's own `bash` and `read` use `maximum` and `exclusiveMinimum`), or an untyped property, used to drop the
  call constraint for every tool in the request, silently, so a malformed call could come back as text. Tool parameters
  now compile leniently: unchecked assertions are left to the tool, and an untyped or composite value may be any JSON value.
  A tool list that still cannot be constrained is reported on stderr. `response_format: json_schema` is unchanged.
