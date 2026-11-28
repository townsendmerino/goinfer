- **Structured output: three fixes at the edges.** A recursive `json_schema` in which some schema lies more than 64
  levels from the root by every path (two definitions of 40 levels each that refer to each other, entered through the
  first) is now a 400: the grammar never reached that schema, so a keyword it does not enforce there went unreported.
  `constrain.TemplateFromSchema` writes `[]`, not `{}`, where a recursive array leads back into itself through a `$ref`
  or a union. And `goinfer_confidence` no longer reports the first item of an array whose items have one option (a
  `const`, or an `enum` of one value): it was reported as an enum at confidence 1.0, against the rule that a value the
  schema forces is not reported.
