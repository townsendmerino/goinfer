package constrain_test

import (
	"fmt"

	"github.com/townsendmerino/goinfer/constrain"
)

// Derive a JSON Schema straight from a Go struct's json tags — the basis of "a
// struct the model cannot violate": GrammarFromStruct compiles this into the
// grammar that drives the logit mask, so constrained output always
// json.Unmarshals back into the struct.
func ExampleSchemaFromStruct() {
	type Person struct {
		Name  string `json:"name"`
		Age   int    `json:"age"`
		Email string `json:"email,omitempty"` // optional
	}
	schema, _ := constrain.SchemaFromStruct(Person{})
	fmt.Println(string(schema))
	// Output: {"additionalProperties":false,"properties":{"age":{"type":"integer"},"email":{"type":"string"},"name":{"type":"string"}},"required":["name","age"],"type":"object"}
}
