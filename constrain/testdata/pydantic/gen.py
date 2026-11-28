#!/usr/bin/env python3
"""Write the JSON Schemas Pydantic emits for the model shapes constrain's tests compile.

Each file is `Model.model_json_schema()` as Pydantic 2.12.4 printed it, untouched: `$defs` with local
`$ref`s for nested and recursive models, `anyOf` with a null alternative for an Optional field, and
`oneOf` with a `discriminator` for a tagged union.

    python3 constrain/testdata/pydantic/gen.py     # rewrites the .json files beside this script
"""
import json
import os
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class Address(BaseModel):
    street: str
    city: str


class Person(BaseModel):
    name: str
    address: Address
    shipping: Address = Field(description="where orders ship")


class Profile(BaseModel):
    name: str
    nickname: Optional[str] = None
    age: Optional[int] = None
    home: Optional[Address] = None


class Comment(BaseModel):
    text: str
    replies: list["Comment"] = []


class LinkedNode(BaseModel):
    value: int
    next: Optional["LinkedNode"] = None


class Cat(BaseModel):
    pet_type: Literal["cat"]
    meows: int


class Dog(BaseModel):
    pet_type: Literal["dog"]
    barks: float


class Owner(BaseModel):
    pet: Union[Cat, Dog] = Field(discriminator="pet_type")
    n: int


class Zoo(BaseModel):
    pets: list[Annotated[Union[Cat, Dog], Field(discriminator="pet_type")]]


class Untagged(BaseModel):
    pet: Union[Cat, Dog]


class Mixed(BaseModel):
    v: Union[int, float]
    w: Union[int, str]
    x: Literal["a", "b"]
    y: Optional[Literal["a", "b"]] = None
    z: Union[str, list[str]]


FIXTURES = {
    "nested_model": Person,
    "optional_field": Profile,
    "recursive_model": Comment,
    "recursive_optional": LinkedNode,
    "discriminated_union": Owner,
    "union_items": Zoo,
    "untagged_union": Untagged,
    "mixed_unions": Mixed,
}

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for name, model in FIXTURES.items():
        with open(os.path.join(here, name + ".json"), "w") as f:
            json.dump(model.model_json_schema(), f, indent=2)
            f.write("\n")
