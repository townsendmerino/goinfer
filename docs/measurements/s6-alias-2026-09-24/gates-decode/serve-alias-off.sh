#!/bin/bash
# S6 gate arm: aliasing off. Same binary for both arms; only GOINFER_METAL_ALIAS differs.
export GOINFER_METAL_ALIAS=0
exec /tmp/serve-s6gate "$@"
