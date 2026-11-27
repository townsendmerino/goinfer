package main

import "github.com/townsendmerino/goinfer/internal/guard/gateidentity"

type identityAsset = gateidentity.IdentityAsset

var (
	_              identityAsset
	runIdentity    = gateidentity.RunIdentity
	replayIdentity = gateidentity.ReplayIdentity
)
