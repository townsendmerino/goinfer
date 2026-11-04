- **Every release asset carries a build attestation.** The release workflow attests each attached file (the binaries,
  the model-in-binary builds, the license and notice files) with GitHub's build provenance, signed keyless through
  Sigstore. `gh attestation verify <file> --repo townsendmerino/goinfer` confirms a download was built by this repo's
  release workflow from its tag. `checksums.txt` is unchanged.
