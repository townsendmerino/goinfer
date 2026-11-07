- **`goinfer-chat cache prune-sidecars` reclaims `.giw` sidecars in a format this build no longer reads.** A dry run by
  default: it lists each older-version sidecar under the pull cache (or `-dir <dir>`) and the space it holds; `-apply`
  deletes them and their `.verified` markers. The current version, a newer one, a legacy unversioned name and a transcode
  in progress are never touched.
