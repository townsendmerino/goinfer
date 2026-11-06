- **Changes behavior: `--max-file-store-bytes` caps the memory that uploaded files and batch results hold, 1 GiB by default.** `POST /v1/files` could keep 256 files
  of 64 MiB each with nothing to delete them, about 16 GiB of RAM. Past the limit the oldest files go first and the newest upload, or the newest batch's output and error files, are always kept, so download a batch's results
  before uploading much more; a dropped file's id answers 404 and says why. `0` restores the old behavior, with only the 256-file limit.
