- **`pull` refuses a download that will not fit, before fetching anything.** It compares what is left to fetch (a
  resumed download counts only its tail) with the disk's free space and names both, instead of failing partway through
  a multi-gigabyte file. Where free space cannot be read, the pull proceeds.
