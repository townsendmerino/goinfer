# build: gcc -O3 -mavx2 -fopenmp scripts/readbw.c -o readbw && ./readbw   (docs/measurements/cpu-decode-roofline-2026-09-23.md)
// AVX2 read-only DRAM bandwidth, this box's ceiling for a pure streaming read.
// build: gcc -O3 -mavx2 -fopenmp readbw.c -o readbw
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static inline __m256i rd(const uint8_t *p, int pf) {
    if (pf) _mm_prefetch((const char *)(p + pf), _MM_HINT_T0);
    __m256i a = _mm256_load_si256((const __m256i *)p);
    __m256i b = _mm256_load_si256((const __m256i *)(p + 32));
    __m256i c = _mm256_load_si256((const __m256i *)(p + 64));
    __m256i d = _mm256_load_si256((const __m256i *)(p + 96));
    return _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c, d));
}

static uint64_t sink;

// each thread reads its own contiguous slice
static double run_slice(const uint8_t *buf, size_t n, int threads, int pf) {
    double t0 = omp_get_wtime();
    uint64_t total = 0;
#pragma omp parallel num_threads(threads) reduction(^ : total)
    {
        int t = omp_get_thread_num();
        size_t per = (n / threads) & ~(size_t)127;
        const uint8_t *p = buf + per * t, *e = p + per;
        __m256i acc = _mm256_setzero_si256();
        for (; p < e; p += 128) acc = _mm256_xor_si256(acc, rd(p, pf));
        uint64_t v[4];
        _mm256_storeu_si256((__m256i *)v, acc);
        total ^= v[0] ^ v[1] ^ v[2] ^ v[3];
    }
    sink ^= total;
    return omp_get_wtime() - t0;
}

// chunks handed out round-robin like a column-sharded matmul (chunk bytes each)
static double run_chunks(const uint8_t *buf, size_t n, int threads, size_t chunk, int pf) {
    double t0 = omp_get_wtime();
    uint64_t total = 0;
    size_t nchunks = n / chunk;
#pragma omp parallel for num_threads(threads) schedule(static, 1) reduction(^ : total)
    for (size_t c = 0; c < nchunks; c++) {
        const uint8_t *p = buf + c * chunk, *e = p + chunk;
        __m256i acc = _mm256_setzero_si256();
        for (; p < e; p += 128) acc = _mm256_xor_si256(acc, rd(p, pf));
        uint64_t v[4];
        _mm256_storeu_si256((__m256i *)v, acc);
        total ^= v[0] ^ v[1] ^ v[2] ^ v[3];
    }
    sink ^= total;
    return omp_get_wtime() - t0;
}

static int cmp(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    size_t n = (size_t)2 << 30; // 2 GiB
    uint8_t *buf = aligned_alloc(2 << 20, n);
    memset(buf, 0x5a, n); // touch: real pages, THP where the kernel grants
    const int reps = 7;
    int tl[] = {1, 2, 4, 8, 12, 16};
    printf("%-34s", "config \\ threads");
    for (int i = 0; i < 6; i++) printf("%8d", tl[i]);
    printf("   (GB/s, median of %d, 2 GiB read)\n", reps);
    struct { const char *name; size_t chunk; int pf; } cfg[] = {
        {"contiguous slice/thread", 0, 0},
        {"contiguous slice + prefetch 512B", 0, 512},
        {"64 KiB chunks round-robin", 64 << 10, 0},
        {"4 KiB chunks round-robin", 4 << 10, 0},
        {"1 KiB chunks round-robin", 1 << 10, 0},
        {"~row-sized 960 B chunks (RR)", 960 & ~127, 0},
    };
    for (size_t c = 0; c < sizeof cfg / sizeof cfg[0]; c++) {
        printf("%-34s", cfg[c].name);
        for (int i = 0; i < 6; i++) {
            double s[16];
            for (int r = 0; r < reps; r++)
                s[r] = cfg[c].chunk ? run_chunks(buf, n, tl[i], cfg[c].chunk, cfg[c].pf)
                                    : run_slice(buf, n, tl[i], cfg[c].pf);
            qsort(s, reps, sizeof(double), cmp);
            printf("%8.1f", (double)n / s[reps / 2] / 1e9);
        }
        printf("\n");
    }
    fprintf(stderr, "sink %llu\n", (unsigned long long)sink);
    return 0;
}
