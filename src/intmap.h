// Simple uint64_t -> uint64_t hash map implementation.
//
// Keys and values may not be equal to INTMAP_EMPTY_KEY or INTMAP_EMPTY_VALUE,
// respectively.
//
// Deletion is not implemented, and the main use is intmap_setdefault()
#pragma once
#include <stdint.h>


// If these are changed, make sure to modify the memset() calls below
#define INTMAP_EMPTY_KEY    0xffffffffffffffffULL
#define INTMAP_EMPTY_VALUE  0

// Aim at a maximum capacity of 1/INTMAP_CAPACITY (should be at least 2)
#define INTMAP_CAPACITY     2

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

// xxHash64's finalization step
static inline uint64_t hash(uint64_t x) {
    x = (x ^ (x >> 33)) * 14029467366897019727ULL;
    x = (x ^ (x >> 29)) * 1609587929392839161ULL;
    return x ^ (x >> 32);
}

typedef struct {
    uint64_t size;
    uint64_t n;
    uint64_t *buf;
    double *values;
} intmap;

static int intmap_set(intmap *im, uint64_t k, double v);
static void intmap_printf(const intmap *im);

static int intmap_create(intmap *im, uint64_t size) {
    // printf("enter intmap_create\n");
    if (size == 0) return 1;
    // printf("size=%u\n", size);
    im->n = 0;
    im->size = size;
    im->buf = malloc(im->size * sizeof(uint64_t));
    im->values = malloc(im->size * sizeof(double));
    if (im->buf == NULL || im->values == NULL) return 1;
    memset(im->buf, INTMAP_EMPTY_KEY, im->size * sizeof(uint64_t));
    memset(im->values, 0, im->size * sizeof(double));
    return 0;
}

static int intmap_clean(intmap *im) {
    if (im == NULL){
        printf("im is NULL\n");
    }
    im->n = 0;
    memset(im->buf, INTMAP_EMPTY_KEY, im->size * sizeof(uint64_t));
    memset(im->values, 0, im->size * sizeof(double));
    return 0;
}


static void intmap_free(intmap *im) {
    free(im->buf);
    free(im->values);
    im->n = 0;
    im->size = 0;
    im->buf = NULL;
    im->values = NULL;
}

static int intmap_expand(intmap *im) {
    uint64_t *old_buf = im->buf;
    double *old_value = im->values;
    uint64_t old_size = im->size;
    if (im->size == 0x8000000000000000ULL) return 1;
    im->size *= 2;
    // im->n = 0;
    im->buf = malloc(im->size * sizeof(uint64_t));
    im->values = malloc(im->size * sizeof(double));
    if (im->buf == NULL) return 1;
    memset(im->buf, INTMAP_EMPTY_KEY, im->size * sizeof(uint64_t));
    memset(im->values, 0, im->size * sizeof(double));
    for (uint64_t i=0; i<old_size; ++i) {
        if (old_buf[i] != INTMAP_EMPTY_KEY) {
            intmap_set(im, old_buf[i], old_value[i]);
        }
    }
    free(old_buf);
    free(old_value);
    return 0;
}

static inline uint64_t intmap_get_slot(const intmap *im, uint64_t k) {
    const uint64_t mask = im->size - 1;
    uint64_t i = ((uint64_t) hash(k)) & mask;

    while (1) {
        if (im->buf[i] == k) return i;
        if (im->buf[i] == INTMAP_EMPTY_KEY) return i;
        i = (i + 1) & mask;
    }
}

static inline double intmap_get(const intmap *im, uint64_t k) {
    uint64_t slot = intmap_get_slot(im, k);
    // if (im->buf[slot] != k) fatal("k=%d doesn't exist\n", k); 
    return im->values[slot];
}


static inline int intmap_del(intmap *im, uint64_t k) {
    uint64_t slot = intmap_get_slot(im, k);
    if (im->buf[slot] == k){
        im->buf[slot] = INTMAP_EMPTY_KEY;
        im->values[slot] = 0;
        im->n--;
        return 1;
    }
    return 0;
}

static inline void intmap_printf(const intmap *im) {
    printf("n=%d size=%d\n", im->n, im->size);
    
    for (uint64_t i = 0; i < im->size; ++i) {
        if (im->buf[i] != INTMAP_EMPTY_KEY) 
            printf("i=%d k=%d v=%f\n", i, im->buf[i], im->values[i]);
    }

}

static int intmap_set(intmap *im, uint64_t k, double v) {
    if (im->n * INTMAP_CAPACITY * 2 > im->size) 
            if (intmap_expand(im) < 0) return -1;
    const uint64_t i = intmap_get_slot(im, k);
    if (im->buf[i] == INTMAP_EMPTY_KEY) {
        im->n++;
        im->buf[i] = k;
        im->values[i] = v;
        return 1;
    }
    im->values[i] = v;
    return 0;
}

static int intmap_inc(intmap *im, uint64_t k, double inc) {
    if (im->n*INTMAP_CAPACITY*2 > im->size)
            if (intmap_expand(im) < 0) return -1;
    const uint64_t i = intmap_get_slot(im, k);
    if (im->buf[i] == INTMAP_EMPTY_KEY) {
        im->n++;
        im->buf[i] = k;
        im->values[i] += inc;
        return 1;
    }
    im->values[i] += inc;
    return 0;
}

static uint64_t intmap_setdefault(intmap *im, uint64_t k, double v) {
    const uint64_t i = intmap_get_slot(im, k);
    if (im->buf[i] == INTMAP_EMPTY_KEY) {
        im->n++;
        im->buf[i] = k;
        im->values[i] = v;
        if (im->n*INTMAP_CAPACITY*2 > im->size)
                if (intmap_expand(im) < 0) return -1;
        return v;
    }
    return im->values[i];
}