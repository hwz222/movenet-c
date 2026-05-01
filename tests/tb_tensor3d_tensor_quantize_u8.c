#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "tensor3d.h"

static uint8_t *read_u8_hex_file(const char *path, size_t n)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open '%s'\n", path); return NULL; }

    uint8_t *buf = malloc(n);
    if (!buf) { fclose(f); return NULL; }

    for (size_t i = 0; i < n; i++) {
        unsigned v;
        if (fscanf(f, " %x", &v) != 1) {
            fprintf(stderr, "read error at index %zu in '%s'\n", i, path);
            free(buf); fclose(f); return NULL;
        }
        buf[i] = (uint8_t)v;
    }
    fclose(f);
    return buf;
}

static int write_i8_hex_file(const char *path, const int8_t *data, size_t n)
{
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "cannot open '%s'\n", path); return -1; }

    for (size_t i = 0; i < n; i++) {
        fprintf(f, "%02x", (uint8_t)data[i]);
        if (i + 1 < n) fputc(' ', f);
    }
    fputc('\n', f);
    fclose(f);
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc != 6) {
        fprintf(stderr,
            "usage: %s <H> <W> <C> <input.hex> <output.hex>\n",
            argv[0]);
        return 1;
    }

    int32_t h     = (int32_t)atoi(argv[1]);
    int32_t w     = (int32_t)atoi(argv[2]);
    int32_t c     = (int32_t)atoi(argv[3]);
    const char *in_path  = argv[4];
    const char *out_path = argv[5];

    size_t n = (size_t)h * w * c;

    uint8_t *src = read_u8_hex_file(in_path, n);
    if (!src) return 1;

    MyTensor *t = tensor_create(h, w, c);
    if (!t) { free(src); return 1; }

    tensor_quantize_u8(t, src);
    free(src);

    int rc = write_i8_hex_file(out_path, t->data, n);
    tensor_free(t);
    return rc == 0 ? 0 : 1;
}
