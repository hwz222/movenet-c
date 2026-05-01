#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "tensor3d.h"

/* Read space-separated int8 hex tokens (e.g. "80 00 7f") into tensor data. */
static int read_i8_hex_file(MyTensor *t, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open '%s'\n", path); return -1; }

    size_t n = tensor_size(t);
    for (size_t i = 0; i < n; i++) {
        unsigned v;
        if (fscanf(f, " %x", &v) != 1) {
            fprintf(stderr, "read error at index %zu in '%s'\n", i, path);
            fclose(f); return -1;
        }
        t->data[i] = (int8_t)(uint8_t)v;
    }
    fclose(f);
    return 0;
}

static int write_i8_hex_file(const MyTensor *t, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "cannot open '%s'\n", path); return -1; }

    size_t n = tensor_size(t);
    for (size_t i = 0; i < n; i++) {
        fprintf(f, "%02x", (uint8_t)t->data[i]);
        if (i + 1 < n) fputc(' ', f);
    }
    fputc('\n', f);
    fclose(f);
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc != 13) {
        fprintf(stderr,
            "usage: %s <H> <W> <C>"
            " <in_a.hex> <in_b.hex> <out.hex>"
            " <scale_out> <zp_out>"
            " <scale_a> <zp_a>"
            " <scale_b> <zp_b>\n",
            argv[0]);
        return 1;
    }

    int32_t h = (int32_t)atoi(argv[1]);
    int32_t w = (int32_t)atoi(argv[2]);
    int32_t c = (int32_t)atoi(argv[3]);

    const char *in_a_path  = argv[4];
    const char *in_b_path  = argv[5];
    const char *out_path   = argv[6];

    float   scale_out = (float)atof(argv[7]);
    int8_t  zp_out    = (int8_t)atoi(argv[8]);
    float   scale_a   = (float)atof(argv[9]);
    int8_t  zp_a      = (int8_t)atoi(argv[10]);
    float   scale_b   = (float)atof(argv[11]);
    int8_t  zp_b      = (int8_t)atoi(argv[12]);

    MyTensor *ta  = tensor_create(h, w, c);
    MyTensor *tb  = tensor_create(h, w, c);
    MyTensor *out = tensor_create(h, w, c);
    if (!ta || !tb || !out) {
        fprintf(stderr, "tensor_create failed\n");
        tensor_free(ta); tensor_free(tb); tensor_free(out);
        return 1;
    }

    if (read_i8_hex_file(ta, in_a_path) != 0 ||
        read_i8_hex_file(tb, in_b_path) != 0) {
        tensor_free(ta); tensor_free(tb); tensor_free(out);
        return 1;
    }

    tensor_add_q(out, ta, tb,
                 scale_out, zp_out,
                 scale_a,   zp_a,
                 scale_b,   zp_b);

    int rc = write_i8_hex_file(out, out_path);

    tensor_free(ta); tensor_free(tb); tensor_free(out);
    return rc == 0 ? 0 : 1;
}
