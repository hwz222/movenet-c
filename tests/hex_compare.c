#include <stdio.h>
#include <stdlib.h>

/* hex_compare <file_a> <file_b>
   Reads space-separated hex tokens from both files and compares them.
   Exits 0 on match, 1 on mismatch or error. */
int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "usage: hex_compare <file_a> <file_b>\n");
        return 1;
    }

    FILE *fa = fopen(argv[1], "r");
    FILE *fb = fopen(argv[2], "r");
    if (!fa || !fb) {
        if (fa) fclose(fa);
        if (fb) fclose(fb);
        fprintf(stderr, "cannot open '%s' or '%s'\n", argv[1], argv[2]);
        return 1;
    }

    size_t idx = 0;
    int match = 1;
    unsigned va, vb;

    while (1) {
        int ra = fscanf(fa, " %x", &va);
        int rb = fscanf(fb, " %x", &vb);

        if (ra == EOF && rb == EOF) break;

        if (ra == EOF || rb == EOF) {
            fprintf(stderr, "LENGTH MISMATCH: files differ at index %zu\n", idx);
            match = 0;
            break;
        }
        if (ra != 1 || rb != 1) {
            fprintf(stderr, "PARSE ERROR at index %zu\n", idx);
            match = 0;
            break;
        }
        if (va != vb) {
            fprintf(stderr, "MISMATCH at [%zu]: a=%02x  b=%02x\n", idx, va, vb);
            match = 0;
        }
        idx++;
    }

    fclose(fa);
    fclose(fb);
    printf("%s\n", match ? "PASS" : "FAIL");
    return match ? 0 : 1;
}
