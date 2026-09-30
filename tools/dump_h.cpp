// Dump target-model tokens + pre-final-norm hidden states (h_nextn) for MTP head training.
// usage: dump_h model.gguf in.txt out.bin [n_ctx=8192]
// out.bin: int32 N | int32 n_embd | int32 tok[N] | fp16 h[N][n_embd]   (h[i] = hidden at position i)
#include "llama.h"
#include "../src/llama-ext.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model in.txt out.bin [n_ctx]\n", argv[0]); return 1; }
    const int n_ctx = argc > 4 ? atoi(argv[4]) : 8192;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = 512; cp.n_ubatch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 3;
    llama_set_embeddings_nextn(ctx, true, /*masked*/ false);
    const int n_embd = llama_model_n_embd(model);

    std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf(); std::string text = ss.str();
    std::vector<llama_token> tok(text.size() + 16);
    int n = llama_tokenize(vocab, text.data(), text.size(), tok.data(), tok.size(), false, true);
    if (n < 0) return 4;
    tok.resize(n);
    if (n > n_ctx) { fprintf(stderr, "truncating %d -> %d tokens\n", n, n_ctx); tok.resize(n_ctx); n = n_ctx; }

    std::vector<_Float16> h((size_t) n * n_embd);
    for (int s = 0; s < n; s += 512) {
        const int c = std::min(512, n - s);
        llama_batch b = llama_batch_get_one(tok.data() + s, c);
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed at %d\n", s); return 5; }
        const float * e = llama_get_embeddings_nextn(ctx);
        if (!e) { fprintf(stderr, "no nextn embeddings\n"); return 6; }
        for (size_t i = 0; i < (size_t) c * n_embd; ++i) h[(size_t) s * n_embd + i] = (_Float16) e[i];
    }
    FILE * o = fopen(argv[3], "wb");
    fwrite(&n, 4, 1, o); fwrite(&n_embd, 4, 1, o);
    fwrite(tok.data(), 4, n, o); fwrite(h.data(), 2, h.size(), o);
    fclose(o);
    fprintf(stderr, "wrote %d tokens x %d\n", n, n_embd);
    return 0;
}
