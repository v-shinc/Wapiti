#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>

#include "wapiti.h"
#include "decoder.h"
#include "model.h"
#include "options.h"
#include "progress.h"
#include "tools.h"
#include "thread.h"
#include "vmath.h"


static uint32_t diff(uint32_t *x, uint32_t *y, uint32_t n) {
    uint32_t i, d = 0;
    for (i = 0; i < n; ++i) 
        if (x[i] != y[i]) 
            ++d;
    return d;
}

typedef struct info_s info_t;
struct info_s {
    uint64_t * count; // store count of weight
	double *avg_theta;
    mdl_t *mdl;
    uint64_t C;
    double loss;
	uint32_t *perm;
};
void one_epoch_ap(job_t *job, uint32_t id, uint32_t cnt, info_t *info){
	unused(id && cnt);
	uint32_t jobsize, pos;
    mdl_t *mdl = info->mdl;
	const uint64_t K  = mdl->nftr;
	pos = id * mdl->opt->jobsize;
	jobsize = mdl->opt->jobsize;

	double * x = mdl->theta;
    double * avg_x = info->avg_theta;
    uint64_t * count = info->count;
	info->loss = 0;

	// First we shuffle the sequence by making a lot of random swap
	// of entry in the permutation index.
	
	
	for (uint32_t s = 0; s < jobsize; s++) {
		const uint32_t a = rand() % jobsize;
		const uint32_t b = rand() % jobsize;
		const uint32_t t = info->perm[a];
		info->perm[a] = info->perm[b];
		info->perm[b] = t;
	}
	for (uint32_t pi = 0; pi < jobsize; pi++) {
		uint32_t s = info->perm[pi];
		const seq_t *seq = mdl->train->seq[s];
		const uint32_t T = seq->len;
		uint32_t out[T], y[T];

		for (uint32_t t = 0; t < T; ++t)
			y[t] = seq->pos[t].lbl;

		tag_viterbi(mdl, seq, (uint32_t*)out, NULL, NULL);

		uint32_t d = diff(out, y, T);
		if (d > 0){
			// printf("update%" PRIu64 "\n", C);

			for (uint32_t t = 0; t < T; ++t) {
				const pos_t *pos = &(seq->pos[t]);
				
				for (uint32_t j = 0; j < pos->ucnt; ++j) {
					uint64_t k = mdl->uoff[pos->uobs[j]];
					size_t y_off = k + y[t];
					size_t out_off = k + out[t];
					if (y_off == out_off) 
						continue;

					avg_x[y_off] += x[y_off] * (info->C - count[y_off]);
					count[y_off] = info->C;
					x[y_off] += 1;

					avg_x[out_off] += x[out_off] * (info->C - count[out_off]);
					count[out_off] = info->C;
					x[out_off] -= 1;
					
				}
				if (t > 0){
					for (uint32_t j = 0; j < pos->bcnt; ++j) {
						uint64_t k = mdl->boff[pos->bobs[j]];
						size_t y_off = k + (y[t-1] * mdl->nlbl + y[t]);
						size_t out_off = k + (out[t-1] * mdl->nlbl + out[t]);
						if (y_off == out_off) 
							continue;

						avg_x[y_off] += x[y_off] * (info->C - count[y_off]);
						count[y_off] = info->C;
						x[y_off] += 1;

						avg_x[out_off] += x[out_off] * (info->C - count[out_off]);
						count[out_off] = info->C;
						x[out_off] -= 1;
					}
				}
				
				
			}
			info->loss += d / (double) T;
			info->C += 1;
		}
	}
    for (uint64_t k = 0; k < K; ++k) {
        avg_x[k] += (info->C - count[k]) * x[k];
        count[k] = info->C;
    }


}
void trn_ap_parallel(mdl_t *mdl) {

	const uint32_t W = mdl->opt->nthread;
	const uint64_t K  = mdl->nftr;
	const uint32_t N  = mdl->train->nseq;
	const uint32_t I  = mdl->opt->maxiter;
	mdl->loss = 0;
	mdl->opt->jobsize = N / W;
    mdl->theta = xvm_new(K);
	memset(mdl->theta, 0.0, sizeof(double) * K);
    info_t *infos[W];
    for (uint32_t i = 0; i < W; ++i){
		infos[i] = xmalloc(sizeof(info_t));
		infos[i]->mdl = xmalloc(sizeof(mdl_t));
		mdl_shallow_copy_except_theta(mdl, infos[i]->mdl);
		infos[i]->mdl->theta = xvm_new(K);
        infos[i]->avg_theta = xvm_new(K);
        infos[i]->count = xmalloc(sizeof(uint64_t) * K);
		infos[i]->perm = xmalloc(sizeof(uint32_t) * mdl->opt->jobsize);
		memset(infos[i]->mdl->theta, 0.0, sizeof(double) * K);
        memset(infos[i]->avg_theta, 0.0, sizeof(double) * K);
        memset(infos[i]->count, 0, sizeof(uint64_t) * K);
		for (uint32_t s = 0; s < mdl->opt->jobsize; s++)
			infos[i]->perm[s] = i * mdl->opt->jobsize + s;
	}
	// double loss = 0.0;
	
	for (uint32_t i = 0; !uit_stop && i < I; ++i) {
        
		mth_spawn((func_t *)one_epoch_ap, W, (void **)infos,
				mdl->train->nseq, mdl->opt->jobsize);
        // printf("i: %" PRIu32 "after mth_spawn\n", i);
		double total_C = 0;
		for (uint32_t j = 0; j < W; ++j){
			total_C += infos[j]->C;
		}
			
		memset(mdl->theta, 0.0, sizeof(double) * K);
		mdl->loss = 0;
		for (uint32_t j = 0; j < W; ++j){
			for (uint64_t k = 0; k < K; ++k) {
				mdl->theta[k] += infos[j]->avg_theta[k] * 1. / total_C;
			}
			mdl->loss += infos[j]->loss;
		}
		mdl->opt->nthread = 1;

		// save model after each iteration
		if (i % 5 == 0){
			char fname[200];
			sprintf(fname, "%s_%d", mdl->opt->output, i+1);
			info(fname);
			FILE* file = fopen(fname, "w");
			if (file == NULL) {
				fatal("cannot open output model");
			}
			mdl_save(mdl, file);
			fclose(file);
		}

		if (uit_progress(mdl, i+1, mdl->loss) == false) {
			break;
		}
		// 更新参数
		for (uint32_t w = 0; w < W; ++w) {
			for (uint64_t k = 0; k < K; ++k) {
				infos[w]->mdl->theta[k] = mdl->theta[k];
				infos[w]->avg_theta[k] = mdl->theta[k] * infos[w]->C;
				
			}
		}
		mdl->opt->nthread = W;
	}


	for (uint32_t i = 0; i < W; ++i){
		xvm_free(infos[i]->mdl->theta);
		free(infos[i]->perm);
		free(infos[i]->mdl);
        xvm_free(infos[i]->avg_theta);
        free(infos[i]->count);
        free(infos[i]);
	}
}