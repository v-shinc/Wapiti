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


void one_epoch_perceptron(job_t *job, uint32_t id, uint32_t cnt, mdl_t *mdl){
	unused(id && cnt);
	uint32_t count, pos;
	// mth_getjob(job, &count, &pos);
	
	pos = id * mdl->opt->jobsize;
	count = mdl->opt->jobsize;
	// printf("pos=%d count=%d", pos, count);

	double * x = mdl->theta;
	mdl->loss = 0.0;
	mdl->wcnt = 0;
	for (uint32_t s = pos; s < pos + count; s++) {
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
						x[y_off]   += 1;
						x[out_off] -= 1;
						
					}
					if (t > 0){
						for (uint32_t j = 0; j < pos->bcnt; ++j) {
							uint64_t k = mdl->boff[pos->bobs[j]];
							size_t y_off = k + (y[t-1] * mdl->nlbl + y[t]);
							size_t out_off = k + (out[t-1] * mdl->nlbl + out[t]);
							x[y_off]   += 1;
							x[out_off] -= 1;
						}
					}
					
					
				}
				
				//free(psc);
				//free(scs);
				// loss += d / (double) T;
				mdl->loss += d / (double) T;
				mdl->wcnt += 1;
			}

	}

}
void trn_perceptron_parallel(mdl_t *mdl) {

	const uint32_t W = mdl->opt->nthread;
	const uint64_t K  = mdl->nftr;
	const uint32_t N  = mdl->train->nseq;
	const uint32_t I  = mdl->opt->maxiter;

	memset(mdl->theta, 0.0, sizeof(double) * K);

	mdl_t *mdls[W];
	for (uint32_t i = 0; i < W; ++i){
		mdls[i] = xmalloc(sizeof(mdl_t));
		mdl_shallow_copy_except_theta(mdl, mdls[i]);
		mdls[i]->theta = xvm_new(K);
		memset(mdls[i]->theta, 0.0, sizeof(double) * K);
	}
	// double loss = 0.0;
	mdl->loss = 0;
	mdl->opt->jobsize = N / W;
	for (uint32_t i = 0; !uit_stop && i < I; ++i) {
		mth_spawn((func_t *)one_epoch_perceptron, W, (void **)mdls,
				mdl->train->nseq, mdl->opt->jobsize);
		double total_error_count = 0;
		for (uint32_t j = 0; j < W; ++j){
			total_error_count += mdls[j]->wcnt;
		}
			
		memset(mdl->theta, 0.0, sizeof(double) * K);
		mdl->loss = 0;
		for (uint32_t j = 0; j < W; ++j){
			for (uint64_t k = 0; k < K; ++k) {
				mdl->theta[k] += mdls[j]->theta[k] * mdls[j]->wcnt * 1. / total_error_count;
				// mdl->theta[k] += mdls[j]->theta[k];
			}
			mdl->loss += mdls[j]->loss;
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
		mdl->opt->nthread = W;
		for (uint32_t j = 0; j < W; ++j){
			for (uint64_t k = 0; k < K; ++k) {
				mdls[j]->theta[k] = mdl->theta[k];
			}
		}
	}
	for (uint32_t i = 0; i < W; ++i){
		xvm_free(mdls[i]->theta);
		free(mdls[i]);
	}
	
}