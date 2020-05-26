/*    
 *    File: ap.c
 *   Title: Averaged Perceptron Algorithm for Wapiti<http://wapiti.limsi.fr/>
 *    How to use averaged perceptron algorithm (this ap.c) with Wapiti?
 *    [1] wapiti/src/wapiti.c:
 *        >       {"rprop-",   trn_rprop }
 *        --- 
 *        <       {"rprop-",   trn_rprop },
 *        <       {"ap_minup",     trn_ap_minup   }
 *    [2] wapiti/src/trainers.h
 *        < void trn_ap(mdl_t *mdl);
 *        < void trn_ap_minup(mdl_t *mdl);
 *  Usage: 
 *       $(wapiti-path)/wapiti train --type crf --algo ap --maxiter 50 --devel \
 *       test.dat train.dat model
 */


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
/******************************************************************************
 * Averaged Perceptron Algorithm
 *
 *   This section implement the averaged perceptron trainer. We use the averaged
 *   perceptron algorithm described by Collins[1].
 *   [1] Michael Collins. Discriminative training methods for hidden Markov
 *       models: theory and experiments with perceptron algorithms. Proceedings
 *       of the Conference on Empirical Methods in Natural Language Processing 
 *       (EMNLP 2002). 1-8. 2002.
 ******************************************************************************/

static uint32_t diff(uint32_t *x, uint32_t *y, uint32_t n) {
    uint32_t i, d = 0;
    for (i = 0; i < n; ++i) 
        if (x[i] != y[i]) 
            ++d;
    return d;
}

void trn_ap_minup(mdl_t *mdl) {
	const uint32_t W = mdl->opt->nthread;
	const uint64_t K  = mdl->nftr;
	const uint32_t I  = mdl->opt->maxiter;
	const uint32_t N  = mdl->train->nseq;
	const uint32_t  U = mdl->reader->nuni;
	const uint32_t  B = mdl->reader->nbi;
	double rho1 = mdl->opt->rho1;
	double * x = mdl->theta;
	memset(x, 0.0, sizeof(double) * K);

	uint64_t * count = xmalloc(sizeof(uint64_t) * K); // store count of weight
	memset(count, 0, sizeof(uint64_t) * K);

	// Initialize model for storing average theta
	mdl_t *mdl_avg = xmalloc(sizeof(mdl_t));
	mdl_shallow_copy_except_theta(mdl, mdl_avg);
	mdl_avg->theta = xvm_new(K);
	memset(mdl_avg->theta, 0.0, sizeof(double) * K);

    double *cached_x = xvm_new(K);
    memset(cached_x, 0, K);

	double *avg_x = mdl_avg->theta;

	uint64_t *update_count = xmalloc(sizeof(uint64_t) * mdl->nobs);
	memset(update_count, 0, sizeof(uint64_t) * mdl->nobs);
	
    uint64_t min_updates = 7;
	uint64_t max_nact = 320728;

	bool *vis = xmalloc(sizeof(bool) * K);
	memset(vis, 0, sizeof(bool) * K);
	for (uint32_t s = 0; s < N; ++s) {
		const seq_t *seq = mdl->train->seq[s];
		const uint32_t T = seq->len;
		uint32_t y[T];

		for (uint32_t t = 0; t < T; ++t)
			y[t] = seq->pos[t].lbl;
		
		for (uint32_t t = 0; t < T; ++t) {
			const pos_t *pos = &(seq->pos[t]);
			
			for (uint32_t j = 0; j < pos->ucnt; ++j) {
				uint64_t k = mdl->uoff[pos->uobs[j]];
				size_t y_off = k + y[t];
				vis[y_off] = true;
			}
			if (t > 0){
				for (uint32_t j = 0; j < pos->bcnt; ++j) {
					uint64_t k = mdl->boff[pos->bobs[j]];
					size_t y_off = k + (y[t-1] * mdl->nlbl + y[t]);
					vis[y_off] = true;
				}
			}
			
		}
		
	}

	bool *overlap = xmalloc(sizeof(bool) * K);
	// Train
	uint32_t *perm = xmalloc(sizeof(uint32_t) * N);
	for (uint32_t s = 0; s < N; s++)
		perm[s] = s;

	uint64_t C = 1;

	for (uint32_t i = 0; !uit_stop && i < I; ++i) {

		// First we shuffle the sequence by making a lot of random swap
		// of entry in the permutation index.
		for (uint32_t s = 0; s < N; s++) {
			const uint32_t a = rand() % N;
			const uint32_t b = rand() % N;
			const uint32_t t = perm[a];
			perm[a] = perm[b];
			perm[b] = t;
		}
		// for (uint32_t s = 0; s < 10; s++){
		// 	printf("%" PRIu32 " ", perm[s]);
		// }

		double loss = 0.0;
		
		for (uint32_t s = 0; s < N; ++s) {
			uint32_t n = perm[s];
			// Tag the sequence with the viterbi
			const seq_t *seq = mdl->train->seq[n];
			const uint32_t T = seq->len;
			uint32_t out[T], y[T];

			for (uint32_t t = 0; t < T; ++t)
				y[t] = seq->pos[t].lbl;

			tag_viterbi(mdl, seq, (uint32_t*)out, NULL, NULL);

			uint32_t d = diff(out, y, T);
			
			if (d > 0){
				for (uint32_t t = 0; t < T; ++t) {
					const pos_t *pos = &(seq->pos[t]);
					
					for (uint32_t j = 0; j < pos->ucnt; ++j) {
						if (y[t] == out[t]) 
							continue;
						uint64_t k = mdl->uoff[pos->uobs[j]];
						size_t y_off = k + y[t];
						size_t out_off = k + out[t];
						if (vis[y_off]){
							cached_x[y_off] += x[y_off] * (C - count[y_off]);
							count[y_off] = C;
							x[y_off] += 1;
						}
						

						if (vis[out_off]){
							cached_x[out_off] += x[out_off] * (C - count[out_off]);
							count[out_off] = C;
							x[out_off] -= 1;
						}
                        

                        update_count[pos->uobs[j]] += 1;
						
					}
					if (t > 0){
						for (uint32_t j = 0; j < pos->bcnt; ++j) {
							
							uint64_t k = mdl->boff[pos->bobs[j]];
							size_t y_off = k + (y[t-1] * mdl->nlbl + y[t]);
							size_t out_off = k + (out[t-1] * mdl->nlbl + out[t]);
							if (y_off == out_off) 
								continue;

							if (vis[y_off]) {
								cached_x[y_off] += x[y_off] * (C - count[y_off]);
								count[y_off] = C;
								x[y_off] += 1;
							}
							

							if (vis[out_off]){
								cached_x[out_off] += x[out_off] * (C - count[out_off]);
								count[out_off] = C;
								x[out_off] -= 1;
							}

                            update_count[pos->bobs[j]] += 1;
						}
					}

				}
			}
			
			
			loss += d / (double) T;
			C += 1;
		} 
		memset(avg_x, 0, sizeof(double) * K);
        uint32_t bcnt = mdl->nlbl * mdl->nlbl;
		uint32_t nact = 0;
		memset(overlap, 0, sizeof(bool) * K);


		for (uint32_t f = 0; f < mdl->nobs; ++f) {
            if (update_count[f] < min_updates)
                continue;
            uint32_t uoff = mdl->uoff[f];
            for (uint32_t j = 0; j < mdl->nlbl; ++j) {
                uint32_t k = uoff + j;
                avg_x[k] = (cached_x[k] + (C - count[k]) * x[k]) / C;
				overlap[k] = true;
				if (avg_x[k] != 0.0) nact += 1;
            }
            uint32_t boff = mdl->boff[f];
            for (uint32_t j = 0; j < bcnt; ++j) {
                uint32_t k = boff + j;
                avg_x[k] = (cached_x[k] + (C - count[k]) * x[k]) / C;
            }
		}
		printf("nact%" PRIu32 "\n", nact);
		// 某一轮开始，不再使用avg[k]=0的特征
		if (i >= 8){
			for (uint32_t k = 0; k < K; ++k) {
				if (avg_x[k] == 0){
					x[k] = 0;
					cached_x[k] = 0;
					vis[k]  = false;
				}
			}
		}
		

		if (uit_progress(mdl_avg, i+1, loss) == false) {
			for (uint32_t k = 0; k < K; ++k) {
				x[k] = mdl_avg->theta[k];
			}
			break;
		}
		
		// save model after each iteration
		if (i % 10 == 0){
			char fname[200];
			sprintf(fname, "%s_%d", mdl->opt->output, i+1);
			info(fname);
			FILE* file = fopen(fname, "w");
			if (file == NULL) {
				fatal("cannot open output model");
			}
			mdl_save(mdl_avg, file);
			fclose(file);
		}
	}
	free(overlap);
	free(perm);
	xvm_free(mdl_avg->theta);
	free(mdl_avg);
	free(count);


}