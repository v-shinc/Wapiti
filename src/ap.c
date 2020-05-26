/*    
 *    File: ap.c
 *   Title: Averaged Perceptron Algorithm for Wapiti<http://wapiti.limsi.fr/>
 *    How to use averaged perceptron algorithm (this ap.c) with Wapiti?
 *    [1] wapiti/src/wapiti.c:
 *        >       {"rprop-",   trn_rprop }
 *        --- 
 *        <       {"rprop-",   trn_rprop },
 *        <       {"ap",     trn_ap   }
 *    [2] wapiti/src/trainers.h
 *        < void trn_ap(mdl_t *mdl);
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

// void one_epoch_perceptron(job_t *job, uint32_t id, uint32_t cnt, mdl_t *mdl){
// 	unused(id && cnt);
// 	uint32_t count, pos;
	
// 	// mth_getjob(job, &count, &pos);
	
// 	pos = id * mdl->opt->jobsize;
// 	count = mdl->opt->jobsize;
// 	// printf("pos=%d count=%d", pos, count);

// 	double * x = mdl->theta;
// 	mdl->loss = 0.0;
// 	mdl->wcnt = 0;
// 	for (uint32_t s = pos; s < pos + count; s++) {
// 		const seq_t *seq = mdl->train->seq[s];
// 			const uint32_t T = seq->len;
// 			uint32_t out[T], y[T];

// 			for (uint32_t t = 0; t < T; ++t)
// 				y[t] = seq->pos[t].lbl;

// 			tag_viterbi(mdl, seq, (uint32_t*)out, NULL, NULL);

// 			uint32_t d = diff(out, y, T);
// 			if (d > 0){
// 				// printf("update%" PRIu64 "\n", C);

// 				for (uint32_t t = 0; t < T; ++t) {
// 					const pos_t *pos = &(seq->pos[t]);
					
// 					for (uint32_t j = 0; j < pos->ucnt; ++j) {
// 						uint64_t k = mdl->uoff[pos->uobs[j]];
// 						size_t y_off = k + y[t];
// 						size_t out_off = k + out[t];
// 						x[y_off]   += 1;
// 						x[out_off] -= 1;
						
// 					}
// 					if (t > 0){
// 						for (uint32_t j = 0; j < pos->bcnt; ++j) {
// 							uint64_t k = mdl->boff[pos->bobs[j]];
// 							size_t y_off = k + (y[t-1] * mdl->nlbl + y[t]);
// 							size_t out_off = k + (out[t-1] * mdl->nlbl + out[t]);
// 							x[y_off]   += 1;
// 							x[out_off] -= 1;
// 						}
// 					}
					
					
// 				}
				
// 				//free(psc);
// 				//free(scs);
// 				// loss += d / (double) T;
// 				mdl->loss += d / (double) T;
// 				mdl->wcnt += 1;
// 			}

// 	}

// }


// void trn_average_perceptron_worker(job_t *job, uint32_t id, uint32_t cnt, mdl_t *mdl){
// 	unused(id && cnt);
// 	uint32_t count, pos;
// 	const uint64_t K  = mdl->nftr;
// 	double * cached_x = xvm_new(K);
// 	memset(cached_x, 0.0, sizeof(double) * K);  

// 	pos = id * mdl->opt->jobsize;
// 	count = mdl->opt->jobsize;

// 	double * x = mdl->theta;
// 	mdl->loss = 0.0;
// 	mdl->wcnt = 0;
// 	uint64_t C = 1;
// 	for (uint32_t s = pos; s < pos + count; s++) {
// 		const seq_t *seq = mdl->train->seq[s];
// 			const uint32_t T = seq->len;
// 			uint32_t out[T], y[T];

// 			for (uint32_t t = 0; t < T; ++t)
// 				y[t] = seq->pos[t].lbl;

// 			tag_viterbi(mdl, seq, (uint32_t*)out, NULL, NULL);

// 			uint32_t d = diff(out, y, T);
// 			if (d > 0){
// 				// printf("update%" PRIu64 "\n", C);

// 				for (uint32_t t = 0; t < T; ++t) {
// 					const pos_t *pos = &(seq->pos[t]);
					
// 					for (uint32_t j = 0; j < pos->ucnt; ++j) {
// 						uint64_t k = mdl->uoff[pos->uobs[j]];
// 						size_t y_off = k + y[t];
// 						size_t out_off = k + out[t];
// 						x[y_off]   += 1;
// 						x[out_off] -= 1;
// 						cached_x[y_off] += C;
// 						cached_x[out_off] -= C;
						
// 					}
// 					if (t > 0){
// 						for (uint32_t j = 0; j < pos->bcnt; ++j) {
// 							uint64_t k = mdl->boff[pos->bobs[j]];
// 							size_t y_off = k + (y[t-1] * mdl->nlbl + y[t]);
// 							size_t out_off = k + (out[t-1] * mdl->nlbl + out[t]);
// 							x[y_off]   += 1;
// 							x[out_off] -= 1;
// 							cached_x[y_off] += C;
// 							cached_x[out_off] -= C;
// 						}
// 					}
					
					
// 				}
				
// 				//free(psc);
// 				//free(scs);
// 				// loss += d / (double) T;
// 				mdl->loss += d / (double) T;
// 				mdl->wcnt += 1;
// 			}
// 			C += 1;
// 	}
// 	for (uint64_t k = 0; k < K; ++k) {
// 			x[k] = x[k] - cached_x[k] / C;
// 		}
// }

// void trn_ap(mdl_t *mdl) {
// 	const uint32_t W = mdl->opt->nthread;
// 	if (W > 1) {
// 		trn_perceptron(mdl);
// 		return;
// 	}
// 	const uint64_t K  = mdl->nftr;
// 	const uint32_t I  = mdl->opt->maxiter;
// 	const uint32_t N  = mdl->train->nseq;
// 	double * x = mdl->theta;
// 	double * cached_x = xvm_new(K);
	
// 	memset(x, 0.0, sizeof(double) * K);
// 	memset(cached_x, 0.0, sizeof(double) * K);  

// 	// initialize model for storing average theta
// 	mdl_t *mdl_avg = xmalloc(sizeof(mdl_t));
// 	mdl_shallow_copy_except_theta(mdl, mdl_avg);
// 	mdl_avg->theta = xvm_new(K);
// 	// double *avg_x = mdl_avg->theta;

// 	memset(mdl_avg->theta, 0.0, sizeof(double) * K);

// 	// Initialize vis array, 'vis' indicate whether feature appear in training data
// 	bool *vis = xmalloc(sizeof(bool) * K);
// 	memset(vis, 0, sizeof(bool) * K);
// 	for (uint32_t s = 0; s < N; ++s) {
// 		const seq_t *seq = mdl->train->seq[s];
// 		const uint32_t T = seq->len;
// 		uint32_t y[T];

// 		for (uint32_t t = 0; t < T; ++t)
// 			y[t] = seq->pos[t].lbl;
		
// 		for (uint32_t t = 0; t < T; ++t) {
// 			const pos_t *pos = &(seq->pos[t]);
			
// 			for (uint32_t j = 0; j < pos->ucnt; ++j) {
// 				uint64_t k = mdl->uoff[pos->uobs[j]];
// 				size_t y_off = k + y[t];
// 				vis[y_off] = true;
// 			}

// 			for (uint32_t j = 0; j < pos->bcnt; ++j) {
// 				uint64_t k = mdl->boff[pos->bobs[j]];
// 				size_t y_off = k + (y[t-1] * mdl->nlbl + y[t]);
// 				vis[y_off] = true;
// 			}
// 		}
		
// 	}


// 	uint32_t *perm = xmalloc(sizeof(uint32_t) * N);
// 	for (uint32_t s = 0; s < N; s++)
// 		perm[s] = s;

// 	uint64_t C = 1;

// 	for (uint32_t i = 0; !uit_stop && i < I; ++i) {

// 		// First we shuffle the sequence by making a lot of random swap
// 		// of entry in the permutation index.
// 		for (uint32_t s = 0; s < N; s++) {
// 			const uint32_t a = rand() % N;
// 			const uint32_t b = rand() % N;
// 			const uint32_t t = perm[a];
// 			perm[a] = perm[b];
// 			perm[b] = t;
// 		}

// 		double loss = 0.0;
		
// 		for (uint32_t s = 0; s < N; ++s) {
// 			uint32_t n = perm[s];
// 			// Tag the sequence with the viterbi
// 			const seq_t *seq = mdl->train->seq[n];
// 			const uint32_t T = seq->len;
// 			uint32_t out[T], y[T];

// 			for (uint32_t t = 0; t < T; ++t)
// 				y[t] = seq->pos[t].lbl;

// 			tag_viterbi(mdl, seq, (uint32_t*)out, NULL, NULL);

// 			uint32_t d = diff(out, y, T);
// 			if (d > 0){
// 				// printf("update%" PRIu64 "\n", C);

// 				for (uint32_t t = 0; t < T; ++t) {
// 					const pos_t *pos = &(seq->pos[t]);
					
// 					for (uint32_t j = 0; j < pos->ucnt; ++j) {
// 						uint64_t k = mdl->uoff[pos->uobs[j]];
// 						size_t y_off = k + y[t];
// 						size_t out_off = k + out[t];


// 						cached_x[y_off] += C * 0.1;
// 						x[y_off]   += 0.1;

// 						if (vis[out_off]){
// 							cached_x[out_off] -= C * 0.1;
// 							x[out_off] -= 0.1;
// 						}
						
// 					}
// 					if (t > 0){
// 						for (uint32_t j = 0; j < pos->bcnt; ++j) {
// 							uint64_t k = mdl->boff[pos->bobs[j]];
// 							size_t y_off = k + (y[t-1] * mdl->nlbl + y[t]);
// 							size_t out_off = k + (out[t-1] * mdl->nlbl + out[t]);
							
// 							cached_x[y_off] += C * 0.1;
// 							x[y_off]  += 0.1;
// 							if (vis[out_off]) {
// 								cached_x[out_off] -= C * 0.1;
// 								x[out_off] -= 0.1;
// 							}
							
// 						}
// 					}
					
					
// 				}
				
// 				//free(psc);
// 				//free(scs);
// 				loss += d / (double) T;
// 			}
// 			C += 1;
// 		}
		
// 		for (uint64_t k = 0; k < K; ++k) {
			
// 			mdl_avg->theta[k] = x[k] - cached_x[k] / C;
// 		}
// 		// save model after each iteration
// 		if (i % 10 == 0){
// 			char fname[200];
// 			sprintf(fname, "%s_%d", mdl->opt->output, i+1);
// 			info(fname);
// 			FILE* file = fopen(fname, "w");
// 			if (file == NULL) {
// 				fatal("cannot open output model");
// 			}
// 			mdl_save(mdl_avg, file);
// 			fclose(file);
// 		}
		

// 		if (uit_progress(mdl_avg, i+1, loss) == false) {
// 			for (uint64_t k = 0; k < K; ++k) {
// 				x[k] = mdl_avg->theta[k];
// 			}
// 			break;
// 		}
			
// 	}
// 	xvm_free(mdl_avg->theta);
// 	free(mdl_avg);
// 	xvm_free(cached_x);
	
// }

#define ABS(x) ( (x)>0?(x):-(x) )
#define MAX(a,b)  (((a)>(b))?(a):(b))

void trn_ap(mdl_t *mdl) {
	const uint64_t K  = mdl->nftr;
	const uint32_t I  = mdl->opt->maxiter;
	const uint32_t N  = mdl->train->nseq;
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
	double *avg_x = mdl_avg->theta;

	// Initialize vis array, 'vis' indicate whether feature appear in training data
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
			// printf("update%" PRIu64 "\n", C);
			if (d > 0){
				for (uint32_t t = 0; t < T; ++t) {
					const pos_t *pos = &(seq->pos[t]);
					
					for (uint32_t j = 0; j < pos->ucnt; ++j) {
						if (y[t] == out[t]) 
							continue;
						uint64_t k = mdl->uoff[pos->uobs[j]];
						size_t y_off = k + y[t];
						size_t out_off = k + out[t];
						avg_x[y_off] += x[y_off] * (C - count[y_off]);
						count[y_off] = C;
						x[y_off] += 1;

						if (vis[out_off] || rho1 == 0){
							avg_x[out_off] += x[out_off] * (C - count[out_off]);
							count[out_off] = C;
							x[out_off] -= 1;
						}
						

						// if (x[y_off] == 0){
						// 	x[y_off] += 1;
						// } else {
						// 	x[y_off] = MAX(0, ABS(x[y_off]) - rho1) / ABS(x[y_off]) * x[y_off] + 1;
						// 	// x[y_off] = (ABS(x[y_off]) - rho1) / ABS(x[y_off]) * x[y_off] + 1;
						// }
						// if (x[out_off] == 0) {
						// 	x[out_off] -= 1;
						// } else {
						// 	x[out_off] = MAX(0, ABS(x[out_off]) - rho1) / ABS(x[out_off]) * x[out_off] - 1;
						// 	// x[out_off] = (ABS(x[out_off]) - rho1) / ABS(x[out_off]) * x[out_off] - 1;
						// }
					}
					if (t > 0){
						for (uint32_t j = 0; j < pos->bcnt; ++j) {
							
							uint64_t k = mdl->boff[pos->bobs[j]];
							size_t y_off = k + (y[t-1] * mdl->nlbl + y[t]);
							size_t out_off = k + (out[t-1] * mdl->nlbl + out[t]);
							if (y_off == out_off) 
								continue;

							avg_x[y_off] += x[y_off] * (C - count[y_off]);
							count[y_off] = C;
							x[y_off] += 1;

							if (vis[out_off] || rho1 == 0){
								avg_x[out_off] += x[out_off] * (C - count[out_off]);
								count[out_off] = C;
								x[out_off] -= 1;
							}

							

							// if (x[y_off] == 0){
							// 	x[y_off] += 1;
							// } else {
							// 	x[y_off] = MAX(0, ABS(x[y_off]) - rho1) / ABS(x[y_off]) * x[y_off] + 1;
							// 	// x[y_off] = (ABS(x[y_off]) - rho1) / ABS(x[y_off]) * x[y_off] + 1;
							// }
							// if (x[out_off] == 0) {
							// 	x[out_off] -= 1;
							// } else {
							// 	x[out_off] = MAX(0, ABS(x[out_off]) - rho1) / ABS(x[out_off]) * x[out_off] - 1;
							// 	// x[out_off] = (ABS(x[out_off]) - rho1) / ABS(x[out_off]) * x[out_off] - 1;
							// }
								
						}
					}

				}
			}
			
			
			//free(psc);
			//free(scs);
			loss += d / (double) T;
			C += 1;
		}
		
		for (uint64_t k = 0; k < K; ++k) {
			avg_x[k] = (avg_x[k] + (C - count[k]) * x[k]) / (double) C;
		}

		if (uit_progress(mdl_avg, i+1, loss) == false) {
			for (uint64_t k = 0; k < K; ++k) {
				x[k] = mdl_avg->theta[k];
			}
			break;
		}
		// 恢复
		for (uint64_t k = 0; k < K; ++k) {
			avg_x[k] = avg_x[k] * C - (C - count[k]) * x[k];
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
	free(vis);
	free(perm);
	xvm_free(mdl_avg->theta);
	free(mdl_avg);
	free(count);

}