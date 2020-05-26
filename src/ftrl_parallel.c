/*
 *      Wapiti - A linear-chain CRF tool
 *
 * Copyright (c) 2009-2013  CNRS
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stdio.h>


#include "wapiti.h"
#include "gradient.h"
#include "model.h"
#include "options.h"
#include "progress.h"
#include "sequence.h"
#include "tools.h"
#include "vmath.h"
#include "thread.h"

/******************************************************************************
 * The FTRL-Proximal trainer
 *
 *   Implementation of the stochatic gradient descend with FTRL (Per-Coordinate 
 *   FTRL-Proximal with L1 and L2 ) described
 *   in [1]. 
 *
 *   [1] Ad Click Prediction: a View from the Trenches
 * 	 ./wapiti train -t #thread -T crf -a ftrl-parallel --ftrl_alpha 1 --lambda1 15 --lambda2 1 --stopeps 0.005 -p template.txt  -d eval2.txt train2.txt  model
 * 
 ******************************************************************************/
typedef struct sgd_idx_s {
	uint64_t *uobs;
	uint64_t *bobs;
} sgd_idx_t;

typedef struct params_s {
	double *z;
	double *n0;
    mdl_t *mdl;
	uint32_t *perm;
	grd_st_t *grd_st;
	sgd_idx_t *idx;
} params_t;

inline int sign(double x) {
	if (x > 0) return 1;
	else if (x < 0) return -1;
	else return 0;
}

#define compute_weight(f) do {       \
	if (z[f] < lambda1 && z[f] > - lambda1) {      \
		mdl->theta[f] = 0;      \
	} else {      \
		mdl->theta[f] = - 1 / ((beta + sqrt(n0[f])) / alpha + lambda2) * (z[f] - sign(z[f]) * lambda1);      \
	}      \
} while (false)
/* sgd_add:
 *   Add the <new> value in the array <obs> of size <cnt>. If the value is
 *   already present, we do nothing, else we add it.
 */
static void sgd_add(uint64_t *obs, uint32_t *cnt, uint64_t new) {
	// First check if value is already in the array, we do a linear probing
	// as it is simpler and since these array will be very short in
	// practice, it's efficient enough.
	for (uint32_t p = 0; p < *cnt; p++)
		if (obs[p] == new)
			return;
	// Insert the new value at the end since we have not found it.
	obs[*cnt] = new;
	*cnt = *cnt + 1;
}


static
void worker(job_t *job, uint32_t id, uint32_t cnt, params_t *params) {
	unused(id && cnt);
	mdl_t *mdl = params->mdl;
	const uint64_t  Y = mdl->nlbl;

	// We first cleanup the gradient and value as our parent don't do it (it
	// is better to do this also in parallel)

	sgd_idx_t *idx = params->idx;
	double beta = mdl->opt->ftrl.beta;
	double alpha = mdl->opt->ftrl.alpha;
	double lambda1 = mdl->opt->ftrl.lambda1;
	double lambda2 = mdl->opt->ftrl.lambda2;
	double *z = params->z;
	double *n0 = params->n0;
	double *w = mdl->theta;
	double *g = params->grd_st->g;
	
	uint32_t count, pos;
	while (mth_getjob(job, &count, &pos)) {
		for (uint32_t s = pos; !uit_stop && s < pos + count; s++){
			uint32_t sp = params->perm[s];
			const seq_t *seq = mdl->train->seq[sp];

			// Receive feature vector x_t and let I = {i | x_i != 0}
			// For i in I compute
			for (uint32_t n = 0; idx[s].uobs[n] != none; n++) {
				uint64_t f = mdl->uoff[idx[s].uobs[n]];
				for (uint32_t y = 0; y < Y; y++, f++) {
					// compute_weight(f);
					if (z[f] <= lambda1 && z[f] >= - lambda1) {
						w[f] = 0;
					} else {
						w[f] = - 1 / ((beta + sqrt(n0[f])) / alpha + lambda2) * (z[f] - sign(z[f]) * lambda1);
					}
				}
			}

			for (uint32_t n = 0; idx[s].bobs[n] != none; n++) {
				uint64_t f = mdl->boff[idx[s].bobs[n]];
				for (uint32_t d = 0; d < Y * Y; d++, f++) {
					// compute_weight(f);
					if (z[f] <= lambda1 && z[f] >= - lambda1) {
						w[f] = 0;
					} else {
						w[f] = - 1 / ((beta + sqrt(n0[f])) / alpha + lambda2) * (z[f] - sign(z[f]) * lambda1);
					}
				}
			}
			// compute gradient using the w computed above
			grd_dospl(params->grd_st, seq);
			
			for (uint32_t n = 0; idx[s].uobs[n] != none; n++) {
				uint64_t f = mdl->uoff[idx[s].uobs[n]];
				for (uint32_t y = 0; y < Y; y++, f++) {
					double g_2 = g[f] * g[f];
					double sigma = (sqrt(n0[f] + g_2) - sqrt(n0[f])) / alpha;
					z[f] += g[f] - sigma * w[f];
					n0[f] += g_2;
					g[f] = 0.0;
				}
			}
			for (uint32_t n = 0; idx[s].bobs[n] != none; n++) {
				uint64_t f = mdl->boff[idx[s].bobs[n]];
				for (uint32_t d = 0; d < Y * Y; d++, f++) {
					double g_2 = g[f] * g[f];
					double sigma = (sqrt(n0[f] + g_2) - sqrt(n0[f])) / alpha;
					z[f] += g[f] - sigma * w[f];
					n0[f] += g_2;
					g[f] = 0.0;
				}
			}

		}
			

		if (uit_stop)
			break;
	}
}


/* trn_ftrl:
 *   Train the model with the FTRL-Proximal algorithm described by Brendan et al.
 */
void trn_ftrl_parallel(mdl_t *mdl) {
	const uint64_t  F = mdl->nftr;
	const uint32_t  U = mdl->reader->nuni;
	const uint32_t  B = mdl->reader->nbi;
	const uint32_t  S = mdl->train->nseq;
	const uint32_t  K = mdl->opt->maxiter;
	const uint32_t  W = mdl->opt->nthread;


	info("    - Build the index\n");
	sgd_idx_t *idx  = xmalloc(sizeof(sgd_idx_t) * S);
	for (uint32_t s = 0; s < S; s++) {
		const seq_t *seq = mdl->train->seq[s];
		const uint32_t T = seq->len;
		uint64_t uobs[U * T + 1];
		uint64_t bobs[B * T + 1];
		uint32_t ucnt = 0, bcnt = 0;
		for (uint32_t t = 0; t < seq->len; t++) {
			const pos_t *pos = &seq->pos[t];
			for (uint32_t p = 0; p < pos->ucnt; p++)
				sgd_add(uobs, &ucnt, pos->uobs[p]);
			for (uint32_t p = 0; p < pos->bcnt; p++)
				sgd_add(bobs, &bcnt, pos->bobs[p]);
		}
		uobs[ucnt++] = none;
		bobs[bcnt++] = none;
		idx[s].uobs = xmalloc(sizeof(uint64_t) * ucnt);
		idx[s].bobs = xmalloc(sizeof(uint64_t) * bcnt);
		memcpy(idx[s].uobs, uobs, ucnt * sizeof(uint64_t));
		memcpy(idx[s].bobs, bobs, bcnt * sizeof(uint64_t));
	}
	

	info("      Done\n");

	uint32_t *perm = xmalloc(sizeof(uint32_t) * S);
	for (uint32_t s = 0; s < S; s++)
		perm[s] = s;

	
	

	double *z = xmalloc(sizeof(double) * F); 
	memset(z, 0, sizeof(double) * F);
	double *n0 = xmalloc(sizeof(double) * F); 
	memset(n0, 0, sizeof(double) * F);
	
	params_t *params[W];
	for (uint32_t i = 0; i < W; ++i) {
		params[i] = xmalloc(sizeof(params_t));
		params[i]->z = z;
		params[i]->n0 = n0;
		params[i]->mdl = mdl;
		params[i]->idx = idx;
		double *g = xmalloc(sizeof(double) * F);
		memset(g, 0, sizeof(double) * F);
		params[i]->grd_st = grd_stnew(mdl, g);
		params[i]->perm = perm;
	}
	
	double beta = mdl->opt->ftrl.beta;
	double alpha = mdl->opt->ftrl.alpha;
	double lambda1 = mdl->opt->ftrl.lambda1;
	double lambda2 = mdl->opt->ftrl.lambda2;

	printf("alpha=%f beta=%f lambda1=%f lambda2=%f\n", alpha, beta, lambda1, lambda2);
	mdl->opt->jobsize = 1;
	for (uint32_t k = 0, i = 0; k < K && !uit_stop; k++) {
		// First we shuffle the sequence by making a lot of random swap
		// of entry in the permutation index.
		for (uint32_t s = 0; s < S; s++) {
			const uint32_t a = rand() % S;
			const uint32_t b = rand() % S;
			const uint32_t t = perm[a];
			perm[a] = perm[b];
			perm[b] = t;
		}
		// And so, we can process sequence in a random order
		
		mth_spawn((func_t *)worker, W, (void **)params, mdl->train->nseq, mdl->opt->jobsize);
		if (uit_stop)
			break;
		// Repport progress back to the user
		if (!uit_progress(mdl, k + 1, -1.0))
			break;
	}
	
	// Cleanup allocated memory before returning
	for (uint32_t s = 0; s < S; s++) {
		free(idx[s].uobs);
		free(idx[s].bobs);
	}
	free(idx);
	free(perm);
	free(n0);
	free(z);
	for (uint32_t i = 0; i < W; ++i) {
		free(params[i]->grd_st->g);
		free(params[i]->grd_st);
		free(params[i]);
	}

}

