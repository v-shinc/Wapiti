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

#include "wapiti.h"
#include "gradient.h"
#include "model.h"
#include "options.h"
#include "progress.h"
#include "sequence.h"
#include "tools.h"
#include "vmath.h"

/******************************************************************************
 * The SGD-L1 trainer
 *
 *   Implementation of the stochatic gradient descend with FTRL (Per-Coordinate 
 *   FTRL-Proximal with L1 and L2 ) described
 *   in [1]. This allow to build really sparse models with the
 *   SGD method.
 *
 *   [1] Ad Click Prediction: a View from the Trenches
 * 	 ./wapiti train  -T crf -a ftrl --ftrl_alpha 1 --lambda1 15 --lambda2 1 --stopeps 0.005 -p template.txt  -d eval2.txt train2.txt  model
 * 
 ******************************************************************************/
typedef struct sgd_idx_s {
	uint64_t *uobs;
	uint64_t *bobs;
} sgd_idx_t;

/* applypenalty:
 *   This macro is quite ugly as it make a lot of things and use local variables
 *   of the function below. I'm sorry for this but this is allow to not
 *   duplicate the code below. Due to the way unigrams and bigrams observation
 *   are stored we must use this two times. As this macro is dangerous when
 *   called outsize of sgd-l1 we undef it just after.
 *   This function match exactly the APPLYPENALTY function defined in [1] pp 481
 *   and the formula on the middle of the page 480.
 */


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

/* trn_ftrl:
 *   Train the model with the FTRL-Proximal algorithm described by Brendan et al.
 */
void trn_ftrl(mdl_t *mdl) {
	const uint64_t  Y = mdl->nlbl;
	const uint64_t  F = mdl->nftr;
	const uint32_t  U = mdl->reader->nuni;
	const uint32_t  B = mdl->reader->nbi;
	const uint32_t  S = mdl->train->nseq;
	const uint32_t  K = mdl->opt->maxiter;
	      double   *w = mdl->theta;
	// First we have to build and index who hold, for each sequences, the
	// list of actives observations.
	// The index is a simple table indexed by sequences number. Each entry
	// point to two lists of observations terminated by <none>, one for
	// unigrams obss and one for bigrams obss.
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
	// double *z = xmalloc(sizeof(double) * F); 
	double *z = xvm_new(F);
	memset(z, 0, sizeof(double) * F);
	// double *n0 = xmalloc(sizeof(double) * F);
	double *n0 = xvm_new(F);
	memset(n0, 0, sizeof(double) * F);
	// double *sigma = xmalloc(sizeof(double) * F);
	// memset(sigma, 0, sizeof(double) * F);
	
	info("      Done\n");
	// We will process sequences in random order in each iteration, so we
	// will have to permute them. The current permutation is stored in a
	// vector called <perm> shuffled at the start of each iteration. We
	// just initialize it with the identity permutation.
	// As we use the same gradient function than the other trainers, we need
	// an array to store it. These functions accumulate the gradient so we
	// need to clear it at start and before each new computation. As we now
	// which features are active and so which gradient cell are updated, we
	// can clear them selectively instead of fully clear the gradient each
	// time.
	// We also need an aditional vector named <q> who hold the penalty
	// already applied to each features.
	uint32_t *perm = xmalloc(sizeof(uint32_t) * S);
	for (uint32_t s = 0; s < S; s++)
		perm[s] = s;
	double *g = xmalloc(sizeof(double) * F);
	for (uint64_t f = 0; f < F; f++)
		g[f] = 0.0;
	// We can now start training the model, we perform the requested number
	// of iteration, each of these going through all the sequences. For
	// computing the decay, we will need to keep track of the number of
	// already processed sequences, this is tracked by the <i> variable.
	double beta = mdl->opt->ftrl.beta;
	double alpha = mdl->opt->ftrl.alpha;
	double lambda1 = mdl->opt->ftrl.lambda1;
	double lambda2 = mdl->opt->ftrl.lambda2;
	printf("alpha=%f beta=%f lambda1=%f lambda2=%f\n", alpha, beta, lambda1, lambda2);
	grd_st_t *grd_st = grd_stnew(mdl, g);
	for (uint32_t k = 0, i = 0; k < K && !uit_stop; k++) {
		// First we shuffle the sequence by making a lot of random swap
		// of entry in the permutation index.
		// for (uint32_t s = 0; s < S; s++) {
		// 	const uint32_t a = rand() % S;
		// 	const uint32_t b = rand() % S;
		// 	const uint32_t t = perm[a];
		// 	perm[a] = perm[b];
		// 	perm[b] = t;
		// }
		// And so, we can process sequence in a random order
		
		for (uint32_t sp = 0; sp < S && !uit_stop; sp++, i++) {
			const uint32_t s = perm[sp];
			const seq_t *seq = mdl->train->seq[s];

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
			grd_dospl(grd_st, seq);
			
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
		// Repport progress back to the user
		if (!uit_progress(mdl, k + 1, -1.0))
			break;
	}
	grd_stfree(grd_st);
	// Cleanup allocated memory before returning
	for (uint32_t s = 0; s < S; s++) {
		free(idx[s].uobs);
		free(idx[s].bobs);
	}
	free(idx);
	free(perm);
	free(g);
	// free(n0);
	// free(z);
	xvm_free(n0);
	xvm_free(z);

}
#undef applypenalty

