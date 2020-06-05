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

/*
 * Author: Xian Wu
 * E-mail: wuxian94@pku.edu.cn
 */

#include <inttypes.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wapiti.h"
#include "gradient.h"
#include "model.h"
#include "options.h"
#include "progress.h"
#include "tools.h"
#include "thread.h"
#include "vmath.h"

#define EPSILON (DBL_EPSILON * 64.0)

#define sign(v) ((v) < -EPSILON ? -1.0 : ((v) > EPSILON ? 1.0 : 0.0))
#define sqr(v)  ((v) * (v))

/******************************************************************************
 * Follow-the-regularized-Leader optimizer
 *
 *   This is an implementation of the FTRL algorithm described by McMahan,
 *   H. Brendan, et al. in [1].
 *
 *   [1] McMahan, H. Brendan, et al. "Ad click prediction: a view from the
 *       trenches." Proceedings of the 19th ACM SIGKDD international conference
 *       on Knowledge discovery and data mining. 2013.
 ******************************************************************************/
typedef struct ftrl_s ftrl_t;
struct ftrl_s {
	mdl_t  *mdl;
    double *z;
    double *n;
	double *g;
};

static void update_theta_worker(job_t *job, uint32_t id, uint32_t cnt, ftrl_t *st) {
	unused(job);
	mdl_t *mdl = st->mdl;
	const uint64_t F = mdl->nftr;
    // const double   alpha  = mdl->opt->ftrl.alpha;
    // const double   beta   = mdl->opt->ftrl.beta;
    // const double   rho1   = mdl->opt->rho1;
    // const double   rho2   = mdl->opt->rho2;
    double beta = mdl->opt->ftrl.beta;
	double alpha = mdl->opt->ftrl.alpha;
	double rho1 = mdl->opt->ftrl.lambda1;
	double rho2 = mdl->opt->ftrl.lambda2;

	double *x = mdl->theta;
    double *z   = st->z,    *n  = st->n;
	const uint64_t from = F * id / cnt;
	const uint64_t to   = F * (id + 1) / cnt;
    printf("id=%d from=%d to=%d\n", id, from, to);
	for (uint64_t f = from; f < to; f++) {
	    if (fabs(z[f]) - rho1 <= EPSILON) {
	        x[f] = 0;
	    } else {
	        double part1 = -1 / ((beta + sqrt(n[f])) / alpha + rho2);
            double part2 = 0;
	        if (sign(z[f])) {
                part2 = z[f] - rho1;
            } else {
                part2 = z[f] + rho1;
	        }
	        x[f] = part1 * part2;
	    }
	}
}

static void update_state_worker(job_t *job, uint32_t id, uint32_t cnt, ftrl_t *st) {
    unused(job);
    mdl_t *mdl = st->mdl;
    const uint64_t F = mdl->nftr;
    const double   alpha  = mdl->opt->ftrl.alpha;
    double *x = mdl->theta;
    double *z   = st->z,    *n  = st->n;
    double *g   = st->g;
    const uint64_t from = F * id / cnt;
    const uint64_t to   = F * (id + 1) / cnt;
    for (uint64_t f = from; f < to; f++) {
        double sigma = (sqrt(n[f] + g[f] * g[f])-sqrt(n[f]))/alpha;
        z[f] = z[f] + g[f] - sigma * x[f];
        n[f] = n[f] + g[f] * g[f];
    }
}

void trn_ftrl_wx(mdl_t *mdl) {
	const uint64_t F = mdl->nftr;
	const uint32_t K = mdl->opt->maxiter;
	const uint32_t W = mdl->opt->nthread;
	// Allocate state memory and initialize it
	double *z = xvm_new(F), *n  = xvm_new(F);
	double *g = xvm_new(F);
	for (uint64_t f = 0; f < F; f++) {
		z[f]  = 0.0;
		n[f]  = 0.0;
	}
	// Prepare the ftrl state used to send information to the ftrl worker
	// about updating weight using the gradient.
	ftrl_t *st = xmalloc(sizeof(ftrl_t));
	st->mdl = mdl;
	st->z = z;  st->n  = n;
	st->g = g;
    ftrl_t *ftrl[W];
	for (uint32_t w = 0; w < W; w++)
        ftrl[w] = st;
	// Prepare the gradient state for the distributed gradient computation.
	grd_t *grd = grd_new(mdl, g);
	// And iterate the gradient computation / weight update process until
	// convergence or stop request
	for (uint32_t k = 0; !uit_stop && k < K; k++) {
	    // 根据论文，模型参数的训练分为参数更新和状态更新两个步骤
        mth_spawn((func_t *)update_theta_worker, W, (void **)ftrl, 0, 0);
        double fx = grd_gradient(grd);
		if (uit_stop)
			break;
		mth_spawn((func_t *)update_state_worker, W, (void **)ftrl, 0, 0);
		if (uit_progress(mdl, k + 1, fx) == false)
			break;
	}
	// Free all allocated memory
	xvm_free(g);
	xvm_free(z);
	xvm_free(n);
	grd_free(grd);
	free(st);
}