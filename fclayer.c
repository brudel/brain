#include <stdlib.h>

training_layer* FCL_make(int n_in, int n_local) {
	FCLayer* L = (FCLayer*) malloc(sizeof(FCLayer));

	L->foward = (*)(struct training_layer*) FCL_foward;
	L->backward = (*)(struct training_layer*) FCL_backward;
	L->store = (*)(struct training_layer*) FCL_store;
	L->input = (double*) malloc(sizeof(double) * n_in);
	L->d_input = (double*) malloc(sizeof(double) * n_in);
	L->n_in = n_in;
	L->n_local = n_local;

	return (training_layer*) L;
}

void FCL_foward(FCLayer* L){
	int i;

	for (i = 0; i < L->n_neural; ++i) {
		L->net[i] = produto_escalar(L->input, L->w[i], L->n_in) + L->o[i];
		L->output[i] = L->f(L->net[i]);
	}
}

void FCL_backward(FCLayer* L){
	int i, j;

	for (i = 0; i < L->n_neural; ++i) {
		L->d_output[i] *= L->df(L->net[i]);
		for (j = 0; j < L->n_in; ++j)
			L->w[i][j] -= L->input[j] * L->d_output[i];

		L->o[i] -= L->d_output[i];
	}

	for (i = 0; i < L->n_in; ++i) {
		L->d_input[i] = 0;
		for (j = 0; j < L->n_in; ++j)
			L->d_input[i] += L->w[j][i] * L->d_output[j];
	}
}