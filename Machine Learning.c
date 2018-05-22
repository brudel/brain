#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#define TRAINING_LAYER_FUNCTIONS void (*foward)(struct TRAINING_LAYER*);	\
	void (*backward)(struct TRAINING_LAYER*);								\
	void (*store)(struct TRAINING_LAYER*);									\
	double* input;															\
	double* d_input;														\
	double* output;															\
	double* d_output;														\
	int n_in;																\
	int n_local;

typedef struct {
	TRAINING_LAYER_FUNCTIONS;
} training_layer;

typedef struct TRAINING_LAYER {
	int n_layers;
	double* input;
	training_layer** L;
} untraining_CNN;

untraining_CNN make_CNN(int n_layers, ...){
	int i, n_in, n_local;
	char c;
	va_list args;
	va_start(args, n_layers);
	untraining_CNN CNN;
	CNN.L = (training_layer**) malloc(sizeof(training_layer*) * n_layers);
	n_in = va_arg(args, int);
	CNN.input = (double*) malloc(sizeof(double) * n_in);

	for (i = 0; i < n_layers; ++i) {
		c = va_arg(args, char);

		if (c == 'f')
			CNN.L[i] = Make_FCL();
	}
}

////////////////////////////////FLC
typedef struct {
	double** w;
	double* o;
	double* net;
	double (*f)(double);
	double (*df)(double);
} FCLayer;

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

int main(int argc, char const *argv[]){
	return 0;
}