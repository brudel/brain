#include <stdlib.h>
#include <stdarg.h>

untraining_CNN make_CNN(int n_layers, int n_out, ...){
	int i;
	va_list args;
	va_start(args, n_layers);
	untraining_CNN CNN;
	CNN.n_layers;
	CNN.L = (training_layer**) malloc(sizeof(training_layer*) * n_layers);

	CNN.output = (double*) malloc(sizeof(double) * n_out);
	CNN.d_output = (double*) malloc(sizeof(double) * n_out);
	double* output = CNN.output, *d_output = CNN.d_output;

	for (i = n_layers; i >= 0; --i) {
		CNN.L[i] = va_arg(args, training_layer*);

		CNN.L[i]->output = output;
		CNN.L[i]->d_output = d_output;

		output = CNN.L[i]->input;
		d_output = CNN.L[i]->d_input;
	}

	CNN.input = output;
}