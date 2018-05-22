#ifndef _CNN_H_
#define _CNN_H_

#define TRAINING_LAYER_ELEMENTS void (*foward)(struct TRAINING_LAYER*);	\
	void (*backward)(struct TRAINING_LAYER*);								\
	void (*store)(struct TRAINING_LAYER*);									\
	double* input;															\
	double* d_input;														\
	double* output;															\
	double* d_output;														\
	int n_in;																\
	int n_local;

typedef struct {
	TRAINING_LAYER_ELEMENTS;
} training_layer;

typedef struct TRAINING_LAYER {
	int n_layers;
	double* input;
	double* output;
	double* d_output;
	training_layer** L;
} untraining_CNN;

#endif