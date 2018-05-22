#ifndef _FCLAYER_H_
#define _FCLAYER_H_

typedef struct {
	TRAINING_LAYER_ELEMENTS;
	double** w;
	double* o;
	double* net;
	double (*f)(double);
	double (*df)(double);
} FCLayer;

#endif