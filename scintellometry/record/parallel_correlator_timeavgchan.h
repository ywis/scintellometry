/* Program for raw voltage dumping -- Written by Jayanta Roy (Last modified on 31 aug 2007)*/
#ifndef CORRELATOR_H
#define CORRELATOR_H
/* header file correlator.h */

#include "/opt/intel/ipp/5.2/ia32/include/ipp.h"
#ifndef __NEWCORR_H__
#include "newcorr.h"
#endif

#define DEBUG
#undef DEBUG
//#define GATHER
//#define AMP_PHAS
#define RAW_DUMP

#define FFTLEN 8192 
#define NCHAN 4
#define CORRLEN (NCHAN*FFTLEN)
#define NUM_PROCESSES 2
#define NUM_CORR 1
#define NNODECORR 1

#define NCORR (NUM_CORR*NCHAN*(NUM_CORR*NCHAN+1)/2)
//#define M_PI 3.141592654
#define ACQ_LEN 32*1024*1024
#define UNPACK_LEN 32*1024*1024
#define FRNG_STEP 0.25
#define NSTEP 2880

#define tx_scale 1
#define compute_scale 1
#define MPI_BUF_CHUNK (ACQ_LEN*tx_scale)/NUM_CORR


#define SWEEPS_PER_DUMP 1
#define CORR_SIZE FFTLEN*NCHAN*NUM_CORR*(NCHAN*NUM_CORR+1)/2

#define NTCHUNK 16
#define FFTBLOCK 64

static int num_acq_chunks_node[NUM_CORR];
static int recv_count[NUM_PROCESSES];
static int send_count[NUM_PROCESSES];
static float corrbuf[CORR_SIZE];
static float corr_sum[CORR_SIZE];
static float corr_sumlfft[CORR_SIZE/2];
static float rdelay_ti[NCHAN*NUM_CORR];
static float phase_ti[2][NUM_CORR*NCHAN*FFTLEN/2];
static float dphase_t0[2][NUM_CORR*NCHAN*FFTLEN/2];
static float rphase_ti[NUM_CORR*NCHAN*FFTLEN/2];

typedef float corrarray_t[NCORR][FFTLEN];
typedef float datarray_t[NCHAN*NUM_CORR][FFTLEN];
/*
typedef struct AntennaParType;
typedef struct CorrType;
typedef struct DasParType;
typedef struct ModelParType;
typedef struct SourceParType;
*/

CorrType corr;
DasParType daspar;
ModelParType mpar;
SourceParType source;

//AntennaParType antenna [30];

corrarray_t corrampphas;
corrarray_t corrbufavg;
corrarray_t corrbuf_acc;


void corr_driver(signed char *buffer, int dump_bit, int nint_corr, struct timeval timestamps, double *ch_freq, int w_buff, int fstop, float step, short int *phase);
void correlate(short int obuf[FFTLEN/FFTBLOCK][UNPACK_LEN/(FFTLEN*NCHAN*NUM_CORR)][NUM_CORR][NCHAN][FFTBLOCK], float corrbuf[FFTLEN/8][NCORR][2][4]);
//void correlate(float *inobuf, float *outcorrbuf);
//void correlate(float *fbuf, float corrbuf[FFTLEN/8][NCORR][2][4]);
void convcorr(float corrbuf[CORR_SIZE],float outcorrbuf[FFTLEN/8][NCORR][2][4]);
void write_corr(float *corr_buf, struct tm* local_t, double time_ms);
void get_antenna(FILE *f, AntennaParType *antenna);
int get_sampler(FILE *f, CorrType *corr);
double mjd_cal(struct tm *t);
double lmst(double mjd);
void calModelPar(double tm, CorrType *corr, ModelParType *mpar, SourceParType *source, int *antmask);
#endif

