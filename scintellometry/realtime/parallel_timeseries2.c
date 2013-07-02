/* Program for raw voltage squaring and turning into a time series 
   based on rawdump code by Jayanta Roy and Ue-Li
   Latest modification 26 jun 2013 -- Marten van Kerkwijk
*/

#include <stdio.h>
#include <time.h>
#include <sys/timeb.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "mpi.h"
#include "omp.h"
#include <xmmintrin.h>
#include <emmintrin.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "parallel_correlator_timeavgchan.h"
#include "newcorr.h"

#define decimate 1

#define NBUF 50
static inline u_int64_t
rdtsc ()
{         u_int64_t d;
          __asm__ __volatile__ ("rdtsc" : "=&A" (d));
          return d;
}

static u_int64_t start_dcal, end_dcal;
static u_int64_t start_fft, end_fft;
static u_int64_t start_mac, end_mac;
static u_int64_t start_gather, end_gather;
static u_int64_t dcal_count = 0, fft_count = 0, mac_count = 0, gather_count = 0;

static MPI_Request request[2*NUM_PROCESSES*NNODECORR];
static MPI_Status status[2*NUM_PROCESSES*NNODECORR];
const int MPIWAITCHUNK=1;

int nrecv=0;
MPI_Comm MPI_COMM_GROUP1, MPI_COMM_G1, MPI_COMM_G2,MPI_COMM_G3,MPI_COMM_G4,MPI_COMM_G5,MPI_COMM_G6,MPI_COMM_G7,MPI_COMM_G8,MPI_COMM_G9,MPI_COMM_G10,MPI_COMM_G11,MPI_COMM_G12,MPI_COMM_G13,MPI_COMM_G14,MPI_COMM_G15;
MPI_Group group1, g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15;

void corr_driver(signed char *buffer, int dump_bit, int nint_corr, struct timeval timestamp, double *ch_freq, int w_buff, int fstop, float step, short int *phase)
	
{	register __m128i r0,r1,r2,r3,r4,r5,r6,r7;
        register __m128 r11,r22,r33,r44,r55,r66,r77,r88;

	__declspec(align(128)) static char rbuf[NBUF][ACQ_LEN];
	__declspec(align(128)) static char muxbuf[2][NCHAN][ACQ_LEN/NCHAN];
	__declspec(align(128)) static char obuf[2][NCHAN][ACQ_LEN/NCHAN];

	struct timeb start, current;
	double sec;
	struct timeval timestampbuf[NBUF];
	int numthreads, mythread, itarget, isource, nstep1,nstep2,nstep3,nstep4; 
	static int numprocs, myrank, numnode, ranknode;
	FILE *fseq, *fpn[4];
	int i,j,k,l,m,io,iot,imp,acqcnt;
	int acq_chunk_offset, node_chunk_offset;
	static int firsttime=1, nchunk;
	static int nbuff = 0;
	static int iteration = 0;
	static int g_size = 2;
	static int nint = 0;
	static int rankcnt;
	static int r_array[(NUM_PROCESSES-1)*2];
	static float delay_t0[NCHAN*NUM_CORR],delay_ti[2][NCHAN*NUM_CORR],dd_t0[2][NCHAN*NUM_CORR];
	static float acq_chunktime, ffttime, pi, tstep, blocktime;
	static double tm0, time_sec = 0;
	struct tm* local_t;
	char time_string[40], *time_ptr;
	double mjd, time_ms, tm1, time_usec;
	int shuffle = 255;
	long buff_cnt=0;
        int fp1,fp2, fp3, fp4;
	long long int sum1, sum2[2], sum4[4], sum8[8];
	int ib, power;

  	nchunk=ACQ_LEN/(decimate*CORRLEN);

	if (firsttime == 1) {
		MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
		MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
		omp_set_nested(1);

	#ifdef GATEHR
		r_array[0] = 0;
		r_array[1] = NUM_CORR;
		r_array[2] = NUM_CORR+1;
		r_array[3] = NUM_CORR+2;
		r_array[4] = NUM_CORR+3;
		r_array[5] = NUM_CORR+4;
		r_array[6] = NUM_CORR+5;
		r_array[7] = NUM_CORR+6;
		r_array[8] = NUM_CORR+7;
		r_array[17] = NUM_CORR;
		r_array[18] = NUM_CORR+2;
		r_array[19] = NUM_CORR+4;
		r_array[20] = NUM_CORR+6;
		r_array[25] = NUM_CORR;
		r_array[26] = NUM_CORR+4;

		MPI_Comm_group(MPI_COMM_WORLD,&g0);
		MPI_Group_incl(g0,(NUM_PROCESSES/2+1),&r_array[0],&group1);
		MPI_Group_incl(g0,g_size,&r_array[1],&g1);
		MPI_Group_incl(g0,g_size,&r_array[3],&g2);
		MPI_Group_incl(g0,g_size,&r_array[5],&g3);
		MPI_Group_incl(g0,g_size,&r_array[7],&g4);
                MPI_Group_incl(g0,g_size,&r_array[17],&g9);
                MPI_Group_incl(g0,g_size,&r_array[19],&g10);
                MPI_Group_incl(g0,g_size,&r_array[25],&g13);

		MPI_Comm_create(MPI_COMM_WORLD,group1,&MPI_COMM_GROUP1);
		MPI_Comm_create(MPI_COMM_WORLD,g1,&MPI_COMM_G1);
		MPI_Comm_create(MPI_COMM_WORLD,g2,&MPI_COMM_G2);
		MPI_Comm_create(MPI_COMM_WORLD,g3,&MPI_COMM_G3);
		MPI_Comm_create(MPI_COMM_WORLD,g4,&MPI_COMM_G4);
                MPI_Comm_create(MPI_COMM_WORLD,g9,&MPI_COMM_G9);
                MPI_Comm_create(MPI_COMM_WORLD,g10,&MPI_COMM_G10);
                MPI_Comm_create(MPI_COMM_WORLD,g13,&MPI_COMM_G13);
	#endif	

		blocktime = (((32*1.024*1.024)/4)/100)*3;
		acq_chunktime = blocktime/NUM_CORR;
                ffttime = acq_chunktime/(nchunk/NUM_CORR);  
		pi = 2.0*M_PI;		

#ifdef RAW_DUMP
		if(myrank >= numprocs/2)	{
		  /* fpn[0] = fopen("full_sample_power.2.dat", "w");
		  if (fpn[0]==NULL) perror(" raw voltage open");
		  fpn[1] = fopen("half_sample_power.2.dat", "w");
		  if (fpn[1]==NULL ) perror(" raw voltage open"); */
		  fpn[2] = fopen("quarter_sample_power.2.dat", "w");
		  if (fpn[2]==NULL ) perror(" raw voltage open");
		  /* fpn[3] = fopen("oneeighth_sample_power.2.dat", "w");
		  if (fpn[3]==NULL ) perror(" raw voltage open");
		  fseq=fopen("timestamp.2.dat","w");
		  if (fseq==NULL ) perror(" timestamp open"); */
		}

#endif		
	}
		
		ftime(&start);
                // NODE1 has to broadcast its timestamp
		        //MPI_Bcast(&timestamp,sizeof(struct timeval),MPI_CHAR,0,MPI_COMM_WORLD);
		
		ftime(&current);
                sec = (((current.time - start.time) * 1000) + (current.millitm - start.millitm)) / 1000.0;
		

        //MPI_Barrier(MPI_COMM_WORLD);

	if(firsttime > 2)  // IGNORE FIRST BUFFER
	{
		ftime(&start);
		if(myrank>=numprocs/2){		
	
        	//#ifdef DEBUG
                	start_dcal = rdtsc ();
        	//#endif
		
		time_ms = timestamp.tv_usec/1000000.000;
		local_t = localtime(&timestamp.tv_sec);
		tm1 = local_t->tm_sec + local_t->tm_min*60 + local_t->tm_hour*3600 + time_ms;
       
		}	
	//OPENMP threading loop start here
	
	omp_set_num_threads(2);	
	#pragma omp parallel default(shared) private(start,current)
	#pragma omp sections nowait 
	{       
	{	ftime(&start);
			
		// NETWORK DATA SHARING IN TDM MODE
		
		if (myrank<numprocs/2){
		printf("send: iteration=%d\n",iteration);
                itarget=(myrank)%(numprocs/2)+numprocs/2;
		        MPI_Send(&timestamp,sizeof(struct timeval),MPI_CHAR,itarget,0,MPI_COMM_WORLD);
                MPI_Isend(buffer, ACQ_LEN*4/4, MPI_CHAR, itarget, 100, MPI_COMM_WORLD, request);
                MPI_Waitall(1,request,status);
		//printf("done send: iteration=%d\n",iteration);
		}
                else{
                itarget=(NUM_CORR+myrank)%(numprocs/2);
		while(1){
		printf("recv: iteration=%d\n",iteration);
		        MPI_Recv(&timestamp,sizeof(struct timeval),MPI_CHAR,itarget,0,MPI_COMM_WORLD,status);
			timestampbuf[iteration%NBUF]=timestamp;
                MPI_Irecv(rbuf[iteration%NBUF], ACQ_LEN*4/4, MPI_CHAR, itarget, 100, MPI_COMM_WORLD,request);
                MPI_Waitall(1,request,status);
		iteration++;
#pragma omp flush (iteration,rbuf)
		}
		}

                ftime(&current);
		sec = (((current.time - start.time) * 1000) + (current.millitm - start.millitm)) / 1000.0;
  		printf("$MPI %d %d 	%d ms\n",myrank, iteration,(int)(sec*1000));
		//#endif
	}
	


	#pragma omp section
        {       int icount,ilen,iter;
		ftime(&start);
                if(myrank>=numprocs/2 ){
		const int nstripe=32;

		ilen=ACQ_LEN;
		printf("write section\n");
		omp_set_num_threads(1);	
		//#pragma omp parallel for default(none) private(time_string,time_ms,local_t,tm1,start,icount,imp,current,sec,ib,sum1,sum2,sum3,sum4,power) shared(fpn,ilen,rbuf,iteration,stderr,myrank,timestampbuf,fseq) schedule(dynamic,1)
                for (iter=2;iter<10000000;iter++) {
#pragma omp flush (iteration)
		if (iteration-iter>NBUF*3/4) {
			fprintf(stderr,"BUFFER LOST, iter=%d iterarion=%d\n",iter,iteration);
			continue;
			}
			imp=omp_get_thread_num();
		while(iteration < iter+1) {
			usleep(10000);
#pragma omp flush (iteration)
		}
#pragma omp flush (rbuf)
	        ftime(&start);

		printf("iter=%d thread=%d \n",iter,imp);
		/* do the work here */
		sum1 = 0;
		sum2[0] = sum2[1] = 0;
		sum4[0] = sum4[1] = sum4[2] = sum4[3] = 0;
		sum8[0] = sum8[1] = sum8[2] = sum8[3] = sum8[4] = sum8[5] = sum8[6] = sum8[7] = 0;
                for(ib=0;ib<ACQ_LEN;ib++){
		  power = rbuf[iter%NBUF][ib];
		  power *= power;
		  /* sum1 += power;
		     sum2[ib/(ACQ_LEN/2)] += power; */
		  sum4[ib/(ACQ_LEN/4)] += power;
		  // sum8[ib/(ACQ_LEN/8)] += power;
		}
		/* get timestamp out to timestamp file */
		time_ms = timestampbuf[iter%NBUF].tv_usec/1000000.000;
		local_t = localtime(&timestampbuf[iter%NBUF].tv_sec);
		tm1 = local_t->tm_sec + local_t->tm_min*60 + local_t->tm_hour*3600 + time_ms;
        strftime (time_string, sizeof (time_string), "%Y %m %d %H %M %S", local_t);
	//#pragma omp critical
	/*	{
		  if (fprintf(fseq,"%s %lf \n", time_string, time_ms)<0) perror("fprintf");
		  fflush(fseq);
		  } */
		/* fprintf(fpn[0], "%s %lf\n%lld\n", time_string, time_ms, sum1);
		fprintf(fpn[1], "%s %lf\n%lld\n%lld\n", time_string, time_ms, 
		sum2[0], sum2[1]); */
		fprintf(fpn[2], "%s %lf\n%lld\n%lld\n%lld\n%lld\n", 
			time_string, time_ms, sum4[0], sum4[1], 
			sum4[2], sum4[3]);
		/* fprintf(fpn[3], "%s %lf\n", time_string, time_ms);
                for(ib=0;ib<8;ib++){
		  fprintf(fpn[3], "%lld\n", sum8[ib]);
		}
                for(ib=0;ib<4;ib++){
                  printf("fflush status=%d\n", fflush(fpn[ib]));
		  } */
		fflush(fpn[2]);
                sync();
	ftime(&current);
        sec = (((current.time - start.time) * 1000) + (current.millitm - start.millitm)) / 1000.0;
        printf("#WRITE %d %d    %d    			%d ms\n",myrank, imp,iter,(int)(sec*1000));
		}


	} //mpyrank if
	} // omp sections




	
        } // main parallel loop
	} // ignore first buffer
        iteration++;
 
    	firsttime++;	
}	
