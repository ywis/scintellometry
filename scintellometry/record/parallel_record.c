/* Program for raw voltage dumping -- Written by Jayanta Roy (Last modified on 31 aug 2007)
   Latest modification done on 06 dec 2007 -- Jayanta Roy
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
#include <strings.h>
#include <string.h>
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
	FILE *ftsamp[4];
	FILE *fseq;
	int i,j,k,l,m,io,iot,imp,acqcnt;
	int acq_chunk_offset, node_chunk_offset;
	static int firsttime=1, nchunk;
	static int nbuff = 0;
	static int iteration = 0;
	static int g_size = 2;
	static int nint = 0;
	static int rankcnt,nodenum;
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
        int fpn[4];
  	nchunk=ACQ_LEN/(decimate*CORRLEN);

	if (firsttime == 1) {
                char hostname[255];
                char *suffix;
                gethostname(hostname, 255);
                suffix=strstr(hostname,"node")+4;
                sscanf(suffix,"%d",&nodenum);
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
                timestamp.tv_usec=timestamp.tv_usec>>8;
                timestamp.tv_usec=timestamp.tv_usec<<8;
                timestamp.tv_usec+=nodenum;

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
                fpn[0] = open("/mnt/a/gsbuser/VLBI/raw_voltage.all.test_2_july2.1.dat", O_CREAT|O_TRUNC|O_WRONLY|O_SYNC,S_IRUSR|S_IWUSR);
                if (fpn[0]<0 ) perror(" raw voltage open");
                fpn[1] = open("/mnt/b/gsbuser/VLBI/raw_voltage.all.test_2_july2.2.dat", O_CREAT|O_TRUNC|O_WRONLY|O_SYNC,S_IRUSR|S_IWUSR);
                if (fpn[1]<0 ) perror(" raw voltage open");
                fpn[2] = open("/mnt/c/gsbuser/VLBI/raw_voltage.all.test_2_july2.3.dat", O_CREAT|O_TRUNC|O_WRONLY|O_SYNC,S_IRUSR|S_IWUSR);
                if (fpn[2]<0 ) perror(" raw voltage open");
                fpn[3] = open("/mnt/d/gsbuser/VLBI/raw_voltage.all.test_2_july2.4.dat", O_CREAT|O_TRUNC|O_WRONLY|O_SYNC,S_IRUSR|S_IWUSR);
                if (fpn[3]<0 ) perror(" raw voltage open");
		ftsamp[0]=fopen("/mnt/a/gsbuser/VLBI/timestamp_voltage.all.test_2_july2.1.dat","w");
                if (ftsamp[0]==NULL ) perror(" timestamp open");
		ftsamp[1]=fopen("/mnt/b/gsbuser/VLBI/timestamp_voltage.all.test_2_july2.2.dat","w");
                if (ftsamp[1]==NULL ) perror(" timestamp open");
		ftsamp[2]=fopen("/mnt/c/gsbuser/VLBI/timestamp_voltage.all.test_2_july2.3.dat","w");
                if (ftsamp[2]==NULL ) perror(" timestamp open");
		ftsamp[3]=fopen("/mnt/d/gsbuser/VLBI/timestamp_voltage.all.test_2_july2.4.dat","w");
                if (ftsamp[3]==NULL ) perror(" timestamp open");
		fseq=fopen("/mnt/a/gsbuser/VLBI/sequence.all.test_2_july2.dat","w");
		omp_set_num_threads(4);	
#pragma omp parallel for default(none) private(time_string,time_ms,local_t,tm1,start,icount,imp,current,sec) shared(fpn,ilen,rbuf,iteration,stderr,myrank,timestampbuf,ftsamp,fseq) schedule(dynamic,1)
                for (iter=2;iter<10000000;iter++) {
                int nodenumremote;
#pragma omp flush (iteration)
		if (iteration-iter>NBUF*3/4) {
			fprintf(stderr,"BUFFER LOST, iter=%d iteration=%d\n",iter,iteration);
			continue;
			}
			imp=omp_get_thread_num();
#pragma omp critical
		{
			fprintf(fseq,"%d %d\n",iter,imp%4);
			fflush(fseq);
		}
		while(iteration < iter+1) {
			usleep(10000);
#pragma omp flush (iteration)
		}
#pragma omp flush (rbuf)
	        ftime(&start);

			printf("iter=%d thread=%d \n",iter,imp);
        	icount=write(fpn[imp%4],rbuf[iter%NBUF],ilen);
		if (icount != ilen) {
			fprintf(stderr,"only wrote %d of %d bytes\n",icount,ilen);
			sleep(360000);
		}
		time_ms = timestampbuf[iter%NBUF].tv_usec/1000000.000;
                nodenumremote = timestampbuf[iter%NBUF].tv_usec&0xff;
		local_t = localtime(&timestampbuf[iter%NBUF].tv_sec);
		tm1 = local_t->tm_sec + local_t->tm_min*60 + local_t->tm_hour*3600 + time_ms;
        strftime (time_string, sizeof (time_string), "%Y %m %d %H %M %S", local_t);
        if (fprintf(ftsamp[imp%4],"%s %lf %d\n", time_string, time_ms,nodenumremote)<0) perror("fprintf");
	fflush(ftsamp[imp%4]);
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
