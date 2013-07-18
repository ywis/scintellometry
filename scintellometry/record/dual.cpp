//-----------------------------------------------------------------------------
//title: Acquisition Logic A/D Board Kernel Driver Checkout Program
//version: Linux 1.0
//date:  March 2004
//designer: Michael Wyrick                                                   
//programmer: Michael Wyrick                                                
//platform: Linux 2.4.x
//language: GCC 3.3.1
//module: waveform.cpp
//-----------------------------------------------------------------------------
//  Purpose: test transfers to the A/D Boards
//  Docs:                                  
//-----------------------------------------------------------------------------
//  
/* Final version of the code -- Written by Jayanta Roy (Last modified on 31 aug 2007)
   Latest modification done on 06 dec 2007 -- Jayanta Roy  
*/
//-----------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <string.h>

#include "aldriver.h"
#include "Board4108.h"
#include "BoardEnum.h"
#include "reg.h"

#ifdef PARALLEL
#include <iostream>
#include "mpi.h"
int rank,size;
extern "C" {
#include "parallel_correlator_timeavgchan.h"
}
#endif
#include "ipps.h"
#include "ipp.h"
#include "newcorr.h"

#define PROC_DATA_FORMAT 1001

//extern long BufferSize;

#define PROFILE
#undef PROFILE

#ifdef PARALLEL
const unsigned int AcqSize = 32*1024*1024;
const unsigned int BufferSize = 64*1024*1024;
#endif
#define MAX(x,y) ((x>y)?x:y)


int AcqNum = 0;

int WhichBuffer = 1;
long amountofdata = 0;

static inline u_int64_t
rdtsc ()
{         u_int64_t d;
          __asm__ __volatile__ ("rdtsc" : "=&A" (d));
          return d;
}
u_int64_t start_tsc, end_tsc;
u_int64_t write_tsc, endwrite_tsc;
u_int64_t pack_tsc, endpack_tsc;
u_int64_t trans_count = 0, write_count = 0, pack_count = 0;
struct timeval timestamp;
struct timezone noreq;
char str[100], *str_ptr, phase_str[40], *phase_ptr;
float src_ra, src_dec, step;
long obs_freq, BW;
double ch_freq[FFTLEN/2];
short int phase[NSTEP];

MPI_Comm MPI_COMM_GROUP, MPI_COMM_GROUP1, MPI_COMM_GROUP2;
MPI_Group group0, group1, group2;


//------------------------------------------------------------------------
//
// This gets called for data returned from the board
//
//------------------------------------------------------------------------
void ProcessData(unsigned char *data, int dump_bit, int nint_corr, int buff_num, int fstop , float step, short int *phase, int format) {
  switch (format) {
#ifdef PARALLEL
	case 1001:
		double *ch = ch_freq;
	       	corr_driver((signed char*)data,dump_bit,nint_corr,timestamp,ch,buff_num,fstop,step,phase);
        	break;
#endif
  		 }
}

//------------------------------------------------------------------------
// Wait for the Board to be ready for a Trigger
//------------------------------------------------------------------------
void WaitForBoardToBeReady(AL4108 *brd) {
  while(brd->TRIGREADY->Get() == 0) {
    usleep(1);
  }
}

//------------------------------------------------------------------------
// 
//------------------------------------------------------------------------
void DoTrigger(AL4108 *brd) {
  brd->TriggerAcq();          // Trigger the Next Acquisition
}

//------------------------------------------------------------------------
// Setup Acquisition
//------------------------------------------------------------------------
void SetupAcq(AL4108 *brd, int mode, int sam_bit, int sam_clk, int dump_bit, int loops, int nint_corr, int buff_num, int fstop, float step, short int *phase)
{ 
  int ftable[256];           // conversion table from unsigned char to 2's complement data
  int Totalrun;     	     //number of total runs
  int t, i, deci;
  int byte1[20000], byte2[20000];
  struct timeb start, finish;
  long buff_cnt=0;
  static int lp_cnt = 1;
  static float blocktime;
  int clk_sel;
  struct tm* local_t;
  char time_string[40];
  double time_ms;
  Totalrun = loops+1; 	 

#ifndef RAW_DUMP
  if(rank>=size/2){ 
  for(int chan=0;chan<FFTLEN/2;chan++)
  	ch_freq[chan] = source.freq[0] + BW/(FFTLEN/2)*chan;
  }//rank if
#endif

  if(sam_bit==4 && sam_clk==66) clk_sel = 1;
  if(sam_bit==4 && sam_clk==33) clk_sel = 1;

  for(i=0;i<128;i++) ftable[i]=i;
  for(i=128;i<256;i++) ftable[i]=i-256;  

  if(rank<size/2){
  if(sam_bit==8 && sam_clk==33) deci = 1;
  else deci = 0;
  printf("Setting Decimate 3 to %i\n", deci);
  brd->DECIMATE3->Set(deci);

  printf("Setting up Acquisition\n");
  // Setup and Run an Acquisition
  
    if(sam_bit==8 && sam_clk==33)
  {     printf("Current mode is 4-CHANNEL 8-BIT 33.33MSPS for rank %d\n", rank);
        brd->TWOTHIRD->Set(0);  //setting of TWOTHIRD reg to 0/1 use the other format setup regs
        brd->AMODE->Set(0x30);
  }    // Setting up 8bit 32MSPS acq mode and set decimate to 1
  if(sam_bit==4 && sam_clk==66 )
  {
        printf("Current mode is 4-CHANNEL 4-BIT 66.66MSPS for rank %d\n", rank);
        brd->AMODE->Set(0x20);    //setting of TWOTHIRD reg to 2 set 2/3 AD sampling rate and ignore all other format setup regs
        brd->TWOTHIRD->Set(2);
  }  // Setting up for 4bit 66MSPS (2/3 AD sampling rate) 4 channel mode

  brd->SetupAcq(AcqSize / 4);
  brd->EnableAcq();

  printf("Setting up DMA\n");

  // Turn on the Circle Buffer with two Interrupts
  brd->TurnOnOffCircleBuffer(1, 2);  

    printf("done circle buffer rank %d\n",rank);
  brd->ClearBufferStatus(0);
  brd->ClearBufferStatus(1);
    printf("done clear buffer rank %d\n",rank);

  // Turn on the DMA with two buffers
  brd->StartDMA(AcqSize * 2);  
 
#ifdef PARALLEL
    printf("AT BARRIER , rank %d\n",rank);
	MPI_Barrier(MPI_COMM_GROUP1);

  if (rank==0) 
  {
    	printf("Sending First Trigger, rank %d\n",rank);
// ULP june 11, 2013
	//DoTrigger(brd);
  }
#else
  printf("Sending First Trigger\n");
  // Trigger the First Acquisition
  DoTrigger(brd);
#endif

/*  // Getting time stamp corresponds to the starting of the Buffer
  gettimeofday( &timestamp, &noreq );	  
  blocktime = ((UNPACK_LEN/4)/(1000*1000)/100)*3	
  blocktime_sec = blocktime
*/
  if (brd->GetPLLStatus() != true) {
    printf("rank %d ERROR: PLL on A/D Board is Not Locked!\n",rank);
  }

  } // rank if

  bool done = false;
  printf(" NODE  BUFFER-NO  BUFFER-STATUS  BUFFER-TRANSFER-TIME  PACKING-TIME  DISK-WRITING-TIME \n"); 

// DATA ACQUISITION LOOP IS STARTING
//  while(Totalrun != lp_cnt) {
  while(1 ) {
#ifdef PROFILE
	  start_tsc = rdtsc ();  
#endif
	  	  
	  if(rank<size/2){
	  int s0 = brd->GetBufferStatus();
 	  //printf(" Board Buffer status at start of loop is %02X\n", s0);
	
          if (s0 == 0)
          {      
		#ifdef DEBUG
			printf(" rank %d waiting for fresh data \n", rank);
		#endif
		brd->WaitForDMAInterrupt();
          }
	#ifdef DEBUG
	  else printf(" rank %d buffer is already available \n", rank);
        #endif
#ifdef PROFILE	
	  end_tsc = rdtsc ();
	  u_int64_t diff_trans = end_tsc - start_tsc;
	  trans_count = u_int64_t(diff_trans/2992.629);
#endif
#ifdef DEBUG
	  printf("counter values of rank %d at start-end of transfering data and transfer time in us %lld %lld %lld \n", rank, start_tsc, end_tsc, trans_count);
#endif
  	  
	  // Give us the status of the Board Buffer which is filled
	  int s1 = brd->GetBufferStatus();
	  //printf(" Filled Board Buffer status of rank %d is %02X\n", rank, s1);
#ifdef PROFILE
	  long long transfer = brd->XFERCNT->Get();
          printf(" %d  %d  %02X  %lld  %lld %lld %lld \n", rank+1, lp_cnt, s1, trans_count, pack_count, write_count, transfer);
#endif
	  //printf("BUFFER STATUS				 	%d  %d  %02X \n", rank+1, lp_cnt, s1); 
	  if(s1 == 3)
	  {
          	local_t = localtime(&timestamp.tv_sec);
        	time_ms = timestamp.tv_usec/1e6;
        	strftime (time_string, sizeof (time_string), "%Y %m %d %H %M %S", local_t);
	
		fprintf(stderr,"BUFFER LOSS for NODE    %d     for buffer nos    %d  	with TIMESTAMP 	%s %lf \n", rank+1, lp_cnt,time_string, time_ms);
          }
	  fprintf(stderr,"%d \n",lp_cnt);

	  if (brd->DMAStatus() == -2) {
                printf("#Node %d DMA Timed out Waiting for Interrupt \n", rank+1);
          } 
	  if (brd->FifoOverFlow()) 
          {
       		printf("#The Board Fifo Overflowed of Node %d, Corrupted Data.\n", rank+1);
      		brd->OFOVERFLOW->Set(1);
		done = true;
      		break;
    	  }
          // Clear that Buffer status which is now filled as it will acquire data after the current one
	  WhichBuffer = !WhichBuffer; 
          buff_cnt=WhichBuffer*BufferSize/2;
	  brd->ClearBufferStatus(WhichBuffer);
	  
//	  Getting time stamp corresponds to the starting of the Buffer
	  gettimeofday( &timestamp, &noreq );	  

	  unsigned char *d = brd->Data;

#ifdef DEBUG 
          fprintf(stderr, "DATA for rank %d ARE %d %d %d ... %d \n",rank, ftable[d[0+WhichBuffer*BufferSize/2]],ftable[d[1+WhichBuffer*BufferSize/2]],ftable[d[2+WhichBuffer*BufferSize/2]],ftable[d[3+WhichBuffer*BufferSize/2]]);   
#endif
	  if (mode == 0)
	  { if(lp_cnt==2){
	      FILE *svol = fopen("/home/jroy/gain/rawdump_TGC.dat","w");
	      if (svol == NULL) perror("fopen");
	      if(sam_bit==4){
                    for(int j=0; j<20000; j=j+2)
                    {       byte1[j] = ftable[((d[j] << 4) & 0xf0)];
                            byte1[j+1] = ftable[(d[j] & 0xf0)];
                            byte2[j] = ftable[((d[j+1] << 4) & 0xf0)];
                            byte2[j+1] = ftable[(d[j+1] & 0xf0)];
                            fprintf(svol, "%d %d %d %d\n", byte1[j],byte1[j+1],byte2[j],byte2[j+1]);
                    }
            }
            else{
	    for(int j=0; j<160000; j=j+4)
	    fprintf(svol, "%4hhd %4hhd %4hhd %4hhd\n", ftable[d[j]], ftable[d[j+1]], ftable[d[j+2]], ftable[d[j+3]]); 
	    }
	    fclose(svol);
	    brd->EndDMA();
	    //exit(-1);
	 }}
  	  
          // Process 16MB of data from user Buffer(filled at rate of 158MB/s) and the timing loop for it
#ifdef PROFILE
          write_tsc = rdtsc ();
#endif

          if(sam_bit==8 || sam_bit == 4) {
		ProcessData(d+WhichBuffer*BufferSize/2, dump_bit, nint_corr, buff_num, fstop, step,phase,PROC_DATA_FORMAT);
	  }	

#ifdef PROFILE
          endwrite_tsc = rdtsc ();
          u_int64_t diff_write = endwrite_tsc - write_tsc;
          write_count = u_int64_t(diff_write/2992.629);
          u_int64_t diff_pack = endpack_tsc - pack_tsc;
          pack_count = u_int64_t(diff_pack/2992.629);
#endif
	}
	else corr_driver((signed char*)NULL,dump_bit,nint_corr,timestamp,ch_freq,buff_num,fstop,step,phase);

#ifdef DEBUG
          printf("counter values of rank %d at start-end of data writing and writing time in us %lld %lld %lld \n", rank, write_tsc,
		endwrite_tsc, write_count);      
         printf("observation is going on for %d seconds \n", lp_cnt);
#endif

  lp_cnt++; // Incrementing the loop count value till it reaches to the Totalrun
  if(rank<size/2){	 
  if (brd->GetPLLStatus() != true) {
    printf("rank %d ERROR: PLL on A/D Board is Not Locked!\n",rank);
  }
  }//rank if	
  } // while
// DATA ACQUISITION STOP HERE

   if(rank<size/2){
  // End the DMA
  brd->EndDMA();

  // End the Acquisition
  brd->EndAcq();
  } //rank if
}
int CalcBNCTriggerRange(int mV) {
  int retval = 0;
  //              if (hardware.FPGA.BoardRevision == 0)
  //                      retval = 2048 - (int)(4096 * (double)(mV / 8070.0));
  //              else
  retval = 2048 + (int)(mV / 3.0);
  if (retval < 0)
    retval = 0;
  if (retval > 4095)
    retval = 4095;
  return retval;
}


//------------------------------------------------------------------------
// main
//------------------------------------------------------------------------
int main(int argc, char *argv[])
{
  try {
    printf("AL4108 Dual Buffer Streaming Demo\n");
    printf("Copyright 2004-2005, Acquisition Logic\n");
   
    int mode = atoi(argv[1]);
    int sam_bit = atoi(argv[2]);
    int sam_clk = atoi(argv[3]);
    int dump_bit = atoi(argv[4]);
    int loops = atoi(argv[5]);
    int nint_corr = 1;
    int buff_num = 1;
    int fstop = 1;
    	


    int r_array[NUM_PROCESSES];
    	
    #ifdef PARALLEL
    		MPI_Init(&argc, &argv);
	        MPI_Comm_size(MPI_COMM_WORLD, &size);
        	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                std::cout << "rank:\t" << rank << "\tof:\t" << size << std::endl;
    #endif
    
     for(int i=0; i<NUM_PROCESSES; i++)
     r_array[i] = i;
     
     MPI_Comm_group(MPI_COMM_WORLD,&group0);
     MPI_Group_incl(group0,NUM_PROCESSES/2,&r_array[0],&group1);
     MPI_Group_incl(group0,NUM_PROCESSES/2,&r_array[NUM_PROCESSES/2],&group2);
     MPI_Comm_create(MPI_COMM_WORLD,group1,&MPI_COMM_GROUP1);
     MPI_Comm_create(MPI_COMM_WORLD,group2,&MPI_COMM_GROUP2);

    //  AD BOARD INITILISATION  START HERE
    if(rank<size/2){
    BoardEnum *brdinfo = new BoardEnum();

    printf("BoardEnum reports %i board(s) in the system\n", brdinfo->NumberOfBoards);
    for(int x=0; x < brdinfo->NumberOfBoards;x++) {
        printf("  Board %i Type is %s\n", x, brdinfo->TypeToString(brdinfo->BoardTypes[x]));
    }
    } 
    // Create a New AL4108 Board Object
    // Pass true to the constructor to upload the RBF file, this only has
    // to be done the first time after a power cycle.  Pass false to skip
    // the RBF file upload.
    AL4108 *Board;
     
    if(rank<size/2){
    int BoardNumber = 0;  // Set This to the Board Number to Use
    Board = new AL4108(false, BoardNumber);
    } //rank if
    
    if(rank>=size/2){
#ifndef RAW_DUMP
    // calculate antenna w-parameter for fringer correction
    FILE *fhd;
    fhd = fopen("/mnt/code/jroy/gsb32/SYS_FILES/antsys.hdr","r");
    get_antenna(fhd, corr.antenna);
    fclose(fhd);
    //printf("antenna parameter is %s\n", corr.antenna[0].name);

    FILE *fp_src=fopen("/mnt/code/jroy/gsb32/SYS_FILES/source.hdr","r");
    while(fgets(str,100,fp_src))
    {       str_ptr=str;
            while(*str_ptr == '*') str_ptr++;
            if (strncmp(str_ptr, "sou_name", 8) == 0) continue;
            if (*str_ptr == '#') continue;
            else {
                     while(*str_ptr++ != ' ');
                     while(isspace(*str_ptr)) str_ptr++;
                     sscanf(str_ptr,"%f", &src_ra);
                     while(*str_ptr++ != ' ');
                     while(isspace(*str_ptr)) str_ptr++;
                     sscanf(str_ptr,"%f", &src_dec);
                     while(*str_ptr++ != ' ');
                     while(isspace(*str_ptr)) str_ptr++;
                     sscanf(str_ptr,"%ld", &obs_freq);
                     while(*str_ptr++ != ' ');
                     while(isspace(*str_ptr)) str_ptr++;
                     sscanf(str_ptr,"%ld", &BW);
                     break;
                 }
    }
    fclose(fp_src);
//#ifdef DEBUG
    fprintf(stderr,"Source RA is %f DEC is %f for rank %d \n",src_ra,src_dec,rank);
    fprintf(stderr,"Observation RF freq is %ld BW is %ld for rank %d \n",obs_freq,BW,rank);
//#endif
    source.ra_app = src_ra;         // ra_app (date)
    source.dec_app = src_dec;       // dec_app (date)
    source.freq[0] = obs_freq;

    fprintf(stderr,"Observation RF freq is %f BW is %d for rank %d \n",source.freq[0],BW,rank);

    fhd = fopen("/mnt/code/jroy/gsb32/SYS_FILES/sampler.hdr","r");
    get_sampler(fhd, &corr);
    fclose(fhd);
#ifdef DEBUG
    printf("antmask value of rank %d is %ld \n", myrank, corr.daspar.antmask);
#endif
    
   FILE *fp_table = fopen("/mnt/code/jroy/gsb32/SYS_FILES/fringe_table.fix.dat", "r");    // Reading from Fringe table
   step = FRNG_STEP*M_PI/180;
   int n_table = 2*M_PI/step;
   for(int i=0;i<n_table;i++)
   {	fgets(phase_str,40,fp_table);
        phase_ptr = phase_str;
        sscanf(phase_ptr,"%hi",&phase[2*i]);
        while(*phase_ptr++ != ' ');
        while(isspace(*phase_ptr)) phase_ptr++;
        sscanf(phase_ptr,"%hi",&phase[2*i+1]);		     
   } 
#endif
   } // rank if
		
//  AD BOARD INITILISATION  START HERE	

    if(rank<size/2){
    printf("rank %d Board is Serial Number %i\n",rank, Board->SerialNumber());

    // Print the FPGA Version
    int ver = Board->FPGAVERSION->Get();
    printf("Version = %02X:%02X\n", (ver & 0xFF00) >> 8, ver & 0x00FF);

    printf("Channel   Master\n");
    printf("----------------\n");
    for(int c=0; c<16; c++)
      printf("  %02i     %7.3f\n", c, Board->ReadTestAD(c));

    Board->ENCDATA->Set(0x0000);
    usleep(100);
    
    // Setup the Number of Channels
    int numchan = 4;
    switch (numchan) {
    case 1 :
      Board->AINRANGE->Set(0);               // 0=1V,1=500mV,2=250mV,3=125mV
      Board->SetUpClockSelectregs(0x200);
      break;
    case 2 :
      Board->AINRANGE->Set(0);               // 0=1V,1=500mV,2=250mV,3=125mV
      Board->BINRANGE->Set(0);               // 0=1V,1=500mV,2=250mV,3=125mV
      Board->SetUpClockSelectregs(0x100);
      break;
    case 4 :
      Board->SetUpClockSelectregs(0);
        //int again=120;
        int again=1;
	Board->SetUpTGC(again,again,again,again);
      break;
    }

    // Set the A Channel Params
    Board->AACCOUPLE->Set(0);              // AC Coupled, 0=on, 1=off
    Board->SetInputOffset(0, 0);           // Setup the Input Offset

    // Set the B Channel Params
    Board->BACCOUPLE->Set(0);              // AC Coupled, 0=on, 1=off
    Board->SetInputOffset(1, 0);           // Setup the Input Offset

    // Set the C Channel Params
    Board->CACCOUPLE->Set(0);              // AC Coupled, 0=on, 1=off
    Board->SetInputOffset(2, 0);           // Setup the Input Offset

    // Set the D Channel Params
    Board->DACCOUPLE->Set(0);              // AC Coupled, 0=on, 1=off
    Board->SetInputOffset(3, 0);           // Setup the Input Offset

    // routine for Gain adjustment read the gain table from gaintab.dat
        FILE *gtab = fopen("/home/gsbuser/gain/gaintab.dat","r");
        if (gtab == NULL) {perror("fopen");exit(-1);}
        int gain[4];
        for(int j=0; j<4; j++){
        fscanf(gtab,"%d", &gain[j]);
		//gain[j]=MAX(255,gain[j]+3);
	}
        Board->SetUpTGC(gain[0],gain[1],gain[2],gain[3]);
    

#ifdef PARALLEL
		if (rank==0) {
			printf("master board, rank: %d\n",rank);
#endif
		  	Board->CMS->Set(3);  // 3 -> 5 MHz clock in
		  	//Board->CMS->Set(1);  // self
                        
			Board->TRIGPOL->Set(0);  // As the GPS is sync to rising edge and dist box invert the signal
			Board->TRIGSRC->Set(5);// PLL sync
			//Board->TRIGSRC->Set(0); // software
                        Board->BNCTERM->Set(1);
                        Board->BNCOUTEN->Set(0);
                        Board->BNCOUT->Set(0);
                        Board->BNCPOL->Set(1);  // require to invert the trigger pol from dist box
                        //Board->SMCOUT->Set(0);
                        Board->SetDAC("BNCTriggerRef", CalcBNCTriggerRange(1250));
			
			/*Board->TRIGPOL->Set(0); // 1=non-inverted
			Board->TRIGSRC->Set(0);  // 0=software
			//Board->TRIGSRC->Set(5);   // 6=PLL sync, rising
			Board->BNCTERM->Set(1);  // 1=on
			Board->BNCOUTEN->Set(1);  // 1=output
			Board->BNCOUT->Set(3);   // 3= totem pole
			Board->BNCSRC->Set(2);	// 2=trigger sync to falling PLL edge 
			//Board->BNCSRC->Set(0);	// Software BNC source 
			Board->BNCPOL->Set(0);  // 1=non-inverted
			//Board->SMCOUT->Set(3);	// 3=totem-pole
			//Board->SMCPOL->Set(0);	// 0=inverted
			//Board->SMCSRC->Set(0);	// 0=software
			Board->SetDAC("BNCTriggerRef", CalcBNCTriggerRange(1250));

			*/			
#ifdef PARALLEL
		}
		else {
			printf("slave board, rank: %d\n",rank);
			Board->CMS->Set(3);  // 3 -> 5 MHz clock in
                        //Board->CMS->Set(1);  // self clock

			Board->TRIGPOL->Set(0);  // As the GPS is sync to rising edge and dist box invert the signal 
			Board->TRIGSRC->Set(5);// PLL sync
			//Board->TRIGSRC->Set(0); // software
			Board->BNCTERM->Set(1);
			Board->BNCOUTEN->Set(0);
			Board->BNCOUT->Set(0);
			Board->BNCPOL->Set(1);  // require to invert the trigger pol from dist box
			//Board->SMCOUT->Set(0);
			Board->SetDAC("BNCTriggerRef", CalcBNCTriggerRange(1250));
		}
#endif
    } //rank if 	
    // Setup and Run
    system("date");
    MPI_Barrier(MPI_COMM_WORLD);
    SetupAcq(Board, mode, sam_bit, sam_clk, dump_bit, loops, nint_corr, buff_num, fstop,step,phase);
    delete Board;
  } catch (exHardware ex) {
    printf(ex.s);
  } catch (...) {
    printf("rank %d: An Exception was Thrown\n",rank);
  }
 
  return(0);
}
