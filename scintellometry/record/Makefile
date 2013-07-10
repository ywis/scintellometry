#-----------------------------------------------------------------------------
#title: Acquisition Logic A/D Board Kernel Driver Checkout Program
#version: Linux 1.0
#date:  March 2004
#designer: Michael Wyrick                                                   
#programmer: Michael Wyrick                                                
#platform: Linux 2.4.x
#language: GCC 3.3.1
#module: BoardAL4108
#-----------------------------------------------------------------------------
#  Purpose: Library for the AL4108
#  Docs:                                  
#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
INCLUDEDIR	= -I/home/jroy/Driver_new/Driver -I/opt/intel/ipp/5.2/ia32/include/ 
CFLAGS		= -O3 -xN -openmp -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE=1 -D_LARGEFILE_SOURCE=1 ${INCLUDEDIR} -DPARALLEL -DSHM_DUMP # -ggdb -c -Wall 
#CFLAGS		= -O3 -xN -openmp -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE=1 -D_LARGEFILE_SOURCE=1 ${INCLUDEDIR} -DPARALLEL 
CFLAGSX          = -O3 -c -Wall ${INCLUDEDIR}
#CC		= /mnt/shared/pen/lam-7.1.3/bin/mpicc #gcc
CC		=   /opt/openmpi-1.2.2/bin/mpicc #gcc
#CC              =   /opt/openmpi/1.2.2-intel-9.1/bin/mpicc #gcc
#CXX             =   /opt/openmpi/1.2.2-intel-9.1/bin/mpiCC #gcc
#CXX   = /mnt/shared/pen/lam-7.1.3/bin/mpiCC
CXX   =   /opt/openmpi-1.2.2/bin/mpiCC
CCC   = icc
TARGET		= mpi_dual

SOURCES	        = dual.cpp reg.cpp Board4108.cpp Dacs.cpp EEPROM_24xx64.cpp I2C_4108.cpp BoardEnum.cpp 
LIBS            = -lipps -lippm -L/opt/intel/ipp/5.2/ia32/sharedlib #-lstdc++
OBJECTS		= ${SOURCES:.cpp=.o} parallel_record.o ipp.o astro_cal.o corrsub.o
HEADERS		= Exceptions.h I2C_4108.h EEPROM_24xx64.h Board4108.h parallel_correlator_timeavgchan.h newcorr.h

.SUFFIXES:
.SUFFIXES:	.cpp .c .o

.cpp.o:
	${CC} ${CFLAGS} -o ${<:.cpp=.o} $<

${TARGET}:	${OBJECTS}
	${CXX} -o ${TARGET} ${CFLAGS} ${OBJECTS} ${LIBS}

dual.o: dual.cpp
	${CXX} ${CFLAGS} -c $<

reg.o: reg.cpp
	${CXX} ${CFLAGS} -c $<

Board4108.o: Board4108.cpp
	${CXX} ${CFLAGS} -c $<

Dacs.o: Dacs.cpp
	${CXX} ${CFLAGS} -c $<

EEPROM_24xx64.o: EEPROM_24xx64.cpp
	${CXX} ${CFLAGS} -c $<

I2C_4108.o: I2C_4108.cpp
	${CXX} ${CFLAGS} -c $<

BoardEnum.o: BoardEnum.cpp
	${CXX} ${CFLAGS} -c $<

ipp.o:  ipp.c
	${CCC} ${CFLAGSX} ${LIBS} -c $<

corrsub.o:  corrsub.c
	${CCC} -c $<

astro_cal.o:  astro_cal.c
	${CCC} -c $<


${OBJECTS}:	${SOURCES} ${HEADERS}

clean:
	rm -f ${OBJECTS} ${TARGET} *~

