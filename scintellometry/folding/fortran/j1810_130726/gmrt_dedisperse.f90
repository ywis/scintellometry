Use MKL_DFTI
type(DFTI_DESCRIPTOR), POINTER :: My_Desc1_Handle, My_Desc2_Handle, My_Desc3_Handle, My_Desc4_Handle
Integer :: Status

integer, parameter :: nhead=0*32*1024*1024,nblock=512, nt=32,ntint=1024*32*1024/(nblock*2)/4
integer, parameter :: nblock2=ntint*nt*nblock,nblock3=128*4,npulsar=4
!nt=45 for 1508, 180 for 0809 and 156 for 0531
integer*1, dimension(nhead) :: raw1
integer*1, dimension(2,nblock) :: raw4
complex, dimension(nblock+1) :: cbufx,cbufy
complex, dimension(nblock2+1) :: cbufx2,cbufy2
complex, dimension(nblock3+1) :: cbufx3
real, dimension(2*nblock) :: rbufx,rbufy,envelope
real, dimension(2*nblock2) :: rbufx2
real, dimension(2*nblock3) :: rbufx3
equivalence(rbufx2,cbufx2)
equivalence(rbufx3,cbufx3)
!real, dimension(nblock) :: rspect
!real, dimension(nblock2) :: rspect2
real, dimension(2*nblock,ntint,nt) :: rbufall
real, dimension(2*nblock3,ntint*nt*nblock/nblock3) :: rbufall0
real, dimension(2*nblock2) :: rbufall2
equivalence(rbufall,rbufall2)
equivalence(rbufall,rbufall0)
real, dimension(2*nblock+2) :: rbuf2
equivalence(rbufx,cbufx)
equivalence(rbuf2,cbufx)
real*8 t0,t,dt,samplerate,t1,ft0,p0,api
real*8 p00(npulsar),dm0(npulsar)
character*80 psrname(npulsar)
character*1 fchiral(2)
complex ctmp,ctmp2,cc,cphase,cang,cphaseold
common /cblock/ cbufx,ctmp,cbufx2,ctmp2
integer mloc1(1)
character*255 fndir1,fndir2,fndirout
real wsave(4*nblock+4),wsave2(4*nblock2+4),wsave3(4*nblock3+4)

api=4*atan(1.)
fndirout='/scratch/pen/Pulsar/'
fndir2=fndir1

cang=(0.90352-0.0032225)
fchiral=(/'L','R'/)
fndir1='/mnt/data-pen1/bahmanya/tape_6/temp1/phased_array/node34/26jul2013/'

ipsr=3
psrname=(/'J1012+5307_s0','b1919+21','j1810+1744','b0809+74_16h00m'/)
! for may 17, 4AM
p00=(/1.55788743954070/1000,226.53301064496944/1000,1.60731438719155/1000*(1-4*3.252e-07),1292.26516193116868/1000 /)
dm0=(/9.0251,12.455,39.7+0.035,5.75130/)


do ichiral=1,2
if (ichiral .eq. 2) fndir1='/mnt/data-pen1/bahmanya/tape_6/temp1/phased_array/node33/26jul2013/'
fndir2=fndir1
open(10,file=trim(fndir1)//trim(psrname(ipsr))//'.raw.Pol-'//fchiral(ichiral)//'1.dat',status='old',form='binary')
open(11,file=trim(fndir1)//trim(psrname(ipsr))//'.raw.Pol-'//fchiral(ichiral)//'2.dat',status='old',form='binary')

   open(30,file=trim(fndirout)//trim(psrname(ipsr))//fchiral(ichiral)//'.float',form='binary')

   Status = DftiCreateDescriptor( My_Desc1_Handle, DFTI_SINGLE, DFTI_REAL, 1, nblock*2 )
   if (Status .ne. 0) write(*,*) 'fft err',Status
   Status = DftiCommitDescriptor( My_Desc1_Handle )
   if (Status .ne. 0) write(*,*) 'fft err',Status
   Status = DftiCreateDescriptor( My_Desc2_Handle, DFTI_SINGLE, DFTI_REAL, 1, nblock2*2 )
   if (Status .ne. 0) write(*,*) 'fft err',Status
   Status = DftiCommitDescriptor( My_Desc2_Handle )
   if (Status .ne. 0) write(*,*) 'fft err',Status
   Status = DftiCreateDescriptor( My_Desc3_Handle, DFTI_SINGLE, DFTI_REAL, 1, nblock3*2 )
   if (Status .ne. 0) write(*,*) 'fft err',Status
   Status = DftiCommitDescriptor( My_Desc3_Handle )
   if (Status .ne. 0) write(*,*) 'fft err',Status
   envelope=0

   do k=1,200000
      write(*,*)k
      do j=1,nt!000!nt
         ibin=(j-1)*ntbin/nt+1
         rnorm=0
         corr=0
         rx=0
         varx=0
         do i=1,ntint
            if (i .le. ntint/2) then
               read(10,end=100) raw4
            else
               read(11,end=100) raw4
            endif
            cbufx(:nblock)=(cmplx(raw4(1,:)*1.,raw4(2,:)*1.))
            cbufx(nblock+1)=aimag(cbufx(1))
            it=nblock/4+1
            cbufy=cbufx
            !      call csfft1d(cbufx,2*nblock,1,wsave)
            Status = DftiComputeBackward( My_Desc1_Handle, cbufx )
            if (Status .ne. 0) pause
            rbufall(:,i,j)=rbufx
            rbufy=rbufx
         end do
      end do
      cbufx2(nblock2+1)=0
      rbufx2=rbufall2
      Status = DftiComputeForward( My_Desc2_Handle, cbufx2 )
      if (Status .ne. 0) pause
      ! dedisperse
      bw=16.6666666
      freq0=156-bw/2
      do i=1,nblock2
         dfreq=-bw*i/nblock2+bw/2
         ang=dm0(ipsr)*dfreq**2/(2.41e-10*freq0**2*(freq0+dfreq))
         ctmp=exp(-2*api*(0.,1.)*ang)
         cbufx2(i)=cbufx2(i)*conjg(ctmp)
      enddo
      cbufx2(1)=0
      Status = DftiComputeBackward( My_Desc2_Handle, cbufx2 )
      rbufall2=rbufx2
      do i=1,ntint*nt*nblock/nblock3
         rbufx3=rbufall0(:,i)
         Status = DftiComputeForward( My_Desc3_Handle, cbufx3 )
         cbufx3(1)=cmplx(real(cbufx3(1)),real(cbufx3(nblock3+1)))
         write(30) cbufx3(:nblock3)
      end do
   enddo
100 continue
   nt1=k-1
   write(*,*) 'read',nt1,'out of ',nt
enddo
end program
