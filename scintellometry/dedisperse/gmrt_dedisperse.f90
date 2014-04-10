Use MKL_DFTI
type(DFTI_DESCRIPTOR), POINTER :: My_Desc1_Handle, My_Desc2_Handle, My_Desc3_Handle, My_Desc4_Handle
Integer :: Status

integer, parameter :: nhead=0*32*1024*1024,nblock=512, nt=32,ntint=1024*32*1024/(nblock*2)/4
integer, parameter :: nblock2=ntint*nt*nblock,nblock3=128,npulsar=4,nrec=512,nrecout=ntint*nt*nblock/nblock3
!nt=45 for 1508, 180 for 0809 and 156 for 0531
integer*1, dimension(nhead) :: raw1
integer*1, dimension(2,nblock) :: raw4
integer*1, dimension(nblock,nrec) :: ibuf
complex, dimension(nblock,nrec) :: cbufall
complex, dimension(nblock3,nrecout) :: cbufallout
integer*1, dimension(nblock3,nrecout) :: ibufout
real, dimension(nblock) :: fscale
real, dimension(nblock3) :: fscaleout
complex, dimension(nblock+1) :: cbufx,cbufy
complex, dimension(nblock2+1) :: cbufx2,cbufy2
complex, dimension(nblock3+1) :: cbufx3
real, dimension(2*nblock) :: rbufx,rbufy,envelope
real, dimension(2*nblock2) :: rbufx2
real, dimension(2*nblock3) :: rbufx3
equivalence(rbufx2,cbufx2)
equivalence(rbufx3,cbufx3)
real, dimension(nblock) :: rspect
real, dimension(nblock2) :: rspect2
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
character*7 psrname(npulsar)
character*1 fchiral(2)
complex ctmp,ctmp2,cc,cphase,cang,cphaseold
common /cblock/ cbufx,ctmp,cbufx2,ctmp2
integer mloc1(1)
character*255 fndir1,fndir2,fndirout
real wsave(4*nblock+4),wsave2(4*nblock2+4),wsave3(4*nblock3+4)

api=4*atan(1.)
fndir1='./'
fndirout='./'
fndir2=fndir1
write(*,*) 'nrecout=',nrecout

cang=(0.90352-0.0032225)
fchiral=(/'L','R'/)
ipsr=1
!psrname=(/'0809+74','1508+55','1957+20','1919+21'/)
psrname=(/'2016+28','1937+21','1957+20','1919+21'/)
!dm0=(/5.75130,19.5990,29.1168*1.001,12.4309/)
!dm0=(/5.75130,19.5990,29.121390086960949536,12.4309/)
dm0=(/14.17600,71.040,29.121390086960949536,12.4309/)



do ichiral=2,2!2
open(10,file=trim(fndir1)//'phasedL'//psrname(ipsr)//'1.dat',status='old',form='binary')

open(30,file=trim(fndirout)//psrname(ipsr)//fchiral(ichiral)//'.4bit',form='binary')

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
!call scfft1d(cbufx,2*nblock,0,wsave)
!call scfft1d(cbufx2,2*nblock2,0,wsave2)
!call scfft1d(cbufx3,2*nblock3,0,wsave3)


envelope=0
rspect=0
rspect2=0
do k=1,2000
   write(*,*)k
   do j=1,nt
      ibin=(j-1)*ntbin/nt+1
      rnorm=0
      corr=0
      rx=0
      varx=0
      do i0=1,ntint,nrec
         read(10,end=100) fscale
         read(10,end=100) ibuf
         call nibble2c(cbufall,fscale,ibuf,nblock,nrec)
         do i=1,nrec
            cbufx=0
            cbufx(:nblock)=cbufall(:,i)
            !      read(10,end=100) cbufx(:nblock)
            it=nblock/4+1
            cbufy=cbufx
            !      call csfft1d(cbufx,2*nblock,1,wsave)
            Status = DftiComputeBackward( My_Desc1_Handle, cbufx )
            if (Status .ne. 0) pause
            rbufall(:,i+i0-1,j)=rbufx
            rbufy=rbufx
         end do
      end do
   end do
   cbufx2(nblock2+1)=0
   rbufx2=rbufall2
   !call scfft1d(cbufx2,2*nblock2,-1,wsave2)
   Status = DftiComputeForward( My_Desc2_Handle, cbufx2 )
   if (Status .ne. 0) pause
   ! dedisperse
   bw=2*16.6666666
   freq0=306+bw/2
   do i=1,nblock2
      dfreq=bw*i/nblock2-bw/2
      ang=dm0(ipsr)*dfreq**2/(2.41e-10*freq0**2*(freq0+dfreq))
      ctmp=exp(2*api*(0.,1.)*ang)
      cbufx2(i)=cbufx2(i)*conjg(ctmp)
   enddo
   cbufx2(1)=0
   Status = DftiComputeBackward( My_Desc2_Handle, cbufx2 )
   !call csfft1d(cbufx2,2*nblock2,1,wsave2)
   rbufall2=rbufx2
   do i=1,ntint*nt*nblock/nblock3
      rbufx3=rbufall0(:,i)
      Status = DftiComputeForward( My_Desc3_Handle, cbufx3 )
      !      call scfft1d(cbufx3,2*nblock3,-1,wsave3)
      
      cbufx3(1)=cmplx(real(cbufx3(1)),real(cbufx3(nblock3+1)))
      cbufallout(:,i)=cbufx3(:nblock3)
      !   write(30) cbufx3(:nblock3)
   end do
   call c2nibble(fscaleout,ibufout,cbufallout,nblock3,nrecout)
   write(30)fscaleout
   write(30) ibufout
enddo
100 continue
nt1=k-1
write(*,*) 'read',nt1,'out of ',nt
enddo
end
