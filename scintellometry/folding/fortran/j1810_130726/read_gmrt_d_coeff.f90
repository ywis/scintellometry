integer, parameter :: nhead=0*32*1024*1024,nblock=512, nt=1024*4*4,ntint=1024*32*1024/(nblock*2)/4,ngate=32*1,ntbin=4*4,ntw=10000
!nt=45 for 1508, 180 for 0809 and 156 for 0531
integer, parameter :: npulsar=4
integer*1, dimension(nhead) :: raw1
integer*1, dimension(2,nblock) :: raw4
complex, dimension(nblock) :: cbufx,cbufy,cbufold
real, dimension(nblock,ngate) :: foldspec
real, dimension(nblock,ngate) :: foldspec1
real, dimension(nblock,ngate,ntbin) :: foldspec2
real, dimension(ngate,ntbin) :: foldspec3
integer,dimension(nblock,ngate)::icount
real, dimension(nblock,nt*ntint/ntw) :: waterfall
real, dimension(nblock,ntbin) :: dynspect,dynspect2,dall
real*8 t0,t,dt,samplerate,t1,ft0,p0,phase,tt
character*80 psrname(npulsar)
real*8 p00(npulsar),dm0(npulsar)
integer igate(4,npulsar)
character*255 fndir1,fndir2
integer*8 iphase
real*8 coeff(12)


fndir1='/mnt/scratch-lustre/mhvk/25_054_20jan14/temp2/pen/25_054_22jan14/'
fndir2=fndir1

ipsr=3
igate(:,2)=(/59,60,61,62/)
igate(:,4)=(/224,226,227,231/)
psrname=(/'J1012+5307_s0','1929+10','j1810+1744','b1919+21'/)

p00=(/1./190.2630146730971,226.53301064496944/1000,1.60731438719155/1000*(1-4*3.252e-07),1.3373021601895 /)

dm0=(/71.04000,3.17600,0.035,5.75130*0/)
open(10,file='/scratch/pen/Pulsar/'//trim(psrname(ipsr))//'R.float',access='stream',status='old')

coeff=(/ -4567450.6719792401d0*0, &
601.3839031302814d0, &
-6.8249573542414501d-06, &
2.1148266254037458d-10, &
1.3568188636518591d-13, &
-3.3535869511481088d-18, &
-6.3699070639028823d-22, &
-9.0212688640369989d-26, &
1.8087164238998356d-29, &
-1.0426671120232157d-33, &
2.6087773419729568d-38, &
-2.4425653108190857d-43 /)

!-3391736.6851270036d0*0, &
!601.40016858227182d0,&
!-6.2885874056292051d-06,&
!-4.7859197051844546d-10,&
!1.2590567804348272d-13,&
!5.9281325525718805d-18,&
!-1.0421894592510491d-21,&
!-3.7092731812364445d-26,&
!5.92649093560978d-30,&
!-3.2470289374333272d-35,&
!-1.0350489892714492d-38,&
!2.5471395787138744d-43 /)
!
if (ipsr .eq. 4) then
   coeff=0
   coeff(2)=1./p00(ipsr)
endif

p0=p00(ipsr)
dm=dm0(ipsr)
foldspec=0
icount=0
samplerate=33333333.333333333 !MSPS
waterfall=0
cbufold=0
do j=1,nt
   t1=(j-1)*2*nblock*(ntint/samplerate)
   if (mod(j,100) .eq.0) write(*,*) t1
   t0=t1
   ibin=(j-1)*ntbin/nt+1
   iresid=mod(j,nt/ntbin)
   do i=1,ntint
      iw=(j-1)*(ntint)+i-1
      iw0=iw/ntw+1
      read(10,end=100) cbufx
!      if (mod(i,2) .eq. 0) cbufx=cbufx-cbufold
!      if (mod(i,2) .eq. 0 .and. mod(j,16) .eq. 1 .and. i .eq. 2) write(*,*)sum(cbufx*conjg(cbufold))/sqrt(sum(abs(cbufx)**2)*sum(abs(cbufold)**2))
      cbufold=cbufx
!      write(*,*) iw0,nt*ntint/ntw
      if( iw0>nt*ntint/ntw) cycle
      waterfall(:,iw0)=waterfall(:,iw0)+abs(cbufx)**2
      t=t0+i*(2*nblock)/samplerate

      tt=1.d0
      phase=0.d0
      do ic = 1,12
         phase=phase+coeff(ic)*tt
         tt=tt*t
      enddo
      iphase=ngate*phase
!      iphase=ngate*t*coeff(2)
      iphase=mod(iphase+40000*ngate,ngate)+1
      !write(*,*) iphase
      if (iphase<1) cycle
!$omp parallel do default(shared) private(freq,dt,iphase1)
      do if=1,nblock
         freq=156-16.6666666666*if/nblock
         dt=4149*dm/freq**2-4149*dm/139**2
         iphase1=ngate
         iphase1=iphase1*dt/p0
	 iphase1=mod(iphase1+iphase+10000*ngate,ngate)+1
         if (iphase1 .eq. 0) pause
         foldspec(if,iphase1)=foldspec(if,iphase1)+abs(cbufx(if))**2
         if (cbufx(if)**2 .ne. 0)icount(if,iphase1)=icount(if,iphase1)+1
      end do      
   end do
foldspec(1,:)=0
if (iresid .eq. 0) then
where(icount>0) 
foldspec=foldspec/icount
elsewhere
foldspec=0
endwhere
where(foldspec .ne. 0) foldspec=foldspec-spread(sum(foldspec,2)/count(foldspec .ne. 0,2),2,ngate)
foldspec1=foldspec1+foldspec
foldspec2(:,:,ibin)=foldspec
!dynspect(:,ibin)=sum(foldspec(:,igate(1,ipsr):igate(2,ipsr)),2)
!dynspect2(:,ibin)=sum(foldspec(:,igate(3,ipsr):igate(4,ipsr)),2)
foldspec=0
icount=0
endif
end do
100 continue
nt1=j-1
write(*,*) 'read',nt1,'out of ',nt
where (waterfall .ne. 0) waterfall=waterfall-spread(sum(waterfall,2)/count(waterfall .ne. 0,2),2,nt*ntint/ntw)
!call pmap('waterfall.pgm',waterfall,nblock,nt*ntint/ntw,2)
open(10,file='flux.dat')
do i=1,ngate
write(10,*) i,sum(foldspec1(:,i),1)
enddo
call pmap('folded'//trim(psrname(ipsr))//'.pgm',foldspec1,nblock,ngate,2)
foldspec3=sum(foldspec2,1)
call pmap('foldedbin'//trim(psrname(ipsr))//'.pgm',foldspec2,nblock,ngate*ntbin,1)
call pmap('folded3'//trim(psrname(ipsr))//'.pgm',foldspec3,ngate,ntbin,1)
open(10,file='dynspect'//trim(psrname(ipsr))//'.bin',form='binary')
write(10) dynspect
write(10) dynspect2
dall=dynspect+dynspect2
dall=dall/spread(sum(dall,1)/nblock,1,nblock)
dynspect=dynspect/spread(sum(dynspect,1)/nblock,1,nblock)
dynspect2=dynspect2/spread(sum(dynspect2,1)/nblock,1,nblock)
dall(1,:)=0
call pmap('dynspect'//trim(psrname(ipsr))//'.pgm',dall,nblock,ntbin,1)
dynspect=dynspect-dynspect2
dynspect(1,:)=0
call pmap('dynspectdiff'//trim(psrname(ipsr))//'.pgm',dynspect,nblock,ntbin,1)
 
contains
  subroutine pmap(fn,rmap1,nx,ny,iscale)
  real rmap(nx,ny),rmap1(nx,ny)
  integer*2, dimension(nx,ny) :: imap
  integer*1, dimension(nx,ny) :: imap1
  character(len=*):: fn
  integer npix,mypos

  npix=min(ny/2-1,nx/2-1,300)
  
  
  rmap=rmap1
!  write(*,*) 'rms=',sqrt(sum(rmap(nx/2-npix:nx/2+npix,ny/2-npix:ny/2+npix)**2)/npix**2/4)
  iscale1=iscale
  do while (iscale1 > 1)      
     rmap=sign((sqrt(abs(rmap))),rmap)
     iscale1=iscale1-1
  end do
  rmax=maxval(rmap)
  rmin=minval(rmap)
  write(*,*) trim(fn),rmax,rmin
  imap=255*(rmap-rmin)/(rmax-rmin)
  imap1=127*(rmap-rmin)/(rmax-rmin)
  open(10,file=fn)
  write(10,'(2hP5)')
  write(10,*)nx,ny
  write(10,*) 255
!  write(10,*) 127
!  INQUIRE(UNIT=10, POS=mypos)
  close(10)
  open(10,file=fn, form='binary',position='append')
!  write(10,pos=mypos) int(imap,1)
  write(10) int(imap,1)
  close(10)
end subroutine pmap

end
