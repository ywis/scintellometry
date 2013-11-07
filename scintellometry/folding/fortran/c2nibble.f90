! written July 14, 2013 by Ue-Li Pen
! optimally compress complex to 4-bit
! theoretical increase in noise is 1/16^2 ~ 0.005

!call testall
!end

subroutine c2nibble(fscale,nibbleout,carray,nfreq,nrec)
implicit none
integer nfreq,nrec
complex carray(nfreq,nrec)
real fscale(nfreq),pi
integer*1, dimension(nfreq,nrec) :: nibbleout,iphase,idf

pi=4*atan(1.)
!$omp parallel default(shared)
!$omp workshare
fscale=sum(abs(carray)**2,2)/nrec/2
iphase=nint(16*(atan2(aimag(carray),real(carray))/(2*pi))-0.5)
! for normal distributed array, PDF(x)=exp(-x^2/(2s))/\sqrt{2\pi s}
! where fscale=2 s^2.
! pdf(x,y)=exp(-x^2/2-y^2/2)/(2\pi)
! in polar coordinates pdf(r,theta)=r exp(-r^2/2)
! CDF(R)=[-exp(-r^2/2)]_0^R=1-exp(-R^2/2)
! and z=CDF(R) means z is uniform distributed on [0,1)
! so inverse of  means R=2\sqrt{-ln(1-z)}
!idf=min(15,nint(16*(1-exp(-abs(carray)**2/spread(fscale,nrec,2)/2))))
idf=nint(16*(1-exp(-abs(carray)**2/spread(fscale,2,nrec)/2))-0.5)
idf=max(0,min(15,idf))
!$omp end workshare
!$omp end parallel
call pack4bit(nibbleout,idf,iphase,nfreq*nrec)
end subroutine c2nibble

subroutine nibble2c(carray,fscale,nibbleout,nfreq,nrec)
implicit none
integer nfreq,nrec
complex carray(nfreq,nrec)
real fscale(nfreq),pi
integer*1, dimension(nfreq,nrec) :: nibbleout,iphase,idf

pi=4*atan(1.)
call unpack4bit(nibbleout,idf,iphase,nfreq*nrec)
!write(*,*) maxval(idf)
!write(*,'(16z3)') idf
!$omp parallel default(shared)
!$omp workshare
carray=cmplx(cos(((iphase+0.5)/16.)*pi*2),sin(((iphase+0.5)/16.)*pi*2))*sqrt(-2*log(1.-((idf+0.5)/16.))*spread(fscale,2,nrec))
!$omp end workshare
!$omp end parallel
end subroutine nibble2c

subroutine pack4bit(nibble,lsb,msb,n)
implicit none
integer n
integer*1, dimension(n) :: nibble,lsb,msb

!$omp parallel default(shared)
!$omp workshare
nibble=int(iand(lsb,15)+msb*16,1)
!$omp end workshare
!$omp end parallel
end subroutine pack4bit

subroutine unpack4bit(nibble,lsb,msb,n)
implicit none
integer n
integer*1, dimension(n) :: nibble,lsb,msb
integer, dimension(n) :: itmp

!$omp parallel default(shared)
!$omp workshare
lsb=iand(nibble,15) ! lsb is unsigned
msb=iand(nibble,240)
msb=msb/16  ! msb is signed
!$omp end workshare
!$omp end parallel
end subroutine unpack4bit


subroutine testall
  implicit none
  integer, parameter :: n=4,nrec=16
  real, dimension(2,n,nrec) :: test,testr
  real, dimension(n) :: fscale
  integer*1 nibble(n,nrec),lsb(2),msb(2),n4(2)
  integer i
  real pi

  pi=4*atan(1.)
  call random_seed()
  call random_number(testr)
  test(1,:,:)=sqrt(-2.*log(testr(1,:,:)))*cos(2*pi*testr(2,:,:))
  test(2,:,:)=sqrt(-2.*log(testr(1,:,:)))*sin(2*pi*testr(2,:,:))

  lsb=(/7,-3/)
  msb=(/1,-1/)
  i=257
!  write(*,*) int(i,1)
  call pack4bit(n4,lsb,msb,2)
  write(*,'(z2)') lsb,msb,n4
  lsb=0
  msb=0
  call unpack4bit(n4,lsb,msb,2)
  write(*,'(z2)') lsb,msb,n4

  write(*,'(8f8.3)') test
  call c2nibble(fscale,nibble,test,n,nrec)
  write(*,'(16z3.2)') nibble
  call nibble2c(testr,fscale,nibble,n,nrec)
  write(*,'(8f8.3)') testr
  write(*,'(4f16.6)') sqrt(fscale)
end subroutine testall
