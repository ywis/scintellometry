subroutine fold(nhead, nblock, nt, ntint, ngate, ntbin, ntw, &
     psr, dm, p0, file1, file2, igate, samplerate, fbottom, fstep)
  implicit none
  integer, intent(in) :: nhead, nblock, nt, ntint, ngate, ntbin, ntw, igate(4)
  real*8, intent(in) :: dm, p0, samplerate, fbottom, fstep
  character*(*), intent(in) :: psr, file1, file2
  integer :: nt, ntint, ntbin
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
  real*8 t0,t,dt,t1,ft0
  integer igate(4,npulsar)
  character*255 fndir1,fndir2

  open(10, file=file1, status='old', form='binary')
  open(11, file=file2, status='old', form='binary')

  allocate(raw1(nhead))
  read(10) raw1  ! if needed, skip some blocks
  read(11) raw1

  foldspec=0
  icount=0
  waterfall=0
  cbufold=0
  do j = 1,nt
     ! equivalent time since start
     t1 = (j-1)*2*nblock*(ntint/samplerate)
     if (mod(j,100) .eq. 0) write(*,*) t1
     t0 = t1
     ibin = (j-1)*ntbin/nt+1
     iresid = mod(j, nt/ntbin)
     do i = 1,ntint
        raw4 = 0
        if (mod(j,16) .lt. 8*1) then
           if (i .le. ntint/2) then
              read(10, end=100) raw4
           else
              read(11, end=100) raw4
           endif
        else
           cycle
        end if
        iw = (j-1)*(ntint)+i-1
        iw0 = iw/ntw+1
        cbufx = cmplx(raw4(1,:)*1.,raw4(2,:)*1.)
        if (mod(i,2) .eq. 0) cbufx = cbufx-cbufold
        ! if (mod(i,2) .eq. 0 .and. mod(j,16) .eq. 1 .and. i .eq. 2) &
        !     write(*,*) sum(cbufx*conjg(cbufold)) / &
        !                sqrt(sum(abs(cbufx)**2)*sum(abs(cbufold)**2))
        cbufold = cbufx
        !  write(*,*) iw0,nt*ntint/ntw
        if( iw0 .gt. nt*ntint/ntw) cycle
        waterfall(:,iw0) = waterfall(:,iw0) + abs(cbufx)**2
        !$omp parallel do default(none) private(freq,dt,t,iphase) shared(t0,p0,dm,foldspec,cbufx,icount,ibin,i,samplerate)
        do if = 1,nblock
           freq = fbottom + fstep*(if-1)
           dt = 4149 * dm * (1./freq**2 - 1./fbottom**2)
           t = t0 + i*(2*nblock)/samplerate - dt + p0/3
           iphase = (ngate*t/p0)
           iphase = mod(iphase+4000000*ngate, ngate)+1
           foldspec(if, iphase) = foldspec(if, iphase) + abs(cbufx(if))**2
           if (cbufx(if)**2 .ne. 0) icount(if,iphase) = icount(if,iphase)+1
        end do
     end do
     if (iresid .eq. 0) then
        where(icount .gt. 0) 
           foldspec = foldspec/icount
        elsewhere
           foldspec = 0
        endwhere
        where(foldspec .ne. 0)
           foldspec = foldspec-spread(sum(foldspec,2) / &
                count(foldspec .ne. 0, 2), 2, ngate)
        endwhere
        foldspec1 = foldspec1+foldspec
        foldspec2(:,:,ibin) = foldspec
        dynspect(:,ibin) = sum(foldspec(:,igate(1):igate(2)), 2)
        dynspect2(:,ibin) = sum(foldspec(:,igate(3):igate(4)), 2)
        foldspec = 0
        icount = 0
     endif
  end do
100 close(10)
  close(11)
  nt1 = j-1
  write(*,*) 'read',nt1,'out of ',nt
  where (waterfall .ne. 0)
     waterfall = waterfall - spread(sum(waterfall,2) / &
          count(waterfall .ne. 0, 2), 2, nt*ntint/ntw)
  end where
  !call pmap('waterfall.pgm',waterfall,nblock,nt*ntint/ntw,2)
  open(10,file='flux.dat')
  do i = 1, ngate
     write(10,*) i, sum(foldspec1(:,i),1)
  enddo
  close(10)
  call pmap('folded'//psr//'.pgm',foldspec1,nblock,ngate,1)
  foldspec3 = sum(foldspec2,1)
  call pmap('foldedbin'//psrname(ipsr)//'.pgm',foldspec2,nblock,ngate*ntbin,2)
  call pmap('folded3'//psrname(ipsr)//'.pgm',foldspec3,ngate,ntbin,1)
  open(10,file='dynspect'//psrname(ipsr)//'.bin',form='binary')
  write(10) dynspect
  write(10) dynspect2
  dall = dynspect+dynspect2
  dall = dall/spread(sum(dall,1)/nblock,1,nblock)
  dynspect = dynspect/spread(sum(dynspect,1)/nblock,1,nblock)
  dynspect2 = dynspect2/spread(sum(dynspect2,1)/nblock,1,nblock)
  dall(1,:) = 0
  call pmap('dynspect'//psrname(ipsr)//'.pgm',dall,nblock,ntbin,1)
  dynspect = dynspect-dynspect2
  dynspect(1,:) = 0
  call pmap('dynspectdiff'//psrname(ipsr)//'.pgm',dynspect,nblock,ntbin,1)
  
contains
  subroutine pmap(fn,rmap1,nx,ny,iscale)
    implicit none
    integer, intent(in) :: nx, ny
    real, intent(in) :: rmap(nx,ny),rmap1(nx,ny)
    character(len=*), intent(in) :: fn
    integer*2, dimension(nx,ny) :: imap
    integer*1, dimension(nx,ny) :: imap1
    integer npix,mypos

    npix = min(ny/2-1,nx/2-1,300)
  
    rmap = rmap1
    !  write(*,*) 'rms=',sqrt(sum(rmap(nx/2-npix:nx/2+npix, &
    !                         ny/2-npix:ny/2+npix)**2)/npix**2/4)
    iscale1 = iscale
    do while (iscale1 > 1)      
       rmap = sign((sqrt(abs(rmap))),rmap)
       iscale1 = iscale1-1
    end do
    rmax = maxval(rmap)
    rmin = minval(rmap)
    write(*,*) trim(fn), rmax, rmin
    imap = 255*(rmap-rmin)/(rmax-rmin)
    imap1 = 127*(rmap-rmin)/(rmax-rmin)
    open(10, file=fn)
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
end subroutine fold
