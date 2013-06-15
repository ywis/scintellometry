subroutine fold(nhead, nblock, nt, ntint, ngate, ntbin, ntw, &
     dm, p0, file1, file2, samplerate, fbottom, fband, &
     foldspec2, waterfall, verbose)
  implicit none
  integer, intent(in) :: nhead, nblock, nt, ntint, ngate, ntbin, ntw
  real*8, intent(in) :: dm, p0, samplerate, fbottom, fband
  character*(*), intent(in) :: file1, file2
  logical, intent(in) :: verbose
  real, intent(out) :: foldspec2(nblock, ngate,ntbin)
  real, intent(out) :: waterfall(nblock, nt*ntint/ntw)
  integer*1 :: raw1(nhead)
  integer*1 :: raw4(2,nblock)
  complex :: cbufx(nblock), cbufold(nblock)
  real :: foldspec(nblock, ngate)
  integer ::icount(nblock, ngate)
  real*8 :: t0, t, dt, t1, freq
  integer :: i, j, if, ibin, iresid, iw, iw0, iphase, nt1

  open(10, file=file1, status='old', form='unformatted', access='stream')
  open(11, file=file2, status='old', form='unformatted', access='stream')

  if(nhead .gt. 0) then
     print*, 'Skipping ', nhead, 'bytes'
     read(10) raw1  ! if needed, skip some blocks
     read(11) raw1
  endif

  foldspec = 0
  icount = 0
  waterfall = 0
  cbufold = 0
  do j = 1, nt
     ! equivalent time since start
     t1 = (j-1)*2*nblock*(ntint/samplerate)
     if (verbose .and. mod(j,100) .eq. 0)then
        print 90, j, nt, t1
90      format('Doing=',i6,'/',i6,'; time=', g18.12)
     end if
     t0 = t1
     ibin = (j-1)*ntbin/nt+1
     iresid = mod(j, nt/ntbin)
     do i = 1, ntint
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
           freq = fbottom + fband*if/nblock
           dt = 4149.d0 * dm * (1.d0/freq**2 - 1.d0/fbottom**2)
           t = t0 + i*(2*nblock)/samplerate - dt + p0/3
           iphase = int(ngate*t/p0)
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
        foldspec2(:,:,ibin) = foldspec
        foldspec = 0
        icount = 0
     endif
  end do
100 close(10)
  close(11)
  nt1 = j-1
  if(verbose)then
     print 95, nt1, nt
95   format('read ',i6,' out of ',i6)
  end if
  where (waterfall .ne. 0)
     waterfall = waterfall - spread(sum(waterfall,2) / &
          count(waterfall .ne. 0, 2), 2, nt*ntint/ntw)
  end where
  return
end subroutine fold