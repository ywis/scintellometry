subroutine fold(file1, file2, dtype, samplerate, fbottom, fband, nblock, &
     nt, ntint, nhead, ngate, ntbin, ntw, dm, phase_coeff, n_coeff, &
     half_data_bug, paired_samples_bug, do_waterfall, do_foldspec, &
     verbose, progress_interval, foldspec2, waterfall)
  implicit none
  character*(*), intent(in) :: file1, file2, dtype
  real*8, intent(in) :: samplerate, fbottom, fband
  integer, intent(in) :: nblock, nt, ntint, nhead, ngate, ntbin, ntw, n_coeff
  real*8, intent(in) :: dm, phase_coeff(n_coeff)
  logical, intent(in) :: half_data_bug, paired_samples_bug
  logical, intent(in) :: do_waterfall, do_foldspec, verbose
  integer, intent(in) :: progress_interval
  real, intent(out) :: foldspec2(nblock, ngate,ntbin)
  real, intent(out) :: waterfall(nblock, nt*ntint/ntw)
  integer*1, allocatable :: raw4(:,:)
  complex :: cbufx(nblock), cbufold(nblock)
  real :: foldspec(nblock, ngate)
  integer ::icount(nblock, ngate)
  real*8 :: tj, tsample, freq, dt, t, tt, phase
  integer :: i, j, if, ip, ibin, iresid, iw, iw0, iphase, status
  logical :: integer1_data, real4_data

  integer1_data = (index('i1', dtype).ge.1)
  real4_data = (index('f4', dtype).ge.1)

  if(integer1_data) allocate(raw4(2,nblock))

  open(10, file=file1, status='old', action='read', &
       form='unformatted', access='stream')
  if(file2.ne.'')then
     open(11, file=file2, status='old', action='read', &
          form='unformatted', access='stream')
  endif

  if(nhead .gt. 0) then
     print*, 'Skipping ', nhead, 'bytes'
     call fseek(10, nhead, 0, status)  ! if needed, skip some blocks
     if(file2.ne.'')call fseek(11, nhead, 0, status)
  endif

  foldspec = 0
  icount = 0
  waterfall = 0
  cbufold = 0
  do j = 1, nt
     ! equivalent time since start
     tj = (j-1)*2*nblock*(ntint/samplerate)
     if (verbose .and. mod(j, progress_interval) .eq. 0)then
        print 90, j, nt, tj
90      format('Doing=',i6,'/',i6,'; time=', g18.12)
     end if
     ! in phased-array mode, only half the data get written
     if(.not. half_data_bug .or. mod(j-1,16) .lt. 8*1)then
        do i = 1, ntint
           if(integer1_data)then
              if (i .le. ntint/2) then
                 read(10, end=100) raw4
              else
                 read(11, end=100) raw4
              endif
              cbufx = cmplx(raw4(1,:)*1.,raw4(2,:)*1.)
              if(paired_samples_bug)then
                 if(mod(i,2) .eq. 0)then
                    cbufx = cbufx-cbufold
                 endif
                 ! if (mod(i,2) .eq. 0 .and. mod(j,16) .eq. 1 .and. i .eq. 2) &
                 !     write(*,*) sum(cbufx*conjg(cbufold)) / &
                 !                sqrt(sum(abs(cbufx)**2)*sum(abs(cbufold)**2))
                 cbufold = cbufx
              endif
           else if(real4_data)then
              read(10, end=100) cbufx
           endif

           if(do_waterfall)then
              iw = (j-1)*ntint + i-1  ! block counter, starting at 0
              iw0 = iw/ntw+1          ! offset in waterfall, 1...ntw
              if( iw0 .le. nt*ntint/ntw) then
                 waterfall(:,iw0) = waterfall(:,iw0) + abs(cbufx)**2
              endif
           endif
           !$omp parallel do default(none) private(freq,dt,t,iphase) shared(t0,p0,dm,foldspec,cbufx,icount,ibin,i,samplerate)
           if(do_foldspec)then
              tsample = tj + (i-1)*(2*nblock)/samplerate
              frequency: do if = 1,nblock  
                 ! calculate time delay due to dispersion, relative to bottom
                 ! of the band
                 freq = fbottom + fband*(if-0.5d0)/nblock
                 dt = 4149.d0 * dm * (1.d0/freq**2 - 1.d0/fbottom**2)
                 t = tsample - dt
                 tt = 1.d0
                 phase = phase_coeff(1)
                 do ip = 2, n_coeff
                    tt = tt*t
                    phase = phase+phase_coeff(ip)*tt
                 end do
                 iphase = int(modulo(phase, 1.)*ngate)+1
                 if (cbufx(if)**2 .ne. 0)then
                    foldspec(if, iphase) = foldspec(if, iphase) + &
                         abs(cbufx(if))**2
                    icount(if,iphase) = icount(if,iphase) + 1
                 endif
              end do frequency
           endif
        end do
     endif
     ibin = (j-1)*ntbin/nt+1  ! bin in the time series: 1..ntbin
     iresid = mod(j, nt/ntbin) ! 0 if the last sample of a bin
     if (do_foldspec .and. iresid .eq. 0) then
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
  if(verbose)then
     print 95, j-1, nt
95   format('read ',i6,' out of ',i6)
  end if
  if(do_waterfall)then
     where (waterfall .ne. 0)
        waterfall = waterfall - spread(sum(waterfall,2) / &
             count(waterfall .ne. 0, 2), 2, nt*ntint/ntw)
     end where
  endif
  return
end subroutine fold
