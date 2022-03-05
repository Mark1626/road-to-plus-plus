!Prints powers of two till argv[1]
program powers

  implicit none

  integer*8 :: i, n
  character(20) :: n_ch

  if (iargc() < 1) then
    print *, 'Arguments must be greater than 2'
    call exit(1)
  end if

  call getarg(1, n_ch)

  read(n_ch(1:4),'(I5)') n

  do i = 1, n
    print *, i, 2**i
  end do


end program powers