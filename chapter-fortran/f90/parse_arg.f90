!This is a GNU extension
PROGRAM parse_arg
  INTEGER :: i
  CHARACTER(len=32) :: arg

  DO i = 1, iargc()
    CALL getarg(i, arg)
    WRITE (*, *) i, arg
  END DO

END PROGRAM parse_arg