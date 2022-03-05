pure real function sq(x)
  implicit none
  real, intent(in) :: x
  sq = x * x
end function sq

!_____________________________________________

program func
  implicit none
  real :: i, o
  real :: sq

  i = 5.0
  o = sq(i)

  print *, 'Square of ', i, ' is ', o

end program func
