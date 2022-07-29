CFLAGS=-std=c99 -march=native -pedantic -O3 `pkg-config --libs cfitsio`

timgauss: timgauss.c reference.h fits-helper.h common.h
	$(CC) -o $@ $< $(CFLAGS)
