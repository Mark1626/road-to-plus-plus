# Chapter Profiling

This chapter is dedicated to profiling with `perf`, `valgrind`

I will be using the sandpile program as an example

## perf

> Before you start compile you binary with `-fno-omit-frame-pointer`

```
// Record every 100Hz
perf record -F 100 <prog>

// Record every 100Hz, with call graph
perf record -F 100 -g <prog>
```

## valgrind

```
./valgrind --tool=memcheck --leak-check=full
```

## Experiments

### Exp-1 perf - Performance Report

```
perf record -g -F 100 ./sandpile > test.ppm
perf report -g 'graph,0.5,caller'
```

### Exp-2 memcheck - Checking for leaks

```
./valgrind --tool=memcheck --leak-check=full ./exp-2
```

The report points out the memory leak from using the new operator

```
==22== Memcheck, a memory error detector
==22== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==22== Using Valgrind-3.14.0 and LibVEX; rerun with -h for copyright info
==22== Command: ./exp-2
==22==
Testing memory leak==22==
==22== HEAP SUMMARY:
==22==     in use at exit: 40 bytes in 1 blocks
==22==   total heap usage: 3 allocs, 2 frees, 73,768 bytes allocated
==22==
==22== 40 bytes in 1 blocks are definitely lost in loss record 1 of 1
==22==    at 0x483650F: operator new[](unsigned long) (vg_replace_malloc.c:423)
==22==    by 0x109176: main (in /path/to/exp-2)
==22==
==22== LEAK SUMMARY:
==22==    definitely lost: 40 bytes in 1 blocks
==22==    indirectly lost: 0 bytes in 0 blocks
==22==      possibly lost: 0 bytes in 0 blocks
==22==    still reachable: 0 bytes in 0 blocks
==22==         suppressed: 0 bytes in 0 blocks
==22==
==22== For counts of detected and suppressed errors, rerun with: -v
==22== ERROR SUMMARY: 1 errors from 1 contexts (suppressed: 0 from 0)
```

### Exp-3 callgrind - Call graph cache profiler

```
valgrind --xtree-memory=full --xtree-memory-file=xtmemory.kcg <prog>
callgrind_annotate --auto=yes --inclusive=yes --sort=curB:100,curBk:100,totB:100,totBk:100,totFdB:100,totFdBk:100  xtmemory.kcg
```

### Exp-4 helgrind - Thread profiling

Compile `sandpile.cc` with `-fopenmp` enabled, there is a problem with the implementation

```
root@296c3475983f:/home/bench# valgrind --tool=helgrind ./sandpile > out.ppm
==1762== Helgrind, a thread error detector
==1762== Copyright (C) 2007-2017, and GNU GPL'd, by OpenWorks LLP et al.
==1762== Using Valgrind-3.14.0 and LibVEX; rerun with -h for copyright info
==1762== Command: ./sandpile
==1762==
WITH_Auto
==1762== ---Thread-Announcement------------------------------------------
==1762==
==1762== Thread #1 is the program's root thread
==1762==
==1762== ---Thread-Announcement------------------------------------------
==1762==
==1762== Thread #4 was created
==1762==    at 0x4CBC4BE: clone (clone.S:71)
==1762==    by 0x4BA6DDE: create_thread (createthread.c:101)
==1762==    by 0x4BA880D: pthread_create@@GLIBC_2.2.5 (pthread_create.c:826)
==1762==    by 0x483C6B7: pthread_create_WRK (hg_intercepts.c:427)
==1762==    by 0x4B6BD61: ??? (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
==1762==    by 0x4B62E09: GOMP_parallel (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
==1762==    by 0x1097CB: Fractal::Sandpile::computeIdentity() (sandpile.cc:94)
==1762==    by 0x10912E: main (sandpile.cc:121)
==1762==
==1762== ----------------------------------------------------------------
==1762==
==1762== Possible data race during write of size 4 at 0x4DBCA70 by thread #1
==1762== Locks held: none
==1762==    at 0x4B6DD42: ??? (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
==1762==    by 0x4B6C194: ??? (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
==1762==    by 0x4B62E09: GOMP_parallel (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
==1762==    by 0x1097CB: Fractal::Sandpile::computeIdentity() (sandpile.cc:94)
==1762==    by 0x10912E: main (sandpile.cc:121)
==1762==
==1762== This conflicts with a previous read of size 4 by thread #4
==1762== Locks held: none
==1762==    at 0x4B6DD9B: ??? (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
==1762==    by 0x4B6B76A: ??? (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
==1762==    by 0x483C8B6: mythread_wrapper (hg_intercepts.c:389)
==1762==    by 0x4BA7FA2: start_thread (pthread_create.c:486)
==1762==    by 0x4CBC4CE: clone (clone.S:95)
==1762==  Address 0x4dbca70 is 128 bytes inside a block of size 192 alloc'd
==1762==    at 0x48367CF: malloc (vg_replace_malloc.c:299)
==1762==    by 0x4B5EB48: ??? (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
==1762==    by 0x4B6B985: ??? (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
==1762==    by 0x4B62DF5: GOMP_parallel (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
==1762==    by 0x1097CB: Fractal::Sandpile::computeIdentity() (sandpile.cc:94)
==1762==    by 0x10912E: main (sandpile.cc:121)
==1762==  Block was alloc'd by thread #1
==1762==
==1762== ----------------------------------------------------------------
```

### Exp-5 perf scripts - Perf Scripts

```sh
perf record
# Walk through perf file and output contents of each record
perf script > out.perf
```

```
        sandpile  1751 13289.335292:   10101010 cpu-clock:uhH:      559a3f21248b Fractal::Sandpile::stabilize+0x24b (/home/bench/sandpile)
```

comm - sandpile
tid - 1751
event - cpu-clock:uhpppH
ip - 559a3f21248b
sym - Fractal::Sandpile::stabilize
symoff - +0x24b
time - 13289.335292
dso - (/home/bench/sandpile)

If perf was recorded with `-g`, the output contains trace like this

```sh
perf record -g
perf script > out.perf

sandpile   131 11005.728397:   10101010 cpu-clock:uhH: 
	    560cdca6739a Fractal::Sandpile::stabilize+0x15a (/home/bench/sandpile)
	    560cdca67984 Fractal::Sandpile::computeIdentity+0x84 (/home/bench/sandpile)
	    560cdca670ef main+0x1f (/home/bench/sandpile)
	    7f36527fd09b __libc_start_main+0xeb (/lib/x86_64-linux-gnu/libc-2.28.so)
	41fd89415541f689 [unknown] ([unknown])
```

Extracting a custom set of events

```
perf script -F  comm,tid,event,ip,sym,srcline,time > out.perf
```

#### Experiment 1 - Identifying a hotspot in the code

Writing a custom script to consume events. The script `srcline-occurance.pl`

```
perf script -F+srcline -F-period -F-time -F-dso -F+sym -F-symoff | srcline-occurance.pl
```

#### Experiment 2 - Spitting perf.data of a MPI run into separate files for individual processes

```
perf script -F+pid | split-process.pl
```

## Exp-6 GCC Profile Guided Optimizations -fprofile-generate

> **Note :** This is really magical

```bash
g++ -o sandpile sandpile.cc -O3 -fprofile-generate

# Do a run and generate a profile sandpile.gcda
./sandpile > out.ppm

# Recompile with profile information
g++ -o sandpile sandpile.cc -O3 -fprofile-use=sandpile.gcda
```

Before profiling
```
./sandpile > out.ppm  4.45s user 0.00s system 90% cpu 4.919 total
./sandpile > out.ppm  4.50s user 0.00s system 99% cpu 4.512 total
./sandpile > out.ppm  4.51s user 0.01s system 99% cpu 4.526 total
```

After profiling

```
./sandpile > out.ppm  0.68s user 0.00s system 59% cpu 1.156 total
./sandpile > out.ppm  0.74s user 0.00s system 99% cpu 0.750 total
./sandpile > out.ppm  0.73s user 0.00s system 99% cpu 0.739 total
```

## Reference

- [Valgrind Docs](https://www.valgrind.org/docs/manual/manual-core.html)
- [Perf Tutorial](https://perf.wiki.kernel.org/index.php/Tutorial)
- [Perf Examples](https://www.brendangregg.com/perf.html)
- [CPP Conf, Tuning C++: Benchmarks, and CPUs, and Compilers! Oh My!](https://www.youtube.com/watch?v=nXaxk27zwlk)
- [Slides on perf](https://indico.cern.ch/event/141309/contributions/1369454/attachments/126021/178987/RobertoVitillo_FutureTech_EDI.pdf)
- [Perf Usages](https://opensource.com/article/18/7/fun-perf-and-python)

