# Chapter 4 - OpenMP

## Chapter 4.1 - Perlin Noise

[README](./4.1/README.md)

Perlin Noise is a [embarringly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) task, which makes it a good candidate to start

## Chapter 4.2 Exploring the depths of OpenMP

Case 1 - Reduction

Case 2 - Case study of cancellation point

> Note: Cancel only cancels current thread

Case 3 - xxtea middle block block collider in OpenMP target constructs

Case 4 - Conditional and Nested OpenMP usage. Nested thread spawning has to be enabled with `omp_set_nested(1)`, control on how many threads to spawn can be done with `OMP_NUM_THREADS=2,8`

Case 5: A curious behaviour in reduction. In reduction if there is a variable changed by a thread, it is not visible to other threads even if barrier is called.


Interstingly `omp master` is several times faster than `omp single` as there is no implicit barrier at the end of the master region, other threads can continue beyond the region without waiting, look at case study the [other chapter](../chapter-acceleration/openacc/sandpile/README.md)

## Debugging

- [libgomp DEBUG](https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fDEBUG.html)

## FAQ

- What is the difference between `critical` and `atomic`

Atomic is used for a single assignment, where as critical is used for a section.
Atomic can only be used for `++x;`, `--x;`, `x++;`, `x--;`, `x binop= expr;`, `x = x binop expr` or `x = expr binop x`.

[Stackoverflow reference](https://stackoverflow.com/questions/7798010/what-is-the-difference-between-atomic-and-critical-in-openmp)

## GPU

- [OpenMP on GPU](https://on-demand.gputechconf.com/gtc/2018/presentation/s8344-openmp-on-gpus-first-experiences-and-best-practices.pdf)

## Reading

- [OpenMP Cancel](http://jakascorner.com/blog/2016/08/omp-cancel.html)
- [OpenMP Guide](https://bisqwit.iki.fi/story/howto/openmp/#Abstract)
- [Using OpenMP for Heterogeneous Systems](https://www.nas.nasa.gov/hecc/assets/pdf/training/OpenMP4.5_3-20-19.pdf)
