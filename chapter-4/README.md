# Chapter 4 - OpenMP

## Chapter 4.1 - Perlin Noise

[README](./4.1/README.md)

Perlin Noise is a [embarringly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) task, which makes it a good candidate to start

## Chapter 4.2 Exploring the depths of OpenMP

Case 1 - Reduction

[README](./4.2/README.md)

## Debugging

- [libgomp DEBUG](https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fDEBUG.html)

## FAQ

- What is the difference between `critical` and `atomic`

Atomic is used for a single assignment, where as critical is used for a section.
Atomic can only be used for `++x;`, `--x;`, `x++;`, `x--;`, `x binop= expr;`, `x = x binop expr` or `x = expr binop x`.

[Stackoverflow reference](https://stackoverflow.com/questions/7798010/what-is-the-difference-between-atomic-and-critical-in-openmp)

## Reading

- [OpenMP Cancel](http://jakascorner.com/blog/2016/08/omp-cancel.html)
