MAKEFLAGS += --no-builtin-rules

.ONESHELL:
.RECIPEPREFIX = >

test.cxx: test.l
> $(LEX) -t $^ > $@

test: test.cxx
> $(CXX) $^ -o $@

clean:
> $(RM) test.cxx test

.PHONY: clean
