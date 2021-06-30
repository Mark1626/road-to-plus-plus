MAKEFLAGS += --no-builtin-rules

.ONESHELL:
.RECIPEPREFIX = >

lang-parser.cc: lang.y
> bison -d -o $@ $^

lang-tokens.cc: lang.l lang-parser.cc
> lex -o lang-tokens.cc lang.l

lang: lang-tokens.cc
> $(CXX) -o lang lang-tokens.cc lang-parser.cc

clean:
> $(RM) lang.cxx lang lang-parser.cc \
lang-parser.hh lang-tokens.cc \


.PHONY: clean
