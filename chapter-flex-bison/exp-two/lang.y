%{
  #include <iostream>
  #include <cstdlib>

  extern int yylex();
  extern int yyparse();
  extern FILE *yyin;

  void yyerror(const char* s);
%}

// Symbol declaration

%union {
  int ival;
  std::string *sval;
}

%token <ival> INT
%token <sval> STRING

// Rule declaration

%%

program : INT program { std::cout << "Int : " << $1 << std::endl; }
        | STRING program { std::cout << "String : "<< *($1) << std::endl; }
        | INT { std::cout << "Int : " << $1 << std::endl; }
        | STRING { std::cout << "String : "<< *($1) << std::endl; }
        ;
%%

int main(int, char**) {
  yyparse();
}

void yyerror(const char* s) {
  std::cout << "Parse error: " << s << std::endl;
  exit(-1);
}
