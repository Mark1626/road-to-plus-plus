%{
  #include <iostream>
  #include <string>

  class Config {
  public:
    std::string name;
    Config(std::string &name) : name(name){};
  };


  Config *config;

  extern int yylex();
  extern int yyparse();

  void yyerror(const char* s);
%}

%union {
  std::string *sval;
  int token;
}

%token <token> T_NAME
%token <sval> T_STRING

%start program

%%

program : T_NAME T_STRING { config = new Config(*$2); }
        ;

%%

void yyerror(const char* s) {
  std::cout << "Parse error: " << s << std::endl;
  exit(-1);
}

int main() {
  yyparse();
  std::cout << config->name << "\n";
}
