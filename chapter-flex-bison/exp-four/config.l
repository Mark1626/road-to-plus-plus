%{
  #include <string>
  #include "parser.hh"
%}

%option noyywrap

%%
[ \t]                      ;
"Name:"                    { yylval.token = T_NAME; return T_NAME; }
[a-zA-Z_][a-zA-Z0-9]+      { yylval.sval = new std::string(yytext, yyleng); return T_STRING; }
%%
