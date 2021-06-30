%{
  #include <iostream>
  #include "lang-parser.hh"

%}
%option noyywrap
%%
[ \t]           ;
[0-9]+            { yylval.ival = std::stoi(yytext); return INT; }
[a-zA-Z]+         { yylval.sval = new std::string(yytext, yyleng); return STRING; }
%%
