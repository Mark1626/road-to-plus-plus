%{
  #include <iostream>

  extern int yylex();

  int sum_int = 0;
  float sum_float = 0;

%}
%option noyywrap
%%
[\t\n]            ;
[0-9]+\.[0-9]+    { 
                    std::cout << "Float " << yytext << std::endl;
                    sum_float += std::stod(yytext);
                  }
[0-9]+            {
                    std::cout << "Int " << yytext << std::endl;
                    sum_int += std::stoi(yytext);
                  }
[a-zA-Z]          ;
%%
int main(int argc, char** argv) {
  // Change this to update source
  // yyin = 
  while(yylex());

  std::cout << "Sum of all floats " << sum_float << std::endl;
  std::cout << "Sum of all ints " << sum_int << std::endl;
}
