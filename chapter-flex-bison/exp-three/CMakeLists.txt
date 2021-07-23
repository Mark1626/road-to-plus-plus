cmake_minimum_required(VERSION 3.20)

project(Lang VERSION 1.0)

find_package(BISON)
find_package(FLEX)

BISON_TARGET(parser lang.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp)
FLEX_TARGET(lexer lang.l ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)

ADD_FLEX_BISON_DEPENDENCY(lexer parser)

add_executable(lang ${BISON_parser_OUTPUTS} ${FLEX_lexer_OUTPUTS})
