BISON_TARGET(parser lang.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp)
FLEX_TARGET(lexer lang.l ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

ADD_FLEX_BISON_DEPENDENCY(lexer parser)

add_executable(lang ${BISON_parser_OUTPUTS} ${FLEX_lexer_OUTPUTS})
