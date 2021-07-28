
include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

BISON_TARGET(parser config.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cc)
FLEX_TARGET(lexer config.l ${CMAKE_CURRENT_BINARY_DIR}/lexer.cc)

ADD_FLEX_BISON_DEPENDENCY(lexer parser)

add_executable(config ${BISON_parser_OUTPUTS} ${FLEX_lexer_OUTPUTS})
