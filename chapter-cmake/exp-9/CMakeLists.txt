project(Exp9)

cmake_minimum_required(VERSION 3.12)

execute_process(
    COMMAND "pwd"
    RESULT_VARIABLE HAD_ERROR
    OUTPUT_VARIABLE CONFIG_OUTPUT
)

set(RESULT ${CONFIG_OUTPUT} CACHE PATH "Path to dir")
message("Value is ${RESULT}")
