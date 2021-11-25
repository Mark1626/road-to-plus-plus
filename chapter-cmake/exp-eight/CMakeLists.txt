project(Exp8)

cmake_minimum_required(VERSION 3.12)

execute_process(
    COMMAND "echo Hello"
    RESULT_VARIABLE HAD_ERROR
    OUTPUT_VARIABLE CONFIG_OUTPUT
)