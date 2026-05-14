cmake_minimum_required(VERSION 3.26)
project(ecodag LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:/msys64/mingw64")

include_directories(${CMAKE_SOURCE_DIR}/src)
find_package(OpenMP)


# Main executable
file(GLOB_RECURSE MAIN_SOURCES
	"main.cpp"
)
add_executable(ecodag ${MAIN_SOURCES})
target_compile_options(ecodag PRIVATE -Wall -Wextra -O2)
if(OpenMP_CXX_FOUND)
	target_link_libraries(ecodag PRIVATE OpenMP::OpenMP_CXX)
endif()
