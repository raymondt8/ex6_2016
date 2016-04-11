project(poisson)

cmake_minimum_required(VERSION 2.6)

enable_language(C)
#enable_language(Fortran)

# CMake Options to enable/disable MPI and OpenMP
option(ENABLE_OPENMP "Enable OpenMP support?" On)
option(ENABLE_MPI    "Enable MPI support?" On)#ON)

if(ENABLE_MPI)
  find_package(MPI)
endif()

if(MPI_FOUND)
  add_definitions(-DHAVE_MPI=1)
  set(INCLUDES ${INCLUDES} ${MPI_INCLUDE_PATH})
  set(DEPLIBS_C ${DEPLIBS_C} ${MPI_C_LIBRARIES})
  #set(DEPLIBS_F ${DEPLIBS_F} ${MPI_Fortran_LIBRARIES})
endif()

if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    add_definitions(-DHAVE_OPENMP=1)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  endif()
endif()

# Send the include directories to the compiler
include_directories(${INCLUDES})

#add_definitions(-std=c99 -lm)
add_definitions(-Wextra -Wall -msse4.1 -pedantic -g -O3  -std=c99 -L/path/to/libs)
#set(CMAKE_C_CLAGS "${CMAKE_C_FLAGS} -Wextra -Wall -pedantic -O3 -std=c99 -g")

# Add a library with common sources
#add_library(common fst.f)

# The C based poisson application
add_executable(poisson main.c parallelPoisson.c parallelPoisson.h serialPoisson.c serialPoisson.h)

# The Fortran based poisson application
#add_executable(poisson-f poisson.f90)

# Link to libraries
target_link_libraries(poisson m fst ${DEPLIBS_C})
link_directories(${CMAKE_SOURCE_DIR})

# We want to link this as a C program
set_target_properties(poisson PROPERTIES LINKER_LANGUAGE C)

# Link to libraries
#target_link_libraries(poisson-f common ${DEPLIBS_F})
