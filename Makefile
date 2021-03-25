CC ?= clang

CFLAGS = -Wall -Wextra -Wshadow -Wcast-align -Wunused -Wpedantic -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wvla
CFLAGS += -Wno-unused-function -Wno-unused-parameter
#CFLAGS += -std=c++2a

CFLAGS += -std=c18
CFLAGS += -fsanitize=address,leak
#CFLAGS += -fsanitize=thread
#CFLAGS += -fsanitize=memory

LDFLAGS +=  -fopenmp -lomp # -lmpi

ifeq (${DEBUG}, 0)
	# perf setting
	CFLAGS += -O3 -flto -DNDEBUG
	TARGET_DIR=target/release
else
	# debug setting
	CFLAGS += -O0 -g
	TARGET_DIR=target/debug
endif

all: serial move_objects

.PHONY: serial
serial: ballAlg

.PHONY: openmp
openmp: ballAlg-omp

.PHONY: mpi
mpi: ballAlg-mpi

ballAlg: ballAlg.c
ballAlg-omp: ballAlg-omp.o
ballAlg-mpi: ballAlg-mpi.o



.PHONY: move_objects
move_objects:
	@mv ballAlg ${TARGET_DIR}/ballAlg

   #@mv ballAlg-omp ${TARGET_DIR}/ballAlg-omp
   #@mv ballAlg-mpi ${TARGET_DIR}/ballAlg-mpi

.PHONY: fmt
fmt:
	@clang-format -i -style=file *.c

.PHONY: tidy
tidy:
	@clang-tidy *.c

.PHONY: clean
clean:
	@rm -f target/release/* target/debug/*
