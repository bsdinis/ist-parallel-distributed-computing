CC = gcc
CXX = gcc++

CFLAGS := -fdiagnostics-color=always
CFLAGS += -Wno-unused-parameter -Wno-unknown-pragmas
CFLAGS += -std=c11

CFLAGS += -O3 -flto -DNDEBUG -g

ifeq (${PROFILE}, 1)
	CFLAGS += -DPROFILE
endif

vpath # clear

INCLUDES := -I.
SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o)

CFLAGS += $(INCLUDES)
LDLIBS += -lm
LDLIBS += -fopenmp # -lomp
#LDLIBS += -lmpi

CFLAGS += -openmp

all: ballAlg ballAlg-omp ballAlg-mpi

ballAlg: ballAlg.o

ballAlg-omp: ballAlg-omp.c
	$(CC) -openmp $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

ballAlg-mpi: ballAlg-mpi.c
	mpicc -openmp $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

.PHONY: fmt
fmt:
	@clang-format -i -style=file $(SOURCES)

.PHONY: tidy
tidy:
	@clang-tidy $(wildcard ballAlg*.c)

.PHONY: clean
clean:
	@rm -f ballAlg ballAlg-omp ballAlg-mpi $(OBJECTS)
	@rm -f *opari* *.mod.c
