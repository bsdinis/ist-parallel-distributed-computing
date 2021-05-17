-include ../buildenv.mk

vpath # clear

INCLUDES := -I.
SOURCES=$(wildcard *.c)
HEADERS=$(wildcard *.h)
OBJECTS=$(SOURCES:.c=.o)

CFLAGS += $(INCLUDES)
LDLIBS += -lm
LDLIBS += -fopenmp # -lomp
#LDLIBS += -lmpi

CFLAGS += -openmp

all: ballQuery ballQuery_pipe ballAlg ballAlg-omp ballAlg_n2 ballAlg_centroid

-include autodep

ballAlg: ballAlg.o
ballAlg_n2: ballAlg_n2.o
ballAlg_centroid: ballAlg_centroid.o
ballAlg_random: ballAlg_random.o

ballQuery: ballQuery.c
	$(CC) -O3 -g -DNEBUG $^ -lm -o $@

ballQuery_pipe: ballQuery_pipe.c
	$(CC) -O3 -g -DNEBUG $^ -lm -o $@

ballAlg-omp: ballAlg-omp.c
	$(CC) -openmp $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@
#	rm opari.rc
#	kinst-ompp $(CC) -openmp $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

ballAlg-mpi: ballAlg-mpi.c

.PHONY: fmt
fmt:
	@clang-format -i -style=file $(SOURCES) $(HEADERS)

.PHONY: tidy
tidy:
	@clang-tidy $(wildcard ballAlg*.c) $(HEADERS)

.PHONY: clean
clean:
	@rm -f ballAlg ballAlg_random ballAlg_n2 ballAlg_centroid ballQuery_pipe ballQuery ballAlg-omp ballAlg-mpi $(OBJECTS)
	@rm -f *opari* *.mod.c

depend : $(SOURCES)
	$(CC) $(INCLUDES) -MM $(SOURCES) > autodep
	@echo $(SOURCES)
