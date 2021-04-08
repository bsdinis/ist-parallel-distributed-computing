-include ../buildenv.mk

vpath # clear
vpath ../lib/%.h

INCLUDES := -I. -I../lib
SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o)

CFLAGS += $(INCLUDES)
LDFLAGS += -L../lib
LDLIBS += -lballalg
LDLIBS += -lm
LDLIBS += -fopenmp -lomp
#LDLIBS += -lmpi

all: ballQuery ballQuery_pipe ballAlg ballAlg_n2 ballAlg_centroid

../lib/libballalg.a:
	make -C ../lib

-include autodep

ballAlg: ballAlg.o ../lib/libballalg.a
ballAlg_n2: ballAlg_n2.o ../lib/libballalg.a
ballAlg_centroid: ballAlg_centroid.o ../lib/libballalg.a
ballAlg_random: ballAlg_random.o ../lib/libballalg.a

ballQuery: ballQuery.c
	$(CC) -O3 -g -DNEBUG -fsanitize=address -lm $^ -o $@

ballQuery_pipe: ballQuery_pipe.c
	$(CC) -O3 -g -DNEBUG -fsanitize=address -lm $^ -o $@

ballAlg-omp: ballAlg-omp.c
ballAlg-mpi: ballAlg-mpi.c

.PHONY: fmt
fmt:
	@clang-format -i -style=file *.c

.PHONY: tidy
tidy:
	@clang-tidy $(SOURCES)

.PHONY: clean
clean:
	@rm -f ballAlg ballAlg_x ballAlg_md_n2 ballAlg_centroid ballQuery_pipe ballQuery ballAlg-omp ballAlg-mpi

depend : $(SOURCES)
	$(CC) $(INCLUDES) -MM $(SOURCES) > autodep
	@echo $(SOURCES)
	make -C ../lib depend
