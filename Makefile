CC ?= clang
CXX ?= clang++

CFLAGS = -Wall -Wextra -Wshadow -Wcast-align -Wunused -Wpedantic -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wvla
CFLAGS += -Wno-unused-parameter -Wno-unknown-pragmas
#CFLAGS += -std=c++2a

CFLAGS += -std=gnu18
CFLAGS += -fsanitize=address,leak
#CFLAGS += -fsanitize=thread
#CFLAGS += -fsanitize=memory

LDFLAGS += -lm -fopenmp -lomp # -lmpi

ifeq (${DEBUG}, 0)
	# perf setting
	CFLAGS += -O3 -flto -DNDEBUG
else
	# debug setting
	CFLAGS += -O3 -g
endif

all: ballQuery ballQuery_pipe ballAlg

SOURCES := ballAlg.c

ballAlg: ballAlg.c

ballQuery: ballQuery.c
	$(CC) -O3 -g -DNEBUG -fsanitize=address -lm $^ -o $@

ballQuery_pipe: ballQuery_pipe.c
	$(CC) -O3 -g -DNEBUG -fsanitize=address -lm $^ -o $@

ballAlg-omp: ballAlg-omp.c
ballAlg-mpi: ballAlg-mpi.c

test:
	ls tests | grep _s | xargs ./sbin/test.sh

all_tests:
	./sbin/test.sh

.PHONY: fmt
fmt:
	@clang-format -i -style=file *.c

.PHONY: tidy
tidy:
	@clang-tidy $(SOURCES)

.PHONY: clean
clean:
	@rm -f ballAlg ballQuery_pipe ballQuery ballAlg-omp ballAlg-mpi
