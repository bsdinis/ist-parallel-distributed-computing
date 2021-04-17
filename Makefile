all: ballAlg ballAlg_n2 ballAlg_centroid ballAlg_random ballQuery ballQuery_pipe ballAlg-omp

.PHONY: ballAlg
ballAlg:
	make -C src ballAlg
	cp src/ballAlg .

.PHONY: ballAlg_n2
ballAlg_n2:
	make -C src ballAlg_n2
	cp src/ballAlg_n2 .

.PHONY: ballAlg_centroid
ballAlg_centroid:
	make -C src ballAlg_centroid
	cp src/ballAlg_centroid .

.PHONY: ballAlg_random
ballAlg_random:
	make -C src ballAlg_random
	cp src/ballAlg_random .

.PHONY: ballQuery
ballQuery:
	make -C src ballQuery
	cp src/ballQuery .

.PHONY: ballQuery_pipe
ballQuery_pipe:
	make -C src ballQuery_pipe
	cp src/ballQuery_pipe .

.PHONY: ballAlg-omp
ballAlg-omp:
	make -C src ballAlg-omp
	cp src/ballAlg-omp .

.PHONY: ballAlg-mpi
ballAlg-mpi:
	make -C src ballAlg-mpi
	cp src/ballAlg-mpi .

test:
	ls tests | grep _s | xargs ./sbin/test.sh

test-omp:
	ls tests | grep _s | xargs ./sbin/test-omp.sh

all_tests:
	./sbin/test.sh

.PHONY: fmt
fmt:
	make -C src fmt

.PHONY: tidy
tidy:
	make -C src tidy

.PHONY: perf
perf:
	perf -F 99 --cal-graph dwarf

.PHONY: clean
clean:
	make -C src clean
	rm -f ballQuery ballQuery_pipe ballAlg ballAlg_n2 ballAlg_centroid ballAlg_random ballAlg_x ballAlg-omp
