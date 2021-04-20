CC ?= gcc
CXX ?= gcc++

CFLAGS := -fdiagnostics-color=always -Wall -Wextra -Wshadow -Wcast-align -Wunused -Wpedantic -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wvla
CFLAGS += -Wno-unused-parameter -Wno-unknown-pragmas
CFLAGS += -std=c17

ifneq (${PROFILE}, 1)
	#CFLAGS += -fsanitize=address,leak
	#LDFLAGS += -fsanitize=address,leak
	#CFLAGS += -fsanitize=thread
	#CFLAGS += -fsanitize=memory
endif


ifeq ($(strip DEBUG), 0)
	# perf setting
	CFLAGS += -O3 -DNDEBUG
else
	# debug setting
	CFLAGS += -O0 -g
endif

ifeq (${PROFILE}, 1)
	# gcc is known to optmize better
	CC = gcc
	CXX = g++
	# when profiling this will not print
	CFLAGS += -O3 -flto -DNDEBUG -g  -DPROFILE
endif
