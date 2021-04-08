CC ?= clang
CXX ?= clang++

CFLAGS := -fdiagnostics-color=always -Wall -Wextra -Wshadow -Wcast-align -Wunused -Wpedantic -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wvla
CFLAGS += -Wno-unused-parameter -Wno-unknown-pragmas
CFLAGS += -std=c17

ifneq (${PROFILE}, 1)
	CFLAGS += -fsanitize=address,leak
	LDFLAGS += -fsanitize=address,leak
	#CFLAGS += -fsanitize=thread
	#CFLAGS += -fsanitize=memory
endif


ifeq (${DEBUG}, 0)
	# perf setting
	CFLAGS += -O3 -flto -DNDEBUG
else
	# debug setting
	CFLAGS += -O0 -g
endif

ifeq (${PROFILE}, 1)
	# when profiling this will not print
	CFLAGS += -O3 -flto -DNDEBUG -g
	CFLAGS += -DPROFILE
endif
