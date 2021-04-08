-include ../buildenv.mk

SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o)

all: libballalg.a

-include autodep

libballalg.a: $(OBJECTS)
	ar rcs $@ $^

.PHONY: fmt
fmt:
	@clang-format -i -style=file $(SOURCES)

.PHONY: tidy
tidy:
	@clang-tidy $(SOURCES)

.PHONY: clean
clean:
	@rm -f $(OBJECTS) libballalg.a

depend : $(SOURCES)
	$(CC) $(INCLUDES) -MM $(SOURCES) > autodep
	@echo $(SOURCES)
