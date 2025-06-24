# Makefile 

# Compilador y flags
CXX := g++
CXXFLAGS := -c -fPIC -std=c++17 -Ieigen-3.4.0  # Add Eigen include path and C++17 standard
LDFLAGS := -shared

# Directorios
BUILD_DIR := build

# Fuentes y objetos
SOURCES := exponents/exponents.cpp exponents/functions.cpp
OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(notdir $(SOURCES)))

# Objetivo principal
all: $(BUILD_DIR)/libfunctions.so

# Regla para la biblioteca
$(BUILD_DIR)/libfunctions.so: $(OBJECTS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(LDFLAGS) -o $@ $^

# Regla genérica para objetos
$(BUILD_DIR)/%.o: exponents/%.cpp exponents/functions.h
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

# Limpieza
clean:
	rm -rf $(BUILD_DIR)