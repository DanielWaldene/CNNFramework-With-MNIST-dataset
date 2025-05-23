BIN_DIR:=bin
BUILD_DIR := build
SRC_DIR :=src
TARGET := $(BIN_DIR)/main
OBJS := $(BUILD_DIR)/main.o  $(BUILD_DIR)/utility.o $(BUILD_DIR)/cnn.o $(BUILD_DIR)/network.o
$(TARGET):$(OBJS)
	g++ -Wall -o $@ $(OBJS)
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp
	g++ -Wall -c $< -o $@
$(BUILD_DIR)/utility.o: $(SRC_DIR)/utility.cpp $(SRC_DIR)/utility.hpp
	g++ -Wall  -c $< -o $@
$(BUILD_DIR)/cnn.o: $(SRC_DIR)/cnn.cpp $(SRC_DIR)/cnn.hpp
	g++ -Wall  -c $< -o $@
$(BUILD_DIR)/network.o: $(SRC_DIR)/network.cpp $(SRC_DIR)/network.hpp
	g++ -Wall  -c $< -o $@
clean:
	rm -rf $(BIN_DIR)/*
	rm -rf $(BUILD_DIR)/*.o
run: 
	./$(TARGET)