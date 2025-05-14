#ifndef STUCKPREDICTION_DATALOADER_H
#define STUCKPREDICTION_DATALOADER_H

// src/data/DataLoader.h
// TODO: Add class functionality for DataLoader

#include <string>

class DataLoader {
public:
    DataLoader();
    ~DataLoader();
    void load_data(const std::string& path);
private:
    // TODO: Add private members and methods
};

#endif // STUCKPREDICTION_DATALOADER_H
