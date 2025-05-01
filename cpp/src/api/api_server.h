#ifndef STUCKPREDICTION_API_SERVER_H
#define STUCKPREDICTION_API_SERVER_H

// src/api/api_server.h
// TODO: Add class functionality for api_server

#include <string>

class api_server {
public:
    api_server();
    ~api_server();
    void start_server();
    void handle_prediction_request();
private:
    // TODO: Add private members and methods
};

#endif // STUCKPREDICTION_API_SERVER_H