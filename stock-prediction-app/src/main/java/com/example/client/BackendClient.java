package com.example.client;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Map;

/**
 * Client for communicating with the stock prediction backend.
 */
public class BackendClient {

    private final String backendUrl;
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;

    public BackendClient() {
        this.backendUrl = loadBackendUrl();
        this.httpClient = HttpClient.newHttpClient();
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Sends a request to the backend to get stock prediction data.
     *
     * @param symbol Stock ticker symbol
     * @return Prediction result from the backend
     * @throws IOException If an I/O error occurs
     * @throws InterruptedException If the operation is interrupted
     */
    public String getPrediction(String symbol) throws IOException, InterruptedException {
        Map<String, String> requestPayload = Map.of("symbol", symbol);
        String requestBody = objectMapper.writeValueAsString(requestPayload);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(backendUrl))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new IOException("Failed with HTTP code: " + response.statusCode());
        }

        return response.body(); // Return the raw JSON or parsed prediction
    }

    private String loadBackendUrl() {
        // Fallback for simplicity; ideally you'd use SnakeYAML or Spring for full parsing
        return "http://localhost:8080/api/predict"; // Hardcoded fallback
    }
}
