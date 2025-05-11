package com.example.gui;

import com.example.client.BackendClient;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;

/**
 * GUI class for interacting with the stock prediction backend.
 */
public class StockPredictionGUI extends JFrame {

    private final JTextField inputField;
    private final JTextArea resultArea;
    private final BackendClient backendClient;

    public StockPredictionGUI() {
        super("Stock Prediction App");
        this.backendClient = new BackendClient();

        inputField = new JTextField(20);
        JButton submitButton = new JButton("Predict");
        resultArea = new JTextArea(10, 30);
        resultArea.setEditable(false);

        submitButton.addActionListener(this::handleSubmit);

        setLayout(new BorderLayout());
        JPanel topPanel = new JPanel();
        topPanel.add(new JLabel("Stock Symbol:"));
        topPanel.add(inputField);
        topPanel.add(submitButton);

        add(topPanel, BorderLayout.NORTH);
        add(new JScrollPane(resultArea), BorderLayout.CENTER);

        setDefaultCloseOperation(EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null); // Center the window
    }

    private void handleSubmit(ActionEvent e) {
        String symbol = inputField.getText().trim();
        if (symbol.isEmpty()) {
            showError("Please enter a stock symbol.");
            return;
        }

        try {
            String prediction = backendClient.getPrediction(symbol);
            resultArea.setText(prediction);
        } catch (Exception ex) {
            showError("Error retrieving prediction: " + ex.getMessage());
        }
    }

    private void showError(String message) {
        JOptionPane.showMessageDialog(this, message, "Error", JOptionPane.ERROR_MESSAGE);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new StockPredictionGUI().setVisible(true));
    }
}
