namespace NeuralNetwork;

using System;

public class Program
{
    public static void Main(string[] args)
    {
        // Create a new neural network with 2 input nodes (spike length and spot size) and 8 hidden nodes
        NeuralNetwork nn = new NeuralNetwork(2, 8);

        // Define input and output data
        // spike length, spot size for fruit
        double[][] inputs = new double[][] {
            new double[] { 1.2, 0.5 },
            new double[] { 0.8, 0.7 },
            new double[] { 1.0, 0.4 },
            new double[] { 1.2, 0.5 },
            new double[] { 0.8, 0.7 },
            new double[] { 1.0, 0.4 },
            new double[] { 0.9, 0.6 },
            new double[] { 1.1, 0.5 },
            new double[] { 0.7, 0.8 },
            new double[] { 1.3, 0.4 },
            new double[] { 0.6, 0.7 },
            new double[] { 1.0, 0.6 },
            new double[] { 0.8, 0.5 },
        };

        // 1 if poisonous, 0 if not
        double[][] outputs = new double[][] {
            new double[] { 1 },
            new double[] { 0 },
            new double[] { 1 },
            new double[] { 1 },
            new double[] { 0 },
            new double[] { 1 },
            new double[] { 0 },
            new double[] { 1 },
            new double[] { 0 },
            new double[] { 1 },
            new double[] { 0 },
            new double[] { 1 },
            new double[] { 0 },

        };

        // Train the neural network
        nn.Train(inputs, outputs, 10000);

        // Make a prediction for a new fruit with spike length 1.1 and spot size 0.6
        double[] prediction = nn.FeedForward(new double[] { 1.1, 0.6 });

        // Print the raw output of the neural network
        Console.WriteLine($"The neural network output is: {prediction[0]}");

        // Interpret the output as a probability
        string predictionResult = prediction[0] > 0.5 ? "Poisonous" : "Not poisonous";
        Console.WriteLine($"This means the fruit is {prediction[0] * 100}% likely to be poisonous, and {100 - prediction[0] * 100}% likely to be not poisonous.");
        // Calculate and print the error
        double error = outputs[0][0] - prediction[0];
        Console.WriteLine($"The squared error between the predicted value and the actual value is: {Math.Pow(error, 2)}");
    }
}