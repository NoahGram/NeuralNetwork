namespace NeuralNetwork;

using System;

public class Program
{
    public static void Main(string[] args)
    {
        // Create a new neural network with 4 input nodes and 2 hidden nodes
        NeuralNetwork nn = new NeuralNetwork(4, 6);

        // Define input and output data
        double[][] inputs = new double[][] {
            new double[] { 0, 0, 1, 1 },
            new double[] { 1, 1, 1, 0 },
            new double[] { 0, 1, 0, 1 },
            new double[] { 1, 0, 1, 0 },
            new double[] { 0, 1, 1, 0 }
        };
        double[][] outputs = new double[][] {
            new double[] { 1 },
            new double[] { 1 },
            new double[] { 0 },
            new double[] { 0 },
            new double[] { 1 }
        };

        // Train the neural network
        nn.Train(inputs, outputs, 10000);

        // Make a prediction
        double[] prediction = nn.FeedForward(new double[] { 0, 0, 1, 1 });

        // Print the prediction
        Console.WriteLine("Prediction: " + prediction[0]);
        
        // Calculate and print the error
        double error = outputs[0][0] - prediction[0];
        Console.WriteLine("Error: " + Math.Pow(error, 2));
    }
}