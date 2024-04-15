namespace NeuralNetwork;

using System;

public class Program
{
    public static void Main(string[] args)
    {
        // Create a new neural network
        NeuralNetwork nn = new NeuralNetwork();

        // Define input and output data
        double[][] inputs = new double[][] {
            new double[] { 0, 0, 1, 1 },
            new double[] { 1, 1, 1, 0 },
            new double[] { 0, 1, 0, 1 },
            new double[] { 1, 0, 1, 0 },
            new double[] { 0, 1, 1, 0 }
        };
        double[] outputs = new double[] { 1, 1, 0, 0, 1 };

        // Train the neural network
        nn.Train(inputs, outputs, 10000);

        // Make a prediction
        double[] prediction = nn.FeedForward(new double[] { 0, 0, 1, 1 });

        // Print the prediction
        Console.WriteLine(prediction[0]);
    }
}