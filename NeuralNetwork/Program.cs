namespace NeuralNetwork;

using System;

public class Program
{
    public static void PredictFruits(NeuralNetwork nn, double[][] fruits, double[][] outputs)
    {
        Console.WriteLine($"The neural network is using the {nn.GetActivationFunction()} activation function.");

        for (int i = 0; i < fruits.Length; i++)
        {
            var fruit = fruits[i];
            // Make a prediction for the fruit
            double[] prediction = nn.FeedForward(fruit);

            // Print the raw output of the neural network
            Console.WriteLine($"The neural network output for fruit with spike length {fruit[0]} and spot size {fruit[1]} is: {prediction[0]}");

            // Interpret the output as a probability
            Console.WriteLine($"This means the fruit is {prediction[0] * 100}% likely to be poisonous, and {100 - prediction[0] * 100}% likely to be not poisonous.");

            // Calculate and print the error
            double error = outputs[i][0] - prediction[0];
            Console.WriteLine($"The squared error between the predicted value and the actual value is: {Math.Pow(error, 2)}");
        }
    }
    
    public static void Main(string[] args)
    {
        // Create a new neural network with 2 input nodes (spike length and spot size) and 10 hidden nodes
        NeuralNetwork nn = new NeuralNetwork(2, 10, NeuralNetwork.ActivationFunction.HyperbolicTangent);

        // Define input and output data for 500 fruits
        // spike length, spot size for fruit
        double[][] inputs = new double[250][];
        double[][] outputs = new double[250][];

        // Generate random data for 500 fruits
        Random rand = new Random();
        for (int i = 0; i < 250; i++)
        {
            inputs[i] = new double[] { Math.Round(rand.NextDouble(), 2), Math.Round(rand.NextDouble(), 2) }; // spike length, spot size
            outputs[i] = new double[] { rand.Next(2) }; // 1 if poisonous, 0 if not
        }

        // Train the neural network
        nn.Train(inputs, outputs, 10000);

        // Define a range of fruits to predict
        double[][] fruitsToPredict = new double[][]
        {
            new double[] { 1.10, 0.65 },
        };

        // Make predictions for the range of fruits
        PredictFruits(nn, fruitsToPredict, outputs);

    }
}