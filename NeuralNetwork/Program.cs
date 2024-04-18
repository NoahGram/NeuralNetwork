using System;

namespace NeuralNetwork
{
    public abstract class Program
    {
        public static void Main(string[] args)
        {
            // Create a new neural network with 4 input nodes (spike length, spot size, color, ripeness) and 10 hidden nodes
            NeuralNetwork nn = new NeuralNetwork(4, 14, NeuralNetwork.ActivationFunction.Sigmoid);
            
            // Define input and output data for 250 fruits
            double[][] inputs = new double[5000][];
            double[][] outputs = new double[5000][];

            // Generate random data for 250 fruits
            FruitGeneration.GenerateRandomData(inputs, outputs, 5000);

            // Train the neural network
            nn.Train(inputs, outputs, 1000, 0.0);

            // Define a range of fruits to predict
            List<double[]> fruitsToPredictList = new List<double[]>();
            Random rand = new Random();

            for (int i = 0; i < 5; i++) 
            {
                fruitsToPredictList.Add(FruitGeneration.GenerateNonPoisonousData(rand));
                fruitsToPredictList.Add(FruitGeneration.GeneratePoisonousData(rand));
            }

            double[][] fruitsToPredict = fruitsToPredictList.ToArray();

            // Make predictions for the range of fruits
            PredictFruits(nn, fruitsToPredict);
        }
        
        // Predict the probability of a fruit being poisonous based on its characteristics
        private static void PredictFruits(NeuralNetwork nn, double[][] fruits, double[][] outputs = null)
        {
            Console.WriteLine($"\nThe neural network is using the {nn.GetActivationFunction()} activation function.");

            for (int i = 0; i < fruits.Length; i++)
            {
                Console.WriteLine("\n-----------------------------------");
                Console.WriteLine($"Predicting fruit {i + 1}:");
                Console.WriteLine("-----------------------------------");
                var fruit = fruits[i];
                // Make a prediction for the fruit
                double[] prediction = nn.FeedForward(fruit);

                // Print the raw output of the neural network
                Console.WriteLine($"\nThe neural network output for fruit with spike length {fruit[0]}, spot size {fruit[1]}, color {fruit[2]}, and ripeness {fruit[3]} is: {prediction[0]}");

                // Interpret the output as a probability
                Console.WriteLine($"This means the fruit is {prediction[0] * 100}% likely to be poisonous, and {100 - prediction[0] * 100}% likely to be not poisonous.");

                // Calculate and print the error if outputs are provided
                if (outputs != null)
                {
                    double error = outputs[i][0] - prediction[0];
                    Console.WriteLine($"The squared error between the predicted value and the actual value is: {Math.Pow(error, 2)}");
                }
            }
        }
    }
}
