using System;

namespace NeuralNetwork
{
    public class Program
    {
        public static void Main(string[] args)
        {
            // Create a new neural network with 4 input nodes (spike length, spot size, color, ripeness) and 10 hidden nodes
            NeuralNetwork nn = new NeuralNetwork(4, 10, NeuralNetwork.ActivationFunction.Sigmoid);

            
            // Define input and output data for 250 fruits
            double[][] inputs = new double[250][];
            double[][] outputs = new double[250][];

            // Generate random data for 250 fruits
            GenerateRandomData(inputs, outputs, 250);


            // Train the neural network
            nn.Train(inputs, outputs, 1000);

            // Define a range of fruits to predict
            double[][] fruitsToPredict = new double[][]
            {
                new double[] { 1, 0.65, 128, 0.75 }, // Fruit with color 128 and ripeness 0.75
                new double[] { 0.10, 0.08, 200, 0.40 }, // Fruit with color 200 and ripeness 0.40
                new double[] { 0.50, 0.30, 100, 0.60 }  // Fruit with color 100 and ripeness 0.60
            };

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
                Console.WriteLine($"\nPredicting fruit {i + 1}:");
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
        
        private static void GenerateRandomData(double[][] inputs, double[][] outputs, int numFruits)
        {
            // Generate random data for the specified number of fruits
            Random rand = new Random();
            for (int i = 0; i < numFruits; i++)
            {
                // Generate random characteristics for each fruit
                double spikeLength = Math.Round(rand.NextDouble(), 2); // Spike length between 0 and 1
                double spotSize = Math.Round(rand.NextDouble(), 2); // Spot size between 0 and 1
                int color = rand.Next(256); // Color value between 0 and 255
                double ripeness = Math.Round(rand.NextDouble(), 2); // Ripeness between 0 and 1

                // Store the characteristics in the input array
                inputs[i] = new double[] { spikeLength, spotSize, color, ripeness };
                // Generate the output data based on the spike length, spot size, color, and ripeness
                outputs[i] = GenerateOutputData(spikeLength, spotSize, color, ripeness);
            }
        }
        
        private static double[] GenerateOutputData(double spikeLength, double spotSize, double color, double ripeness)
        {
            // Determine if the fruit is poisonous based on spike length, spot size, color, and ripeness
            // For example, if spike length is more than 0.75, spot size is less than 0.3, color is dark (less than 100), and ripeness is high (more than 0.65), mark it as poisonous
            // Otherwise, mark it as non-poisonous
            var poisonousStatus = (spikeLength > 0.5 && spotSize < 0.3 && color < 100 && ripeness > 0.65) ? 1 : 0;
            return new double[] { poisonousStatus };
        }
    }
}
