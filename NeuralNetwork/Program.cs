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
            double[][] inputs = new double[1000][];
            double[][] outputs = new double[1000][];

            // Generate random data for 250 fruits
            GenerateRandomData(inputs, outputs, 1000);


            // Train the neural network
            nn.Train(inputs, outputs, 1000, 0.0);

            // Define a range of fruits to predict
            List<double[]> fruitsToPredictList = new List<double[]>();
            Random rand = new Random();

            for (int i = 0; i < 10; i++) 
            {
                fruitsToPredictList.Add(GenerateNonPoisonousData(rand));
                fruitsToPredictList.Add(GeneratePoisonousData(rand));
            }

            double[][] fruitsToPredict = fruitsToPredictList.ToArray();

            // Make predictions for the range of fruits
            PredictFruits(nn, fruitsToPredict);
            
            // Make predictions for all 250 fruits
            //double[] predictedOutputs = PredictFruits2(nn, inputs);

            // Generate the confusion matrix
            //double[] actualOutputs = outputs.Select(output => output[0]).ToArray();
            //GenerateConfusionMatrix(actualOutputs, predictedOutputs);
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
        
        private static void GenerateRandomData(double[][] inputs, double[][] outputs, int numFruits)
        {
            // Generate random data for the specified number of fruits
            Random rand = new Random();
            int numPoisonous = (int)(numFruits * 0.5);
            int numNonPoisonous = (int)(numFruits * 0.5);

            for (int i = 0; i < numPoisonous; i++)
            {
                // Generate random poisonous fruit
                inputs[i] = GeneratePoisonousData(rand);

                var spikeLength = inputs[i][0];
                var spotSize = inputs[i][1];
                var color = inputs[i][2];
                var ripeness = inputs[i][3];

                outputs[i] = GenerateOutputData(spikeLength, spotSize, color, ripeness);
            }
            for (int i = numPoisonous; i < numPoisonous + numNonPoisonous; i++)
            {
                // Generate random non-poisonous fruit
                inputs[i] = GenerateNonPoisonousData(rand);

                var spikeLength = inputs[i][0];
                var spotSize = inputs[i][1];
                var color = inputs[i][2];
                var ripeness = inputs[i][3];

                outputs[i] = GenerateOutputData(spikeLength, spotSize, color, ripeness);
            }
        }
        
        private static double[] GeneratePoisonousData(Random rand)
        {
            // Generate random characteristics for each fruit
            double spikeLength = Math.Round(rand.NextDouble() * 0.49, 2) + 0.51; // Spike length between 0.51 and 1
            double spotSize = Math.Round(rand.NextDouble() * 0.29, 2); // Spot size between 0 and 0.29
            int color = rand.Next(100); // Color value between 0 and 99
            double ripeness = Math.Round(rand.NextDouble() * 0.64, 2); // Ripeness between 0 and 0.64

            // Store the characteristics in the input array
            var inputs = new double[] { spikeLength, spotSize, color, ripeness };
            return inputs;
        }

        private static double[] GenerateNonPoisonousData(Random rand)
        {
            // Generate random characteristics for each fruit
            double spikeLength = Math.Round(rand.NextDouble()*0.50, 2); // Spike length between 0 and 1
            double spotSize = Math.Round(rand.NextDouble()*0.7, 2)+0.3; // Spot size between 0 and 1
            int color = rand.Next(155)+100; // Color value between 0 and 255
            double ripeness = Math.Round(rand.NextDouble()*0.35, 2)*0.65; // Ripeness between 0 and 1

            // Store the characteristics in the input array
            var inputs = new double[] { spikeLength, spotSize, color, ripeness };
            return inputs;
        }

        private static double[] GenerateOutputData(double spikeLength, double spotSize, double color, double ripeness, bool? forcePoisonous = null)
        {
            // Determine if the fruit is poisonous based on spike length, spot size, color, and ripeness
            // For example, if spike length is more than 0.75, spot size is less than 0.3, color is dark (less than 100), and ripeness is high (more than 0.65), mark it as poisonous
            // Otherwise, mark it as non-poisonous
            var poisonousStatus = 0;
            /*
            if (forcePoisonous.HasValue)
            {
                poisonousStatus = forcePoisonous.Value ? 1 : 0;
            }
            else
            */
            {
                poisonousStatus = (spikeLength > 0.5 && spotSize < 0.3 && color < 100 && ripeness < 0.65) ? 1 : 0;
            }
            return new double[] { poisonousStatus };
        }
        
        /*private static double[] GenerateOutputData(double spikeLength, double spotSize, double color, double ripeness)
        {
            // Determine if the fruit is poisonous based on spike length, spot size, color, and ripeness
            // For example, if spike length is more than 0.75, spot size is less than 0.3, color is dark (less than 100), and ripeness is high (more than 0.65), mark it as poisonous
            // Otherwise, mark it as non-poisonous
            var poisonousStatus = (spikeLength > 0.5 && spotSize < 0.3 && color < 100 && ripeness < 0.65) ? 1 : 0;
            return new double[] { poisonousStatus };
        }*/
        
        private static (double[], double[]) PredictFruits2(NeuralNetwork nn, double[][] fruits, double[][] actualOutputs)
        {
            double[] predictions = new double[fruits.Length];
            double[] actuals = new double[fruits.Length];
            for (int i = 0; i < fruits.Length; i++)
            {
                var fruit = fruits[i];
                double[] prediction = nn.FeedForward(fruit);
                predictions[i] = prediction[0];
                actuals[i] = actualOutputs[i][0];
            }
            return (predictions, actuals);
        }
        
        private static void GenerateConfusionMatrix(double[] actual, double[] predicted)
        {
            int truePositives = 0;
            int trueNegatives = 0;
            int falsePositives = 0;
            int falseNegatives = 0;

            for (int i = 0; i < actual.Length; i++)
            {
                if (actual[i] == 1 && predicted[i] >= 0.5)
                    truePositives++;
                else if (actual[i] == 0 && predicted[i] < 0.5)
                    trueNegatives++;
                else if (actual[i] == 0 && predicted[i] >= 0.5)
                    falsePositives++;
                else if (actual[i] == 1 && predicted[i] < 0.5)
                    falseNegatives++;
            }

            Console.WriteLine($"Confusion Matrix:");
            Console.WriteLine($"TP: {truePositives}\tFP: {falsePositives}");
            Console.WriteLine($"FN: {falseNegatives}\tTN: {trueNegatives}");
        }
    }
}
